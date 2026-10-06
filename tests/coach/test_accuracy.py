"""Offline upstream parity, saved-game migration and report snapshot checks."""
import copy
import io
import json
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.pgn

from analysis.lichess_accuracy import (division, force_cp, game_accuracy, judgment,
    move_accuracy, move_metrics, phase_accuracies, round_percent, win_percent)
from analysis.game.pipeline import analyze_game
from analysis.game.summary import compact_summary
from analysis.game.performance import refresh_performance
from coach.report_performance import insert_performance_snapshot, performance_table
from coach.report_output import ReportGate
from tests.coach.fixtures import FakeEngines, ScriptedCodex


class AccuracyTests(unittest.TestCase):
    def test_upstream_accuracy_examples_and_black_start(self):
        # Cases from Lila AccuracyPercentTest.scala, with its numeric tolerances.
        cases = [([15, 15], (100, 100), 1), ([-900, -900], (10, 100), 5),
                 ([15, 900], (100, 10), 5), ([-900, 0], (10, 10), 5),
                 ([15]*20, (100, 100), 1), ([15]*20+[-900], (50, 100), 5),
                 ([15]*21+[900], (100, 50), 5), ([-50, 15]*5, (76, 76), 8),
                 ([-50, 15]*50, (76, 76), 8), ([-135, 15]*50, (54, 54), 8),
                 ([-435, 15]*50, (20, 20), 8)]
        for scores, expected, tolerance in cases:
            with self.subTest(scores=scores):
                result = game_accuracy(scores)
                for side, value in zip(('white', 'black'), expected):
                    self.assertAlmostEqual(result[side], value, delta=tolerance)
        for white in (True, False):
            self.assertIsNone(game_accuracy([], white))
            self.assertIsNone(game_accuracy([15], white))
        for scores, expected in [([900, 900], (10, 100)), ([15, -900], (100, 10)),
                                 ([900, 0], (10, 10))]:
            result = game_accuracy(scores, False)
            for side, value in zip(('black', 'white'), expected):
                self.assertAlmostEqual(result[side], value, delta=5)

    def test_zero_accuracy_harmonic_floor_and_missing_eval_alignment(self):
        self.assertEqual(move_accuracy(100, 0), 0)
        self.assertEqual(move_accuracy(20, 30), 100)
        result = game_accuracy([-1000, 1000], initial=1000)
        self.assertEqual(result, {'white': .5, 'black': .5})
        rows = move_metrics([None, -100, 50, 100])
        self.assertEqual([r['side'] for r in rows], ['white', 'black', 'white', 'black'])
        self.assertEqual([r['centipawn_loss'] for r in rows], [None, None, 0, 50])
        self.assertIsNone(game_accuracy([None, 100]))
        self.assertEqual(round_percent(82.5), 83)

    def test_advice_thresholds_are_5_10_15_percent_not_accuracy_loss(self):
        import math
        for drop, expected in [(4.99, None), (5.01, 'inaccuracy'), (9.99, 'inaccuracy'),
                               (10.01, 'mistake'), (14.99, 'mistake'), (15.01, 'blunder')]:
            cp = math.log((50-drop)/(50+drop))/.00368208
            for white in (True, False):
                self.assertEqual(judgment(0, cp if white else -cp, white), expected)
        # Advice uses unceiled chances; game/phase accuracy caps at +/-1000 CP.
        self.assertEqual(win_percent(100000), win_percent(1000))
        self.assertIsNone(judgment(2000, 1000))

    def test_mates_acpl_and_black_perspective(self):
        for white in (True, False):
            sign = 1 if white else -1
            loss_mate, win_mate = ('#-2', '#2') if white else ('#2', '#-2')
            for cp, label in [(-1000, 'inaccuracy'), (-999, 'mistake'),
                              (-701, 'mistake'), (-700, 'blunder')]:
                self.assertEqual(judgment(cp*sign, loss_mate, white), label)
            for cp, label in [(1000, 'inaccuracy'), (999, 'mistake'),
                              (701, 'mistake'), (700, 'blunder')]:
                self.assertEqual(judgment(win_mate, cp*sign, white), label)
            self.assertEqual(judgment(win_mate, loss_mate, white), 'blunder')
            self.assertIsNone(judgment(loss_mate, loss_mate, white))
            self.assertIsNone(judgment(0, win_mate, white))
        self.assertEqual(force_cp('#-0'), -1000)
        self.assertEqual(force_cp('#0'), 1000)
        self.assertEqual(move_metrics([5000, -5000])[1]['centipawn_loss'], 0)
        self.assertEqual(move_metrics([-5000, 5000])[1]['centipawn_loss'], 2000)

    def test_reference_game2_reproduces_every_supplied_lichess_result(self):
        data = json.loads(Path(__file__).with_name('data').joinpath('game2-lichess-accuracy.json').read_text(encoding='utf-8'))
        board = chess.Board(data['start_fen']); boards = [board.copy()]
        for move in data['moves']:
            board.push_uci(move); boards.append(board.copy(stack=False))
        div = division(boards)
        self.assertEqual(div, {'middle': 20, 'end': 40})
        scores = data['scores_cp']
        actual = game_accuracy(scores)
        phases = phase_accuracies(scores, div)
        for side, expected in data['expected'].items():
            rows = [r for r in move_metrics(scores) if r['side'] == side]
            observed = {'accuracy': round_percent(actual[side]),
                        'average_centipawn_loss': round_percent(statistics.mean(r['centipawn_loss'] for r in rows)),
                        **{k: round_percent(v) for k, v in phases[side].items()},
                        **{k: sum(r['judgment'] == v for r in rows) for k, v in
                           [('inaccuracies', 'inaccuracy'), ('mistakes', 'mistake'), ('blunders', 'blunder')]}}
            self.assertEqual(observed, expected)
        self.assertAlmostEqual(actual['white'], 82.64303836356419, places=10)
        self.assertAlmostEqual(actual['black'], 86.43785179447963, places=10)

    def test_short_game_and_endgame_fen_have_no_invented_phase_scores(self):
        board = chess.Board()
        div = division([board])
        self.assertEqual(div, {'middle': None, 'end': None})
        self.assertEqual(phase_accuracies([15, 15], div), {'white': {}, 'black': {}})
        board = chess.Board('8/8/8/5k2/8/8/P7/K7 w - - 0 1')
        self.assertEqual(division([board]), {'middle': None, 'end': 0})


class PerformanceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.engines = FakeEngines()
        self.addCleanup(self.engines._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('[White "Test White"]\n[Black "Test Black"]\n\n1. e4 e5 2. Nf3 Nc6 *'))
        self.analysis = analyze_game(game, self.engines, 'white', 1400, progress=lambda _: None)

    def test_analysis_persists_shared_metric_and_summary_only_reads_it(self):
        data = self.analysis
        self.assertEqual(data['performance']['method'], 'lichess')
        self.assertEqual(data['performance']['players']['white']['moves_scored'], 2)
        self.assertTrue(all('accuracy' in r and 'centipawn_loss' in r for r in data['moves']))
        before = copy.deepcopy(data)
        self.assertEqual(compact_summary(data)['performance'], data['performance'])
        self.assertEqual(data, before)
        self.assertIs(refresh_performance(data), data)
        self.assertEqual(data, before)

    def test_cached_upgrade_changes_only_quality_hints_and_preserves_tactical_evidence(self):
        data = self.analysis
        del data['performance']
        first = data['moves'][0]
        first['flags'] = ['sacrifice', 'mistake', 'natural_but_bad']
        first['played']['eval'] = 20  # Must use the next root, not this shallow candidate score.
        data['moves'][1]['position_eval'] = -9
        original = copy.deepcopy(data['played_elo'])
        sf, maia = len(self.engines.sf_calls), len(self.engines.human_calls)
        refresh_performance(data)
        self.assertIn('blunder', first['flags'])
        self.assertIn('sacrifice', first['flags'])
        self.assertNotIn('mistake', first['flags'])
        self.assertEqual(first['centipawn_loss'], 915)
        self.assertEqual(data['played_elo'], original)
        self.assertEqual((len(self.engines.sf_calls), len(self.engines.human_calls)), (sf, maia))
        for side in ('white', 'black'):
            player = data['performance']['players'][side]
            rows = [r for r in data['moves'] if r['side'] == side]
            for field, flag in [('inaccuracies','inaccuracy'), ('mistakes','mistake'), ('blunders','blunder')]:
                self.assertEqual(player[field], sum(flag in r['flags'] for r in rows))

    def test_reference_evals_flow_through_saved_analysis_and_snapshot(self):
        fixture = json.loads(Path(__file__).with_name('data').joinpath('game2-lichess-accuracy.json').read_text(encoding='utf-8'))
        data = copy.deepcopy(self.analysis)
        data.pop('performance')
        data['moves'] = []
        board = chess.Board(fixture['start_fen'])
        for index, (move, cp) in enumerate(zip(fixture['moves'], fixture['scores_cp'])):
            data['moves'].append({'ply': index+1, 'fen': board.fen(),
                'side': 'white' if board.turn else 'black',
                'position_eval': (fixture['scores_cp'][index-1] if index else 15)/100,
                'played': {'move': move, 'eval': cp/100},
                'candidate_moves': [], 'flags': []})
            board.push_uci(move)
        refresh_performance(data)
        for side, expected in fixture['expected'].items():
            actual = data['performance']['players'][side]
            self.assertEqual(actual['average_centipawn_loss'], expected['average_centipawn_loss'])
            self.assertEqual(round_percent(actual['accuracy']), expected['accuracy'])
            for phase in ('opening', 'middlegame', 'endgame'):
                self.assertEqual(round_percent(actual['phases'][phase]), expected[phase])
            for kind in ('inaccuracies', 'mistakes', 'blunders'):
                self.assertEqual(actual[kind], expected[kind])
        self.assertIn('| Accuracy | 83% | 86% |', performance_table(data['performance']))

    def test_custom_fen_uses_actual_initial_eval(self):
        game = chess.pgn.read_game(io.StringIO('[SetUp "1"]\n[FEN "8/8/8/5k2/8/8/P7/K7 w - - 0 1"]\n\n1. a3 *'))
        engines = FakeEngines(game.board().fen())
        self.addCleanup(engines._temp.cleanup)
        data = analyze_game(game, engines, 'white', 1400, progress=lambda _: None)
        self.assertEqual(data['performance']['initial_cp'], 20)
        self.assertEqual(data['moves'][0]['stage'], 'endgame')

    def test_snapshot_rendering_is_idempotent_and_escapes_names(self):
        data = self.analysis
        data['performance']['players']['white']['name'] = 'A|B\n<script>'
        draft = '# Review\n\n## Performance snapshot\n<!-- performance-statistics -->\nActual Elo 1400.\n\n## Opening\nBody'
        rendered = insert_performance_snapshot(draft, data)
        self.assertEqual(insert_performance_snapshot(rendered, data), rendered)
        self.assertIn('A\\|B &lt;script&gt;', rendered)
        self.assertEqual(rendered.count('| Accuracy |'), 1)
        self.assertIn('Actual Elo 1400.', rendered)
        self.assertEqual(insert_performance_snapshot('# Small review\nOne position.', data), '# Small review\nOne position.')
        fallback = insert_performance_snapshot('## Ratings\nActual Elo 1400.\n## Opening', data)
        self.assertLess(fallback.index('| Accuracy |'), fallback.index('## Opening'))

    def test_report_gate_publishes_the_exact_local_table_without_a_model_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = SimpleNamespace(directory=Path(tmp))
            gate = ReportGate(library, validator=lambda report: None, render=lambda text: insert_performance_snapshot(text, self.analysis))
            draft = '## Performance snapshot\nActual Elo 1400.\n\n## Opening\nBody'
            self.assertTrue(gate.validate(draft))
            report = gate.publish(draft)
            self.assertIn(performance_table(self.analysis['performance']), report)
            self.assertEqual(gate.attempts, 1)
            self.assertEqual((Path(tmp)/'coaching.md').read_text(encoding='utf-8').strip(), report)

    def test_normal_agent_run_contains_the_table_and_saves_upgraded_analysis(self):
        del self.analysis['performance']
        model = ScriptedCodex()
        model.index = 5
        with tempfile.TemporaryDirectory() as tmp:
            report = model.run(self.analysis, self.engines, Path(tmp))
            self.assertIn('| Average centipawn loss |', report)
            self.assertIn('| Accuracy |', report)
            saved = json.loads((Path(tmp)/'analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['performance'], self.analysis['performance'])


if __name__ == '__main__':
    unittest.main()
