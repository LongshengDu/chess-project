"""Coach-only scale/account revisions reuse game evidence; all engines/API mocked."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import chess.pgn

from analysis import elo_convert
from analysis.cache import write_json
from analysis.game.pipeline import analyze_game
from analysis.player_rating.service import evidence_cache_path
from analysis.settings import CONFIG
from coach.coach import main
from tests.coach.fixtures import FakeEngines


class CoachScaleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pgn = self.root/'revised.pgn'
        self.output = self.root/'output'/'revised-full'
        self.output.mkdir(parents=True)
        self.selector = patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve')
        self.selector.start()
        self.addCleanup(self.selector.stop)

    def write_game(self, **headers):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        game.headers.update(Site='Chess.com', TimeControl='300+0', WhiteElo='1400', BlackElo='1300', **headers)
        self.pgn.write_text(str(game), encoding='utf-8')
        return game

    def test_coach_only_uses_revised_pgn_and_replaces_selected_account_and_scale_overrides(self):
        original_game = self.write_game()
        seed_engines = FakeEngines(original_game.board().fen())
        self.addCleanup(seed_engines._temp.cleanup)
        analysis = analyze_game(original_game, seed_engines, 'white', 1499,
                                rating_scale='lb', progress=lambda _: None)
        write_json(self.output/'analysis.json', analysis)
        saved_moves, saved_positions = deepcopy(analysis['moves']), deepcopy(analysis['positions'])
        evidence_key = analysis['rating_fit']['evidence_key']
        evidence_path = evidence_cache_path(seed_engines.cache.directory, evidence_key)
        evidence_bytes = evidence_path.read_bytes()
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        game.headers.update(Site='https://lichess.org/revised', TimeControl='600+0',
                            WhiteElo='1800', BlackElo='1750', White='Updated player', Event='Current metadata')
        self.pgn.write_text(str(game), encoding='utf-8')
        engines = MagicMock()
        engines.__enter__.return_value = engines
        received = []

        def coach_stub(current, opened, output, **_kwargs):
            self.assertIs(opened, engines)
            self.assertEqual(output, self.output)
            received.append(deepcopy(current))
            current['agent_run'] = {'usage': {'total_tokens': 0}}

        def run(side, elo, *extra):
            with patch('coach.coach.Engines', return_value=engines) as constructor, \
                 patch('coach.coach.analyze_game', side_effect=AssertionError('Coach-only must not reanalyze the game')), \
                 patch('analysis.player_rating.service.collect_evidence', side_effect=AssertionError('Use saved rating evidence')), \
                 patch('analysis.player_rating.figures.export_saved_figures') as figures, \
                 patch('coach.agent_runner.run_coach', side_effect=coach_stub) as coach, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main([str(self.pgn), '--side', side, '--elo', str(elo), '--coach-only',
                               '--cache-dir', str(seed_engines.cache.directory), *extra])
            self.assertEqual(status, 0)
            constructor.assert_called_once()
            coach.assert_called_once()
            figures.assert_called_once()
            self.assertEqual(figures.call_args.kwargs.get('output_dir', figures.call_args.args[1]),
                             self.output/'player-rating')
            saved = json.loads((self.output/'analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['headers'], dict(game.headers))
            self.assertEqual(saved['moves'], saved_moves)
            self.assertEqual(saved['positions'], saved_positions)
            self.assertEqual(saved['rating_fit']['evidence_key'], evidence_key)
            self.assertEqual(evidence_path.read_bytes(), evidence_bytes)
            self.assertEqual((self.output/'game.pgn').read_text(encoding='utf-8'), self.pgn.read_text(encoding='utf-8'))
            return saved

        inferred = run('black', 1600)
        self.assertNotIn('rating_scale_override', inferred)
        self.assertEqual(inferred['rating_account_overrides'], {'Black': 1600})
        self.assertEqual(inferred['selected_player'], {'side': 'black', 'actual_elo': 1600})
        self.assertEqual(inferred['played_elo_scale']['scale'], 'lr')
        self.assertEqual(inferred['played_elo_scale']['source'], 'pgn_headers')
        self.assertEqual(inferred['played_elo_scale']['actual_ratings'], {'White': 1800., 'Black': 1600.})
        self.assertAlmostEqual(inferred['played_elo_scale']['native_actual_ratings']['White'],
                               elo_convert.convert(1800., 'lr', 'lb'), places=8)
        overridden = run('white', 1650, '--rating-scale', 'cr')
        self.assertEqual(overridden['rating_scale_override'], 'cr')
        self.assertEqual(overridden['rating_account_overrides'], {'White': 1650})
        self.assertEqual(overridden['selected_player'], {'side': 'white', 'actual_elo': 1650})
        self.assertEqual(overridden['played_elo_scale']['scale'], 'cr')
        self.assertEqual(overridden['played_elo_scale']['actual_ratings'], {'White': 1650., 'Black': 1750.})
        self.assertNotEqual(overridden['rating_fit']['context_signature'], inferred['rating_fit']['context_signature'])
        self.assertEqual([item['selected_player']['side'] for item in received], ['black', 'white'])
        self.assertEqual(engines.__exit__.call_count, 2)

    def test_unsupported_scale_is_rejected_before_engines_or_game_analysis(self):
        game = self.write_game()
        game.headers.update(Site='https://lichess.org/fixture', TimeControl='60+0')
        self.pgn.write_text(str(game), encoding='utf-8')
        with patch('coach.coach.Engines') as engines, patch('coach.coach.analyze_game') as analyze, \
             patch('coach.agent_runner.run_coach') as coach, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            status = main([str(self.pgn), '--side', 'white', '--elo', '1500', '--analysis-only'])
        self.assertEqual(status, 1)
        self.assertIn('bullet', errors.getvalue())
        engines.assert_not_called()
        analyze.assert_not_called()
        coach.assert_not_called()

    def test_chesscom_rating_floor_is_rejected_before_engines_for_headers_and_cli(self):
        for source in ('header', 'selected'):
            with self.subTest(source=source):
                game = self.write_game()
                if source == 'header':
                    game.headers['BlackElo'] = '100'
                    self.pgn.write_text(str(game), encoding='utf-8')
                actual = '100' if source == 'selected' else '1500'
                with patch('coach.coach.Engines') as engines, patch('coach.coach.analyze_game') as analyze, \
                     patch('coach.agent_runner.run_coach') as coach, \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                    status = main([str(self.pgn), '--side', 'white', '--elo', actual, '--analysis-only'])
                self.assertEqual(status, 1)
                self.assertIn('lower asymptote', errors.getvalue())
                engines.assert_not_called()
                analyze.assert_not_called()
                coach.assert_not_called()


if __name__ == '__main__':
    unittest.main()
