"""ana.md contract, compact branch references, mate semantics and local evidence."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn

from coach.tools_chess import ChessTools, checking_lines
from analysis.game.pipeline import analyze_game
from analysis.position_evaluation import RATINGS
from analysis.game.summary import compact_summary, rating_note
from analysis.game.history import history_at
from analysis.settings import CONFIG
from tests.coach.fixtures import FakeEngines


class SchemaTests(unittest.TestCase):
    def test_exact_move_contract_and_candidate_union_preserve_rare_played_move(self):
        class Engines(FakeEngines):
            def initial_analysis(self, *args):
                result = super().initial_analysis(*args)
                result['best_move'] = 'h2h4'
                return result
        engines = Engines()
        self.addCleanup(engines._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('1. e4 *'))
        with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'):
            data = analyze_game(game, engines, 'white', 1270, progress=lambda _: None)
        self.assertEqual(data['played_elo_central_interval'], .20)
        self.assertIn('central 20% interval', compact_summary(data)['elo_note'])
        legacy = {k: v for k, v in data.items() if k != 'played_elo_central_interval'}
        legacy['played_elo_confidence'] = .68
        self.assertIn('central 68% interval', compact_summary(legacy)['elo_note'])
        legacy.pop('played_elo_confidence')
        self.assertIn('saved interval', compact_summary(legacy)['elo_note'])
        row = data['moves'][0]
        self.assertEqual(set(row), {'ply', 'label', 'side', 'stage', 'fen', 'position_eval', 'played', 'maia', 'candidate_moves', 'flags', 'accuracy', 'centipawn_loss'})
        self.assertEqual(set(row['played']), {'move', 'san', 'eval', 'loss'})
        self.assertEqual(set(row['maia']), set(map(str, RATINGS)))
        candidates = {c['move']: c for c in row['candidate_moves']}
        expected = {'e2e4', 'h2h4'}
        for rating, choices in row['maia'].items():
            self.assertEqual(len(choices), 5)
            for choice in choices:
                self.assertEqual(set(choice), {'move', 'san', 'p', 'eval', 'loss'})
                expected.add(choice['move'])
                self.assertEqual(choice['p'], candidates[choice['move']]['maia_p'][rating])
                self.assertEqual(choice['eval'], candidates[choice['move']]['eval'])
        self.assertEqual(set(candidates), expected)
        self.assertTrue(all(set(c) == {'move', 'san', 'eval', 'loss', 'maia_p'} for c in candidates.values()))
        self.assertNotIn('history_before', json.dumps(data))
        self.assertNotIn('joint_log_likelihood', json.dumps(data))

    def test_point_only_ratings_do_not_claim_an_uncertainty_interval(self):
        saved = {'played_elo_name': 'Shared-curve uncertainty ensemble',
                 'played_elo_central_interval': None,
                 'played_elo': {'white': {'estimate': 1700, 'interval': None, 'uncertainty': None},
                                'black': {'estimate': 1600, 'interval': None, 'uncertainty': None}},
                 'played_elo_interval_scope': 'Point-only decision ensemble.'}
        note = rating_note(saved)
        self.assertIn('point estimates; no uncertainty interval', note)
        self.assertIn('Point-only decision ensemble.', note)
        self.assertNotIn('saved interval', note)
        self.assertNotIn('None', note)

    def test_legacy_interval_note_is_used_only_when_a_saved_interval_exists(self):
        self.assertIn('saved interval', rating_note({'played_elo': {'white': {'interval': [1300, 1700]}}}))
        self.assertIn('point estimates', rating_note({'played_elo': {'white': {'estimate': 1500}}}))

    def test_mate_and_deeper_branches_are_legal_and_history_aware(self):
        engines = FakeEngines()
        self.addCleanup(engines._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('1. g4 e5 2. f3 Qh4# 0-1'))
        data = analyze_game(game, engines, 'white', 1270, progress=lambda _: None)
        self.assertEqual(data['moves'][-1]['played']['eval'], '#-0')
        self.assertIsNone(data['moves'][-1]['played']['loss'])
        self.assertEqual(checking_lines(game.end().board()),
                         [{'piece': 'queen', 'squares': ['h4', 'g3', 'f2', 'e1']}])
        with tempfile.TemporaryDirectory() as tmp:
            library = ChessTools(data, engines, Path(tmp))
            result = library.explore_candidate(3, 'g1f3')
            ref = result['after_defense']
            history = history_at(data, **ref)
            self.assertEqual(history[:3], ['g2g4', 'e7e5', 'g1f3'])
            self.assertEqual(engines.board(history).fen(), result['fen_after_defense'])
            self.assertTrue(any(h == history for h, _, _ in engines.human_calls))
            board = engines.board(history)
            reply = next(iter(board.legal_moves)).uci()
            deeper = library.explore_candidate(ref['ply'], reply, ref['line'])
            self.assertEqual(history_at(data, **deeper['after_defense'])[:len(history)+1], history+[reply])
            with self.assertRaises(ValueError):
                library.get_position(3, ['e2e5'])
            with self.assertRaises(ValueError):
                library.get_position(5, ['e2e4'])


if __name__ == '__main__':
    unittest.main()
