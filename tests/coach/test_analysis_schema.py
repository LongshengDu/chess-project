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
from analysis.game.summary import compact_summary
from analysis.game.history import history_at
from tests.coach.fixtures import FakeAnalysisSession


class SchemaTests(unittest.TestCase):
    def test_exact_move_contract_and_candidate_union_preserve_rare_played_move(self):
        class AnalysisSession(FakeAnalysisSession):
            def initial_analysis(self, *args):
                result = super().initial_analysis(*args)
                if 'h2h4' in {line['uci'] for line in result['lines']}:
                    result['best_move'] = 'h2h4'
                return result
        session = AnalysisSession()
        self.addCleanup(session._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('1. e4 *'))
        data = analyze_game(game, session, actual_elo=1270, progress=lambda _: None)
        row = data['moves'][0]
        self.assertEqual(set(row), {'ply', 'label', 'side', 'stage', 'fen', 'position_eval', 'played', 'maia', 'candidate_moves', 'flags', 'accuracy', 'centipawn_loss'})
        self.assertEqual(set(row['played']), {'move', 'san', 'eval', 'loss'})
        self.assertEqual(set(row['maia']), set(map(str, range(600, 2601, 100))))
        candidates = {c['move']: c for c in row['candidate_moves']}
        expected = {'e2e4', 'h2h4'}
        for rating in map(str, RATINGS):
            record = row['maia'][rating]
            self.assertEqual(set(record), {'moves', 'expected_accuracy', 'absolute_deviation'})
            self.assertIsInstance(record['expected_accuracy'], float)
            self.assertIsInstance(record['absolute_deviation'], float)
            choices = record['moves']
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

    def test_maia_comparison_reuses_nested_records_at_low_native_anchors(self):
        session = FakeAnalysisSession()
        self.addCleanup(session._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('1. e4 *'))
        data = analyze_game(game, session, actual_elo=700, progress=lambda _: None)
        with tempfile.TemporaryDirectory() as tmp:
            library = ChessTools(data, session, Path(tmp), side='white')
            with patch.object(session, 'human', side_effect=AssertionError('Reuse saved Maia')):
                result = library.maia_compare(1, [600, 700, 1600, 2600], topk=2)
        self.assertEqual(result['conditioning'], 'equal_rating')
        self.assertEqual(set(result['maia']), {'600', '700', '1600', '2600'})
        for rating, choices in result['maia'].items():
            self.assertEqual(choices, data['moves'][0]['maia'][rating]['moves'][:2])
            self.assertEqual(len(data['moves'][0]['maia'][rating]['moves']), 5)

    def test_compact_summary_exposes_only_descriptive_accuracy_and_native_maia_context(self):
        session = FakeAnalysisSession()
        self.addCleanup(session._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        data = analyze_game(game, session, actual_elo=1270, progress=lambda _: None)
        summary = compact_summary(data, 'white')
        evidence = summary['accuracy_curve']
        self.assertEqual(evidence['ratings'], list(range(600, 2601, 100)))
        self.assertEqual(evidence['rating_scale'], 'lb')
        self.assertEqual(len(evidence['expected_accuracy']), 21)
        self.assertEqual(len(evidence['absolute_deviation']), 21)
        self.assertEqual(evidence['position_selection'], data['accuracy_curve']['position_selection'])
        self.assertEqual(evidence['pooling'], data['accuracy_curve']['pooling'])
        self.assertNotIn('elo_note', summary)
        for side in ('white', 'black'):
            self.assertEqual(evidence['players'][side]['average_accuracy'],
                             data['accuracy_curve']['players'][side]['average_accuracy'])
            self.assertEqual(evidence['players'][side]['lichess_accuracy'],
                             data['performance']['players'][side]['accuracy'])
            self.assertNotIn('expected_accuracy', evidence['players'][side])

    def test_mate_and_deeper_branches_are_legal_and_history_aware(self):
        session = FakeAnalysisSession()
        self.addCleanup(session._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('1. g4 e5 2. f3 Qh4# 0-1'))
        data = analyze_game(game, session, actual_elo=1270, progress=lambda _: None)
        self.assertEqual(data['moves'][-1]['played']['eval'], '#-0')
        self.assertIsNone(data['moves'][-1]['played']['loss'])
        self.assertEqual(checking_lines(game.end().board()),
                         [{'piece': 'queen', 'squares': ['h4', 'g3', 'f2', 'e1']}])
        with tempfile.TemporaryDirectory() as tmp:
            library = ChessTools(data, session, Path(tmp), side='white')
            result = library.explore_candidate(3, 'g1f3')
            ref = result['after_defense']
            history = history_at(data, **ref)
            self.assertEqual(history[:3], ['g2g4', 'e7e5', 'g1f3'])
            self.assertEqual(session.board(history).fen(), result['fen_after_defense'])
            self.assertTrue(any(h == history for h, _, _ in session.human_calls))
            board = session.board(history)
            reply = next(iter(board.legal_moves)).uci()
            deeper = library.explore_candidate(ref['ply'], reply, ref['line'])
            self.assertEqual(history_at(data, **deeper['after_defense'])[:len(history)+1], history+[reply])
            with self.assertRaises(ValueError):
                library.get_position(3, ['e2e5'])
            with self.assertRaises(ValueError):
                library.get_position(5, ['e2e4'])


if __name__ == '__main__':
    unittest.main()
