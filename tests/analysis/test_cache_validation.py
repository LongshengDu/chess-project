"""Only complete measured chess evidence is admitted to the shared cache."""
import tempfile
import unittest
from pathlib import Path

import chess

from analysis.cache.positions import PositionCache
from analysis.cache.validation import validate_measurement


class CacheValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = PositionCache(temporary.name)
        self.board = chess.Board()
        legal = [move.uci() for move in self.board.legal_moves]
        self.request = {'kind': 'evaluation', 'depth': 28}
        self.result = {'complete': True, 'coverage_complete': True, 'target_reached': False,
                       'cp_vec': dict.fromkeys(legal, 10), 'mate_vec': {},
                       'root_move_depth_vec': dict.fromkeys(legal, 12), 'best_move': legal[0],
                       'engine_moves': legal[:4]}

    def test_complete_time_limited_search_is_valid_without_reaching_depth_ceiling(self):
        reference = self.cache.put(self.board, 'stockfish', self.request, self.result)
        self.assertFalse(self.cache.get_reference(reference)['target_reached'])

    def test_incomplete_or_unavailable_evidence_never_creates_a_position_file(self):
        for update in ({'complete': False}, {'coverage_complete': False}, {'available': False},
                       {'cp_vec': {}}, {'root_move_depth_vec': {}}, {'best_move': 'a1a8'}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.cache.put(self.board, 'stockfish', self.request, {**self.result, **update})
        self.assertEqual(list((self.cache.directory/'positions').glob('*.json')), [])

    def test_missing_maia_value_is_not_a_prediction_even_with_a_forced_policy(self):
        board = chess.Board('7k/8/5K2/8/8/8/8/8 b - - 0 1')
        with self.assertRaises(ValueError):
            self.cache.put(board, 'maia', {'own_rating': 1600, 'opponent_rating': 1600},
                           {'policy': {'h8g8': 1.}, 'value': None})

    def test_engine_move_list_is_required_complete_and_legal(self):
        missing = dict(self.result)
        del missing['engine_moves']
        with self.assertRaisesRegex(ValueError, 'incomplete or invalid'):
            self.cache.put(self.board, 'stockfish', self.request, missing)
        for moves in (None, [], ['e2e4', 'e2e4'], ['a1a8'], [123], [['e2e4']]):
            with self.subTest(moves=moves), self.assertRaisesRegex(ValueError, 'incomplete or invalid'):
                self.cache.put(self.board, 'stockfish', self.request, {**self.result, 'engine_moves': moves})
        self.assertEqual(list((self.cache.directory/'positions').glob('*.json')), [])

    def test_terminal_draw_is_valid_complete_evidence_with_empty_move_maps(self):
        board = chess.Board('8/8/8/8/8/6k1/8/7K w - - 0 1')
        result = dict(self.result, cp_vec={}, mate_vec={}, root_move_depth_vec={}, best_move=None, engine_moves=[])
        validate_measurement(board, 'stockfish', self.request, result)
        with self.assertRaises(ValueError):
            validate_measurement(board, 'stockfish', self.request, {**result, 'engine_moves': ['h1h2']})

    def test_continuations_require_all_requested_roots_and_a_legal_line(self):
        request = {'kind': 'exploration', 'version': 2, 'multipv': 2, 'root_moves': None}
        line = {'cp': 10, 'mate': None, 'depth': 12, 'pv_uci': ['e2e4', 'e7e5']}
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_measurement(self.board, 'stockfish', request, {'lines': [line]})
        request['multipv'] = 1
        validate_measurement(self.board, 'stockfish', request, {'lines': [line]})
        with self.assertRaises(ValueError):
            validate_measurement(self.board, 'stockfish', request,
                                 {'lines': [{**line, 'pv_uci': ['e2e5']}]})
