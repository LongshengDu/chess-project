"""Saved truncated continuations must never become purported complete new PVs."""
import unittest

import chess

from tests.analysis.cache_exploration_conversion import convert_exploration


class ExplorationConversionTests(unittest.TestCase):
    def test_short_full_session_line_converts_but_cutoff_line_does_not(self):
        request = {'kind': 'exploration', 'version': 1, 'engine': {'stockfish': ['engine', 1]},
                   'movetime_ms': 500, 'depth': 18, 'multipv': 1, 'root_moves': None, 'pv_plies': 4}
        result = {'lines': [{'cp': 20, 'mate': None, 'depth': 15, 'pv_uci': ['e2e4', 'e7e5'],
                              'san': 'e4', 'pv_san': ['e4', 'e5']}]}
        root, canonical, raw = convert_exploration(chess.Board(), request, result)
        self.assertEqual(root.fen(), chess.STARTING_FEN)
        self.assertEqual(canonical['version'], 2)
        self.assertNotIn('pv_plies', canonical)
        self.assertEqual(set(raw['lines'][0]), {'cp', 'mate', 'depth', 'pv_uci'})
        self.assertIsNone(convert_exploration(root, {**request, 'pv_plies': 2}, result))

    def test_web_parent_moves_to_child_context_and_removes_prefix(self):
        board = chess.Board()
        board.push_uci('e2e4')
        request = {'kind': 'continuation', 'version': 1, 'engine': {'stockfish': ['engine', 1]},
                   'move': 'e7e5', 'seconds': .5, 'depth': 18}
        result = {'pv_uci': ['e7e5', 'g1f3', 'b8c6'], 'white_cp': 20,
                  'white_mate': None, 'depth': 15, 'bound': False}
        root, canonical, raw = convert_exploration(board, request, result)
        self.assertEqual([move.uci() for move in root.move_stack], ['e2e4', 'e7e5'])
        self.assertEqual(raw['lines'][0]['pv_uci'], ['g1f3', 'b8c6'])
        self.assertEqual(canonical['movetime_ms'], 500)
        self.assertEqual(canonical['root_moves'], None)
        self.assertIsNone(convert_exploration(board, request, {**result, 'bound': True}))

    def test_proven_terminal_pv_at_cutoff_is_complete(self):
        board = chess.Board()
        for uci in ('f2f3', 'e7e5', 'g2g4'):
            board.push_uci(uci)
        request = {'kind': 'exploration', 'version': 1, 'engine': {'stockfish': ['engine', 1]},
                   'movetime_ms': 500, 'depth': 18, 'multipv': 1, 'root_moves': None, 'pv_plies': 1}
        result = {'lines': [{'cp': None, 'mate': -1, 'depth': 15, 'pv_uci': ['d8h4']}]}
        self.assertIsNotNone(convert_exploration(board, request, result))
        board.push_uci('d8h4')
        self.assertEqual(convert_exploration(board, request, {'lines': []})[2], {'lines': []})

    def test_partial_multipv_invalid_pv_and_ambiguous_score_are_dropped(self):
        request = {'kind': 'exploration', 'version': 1, 'engine': {'stockfish': ['engine', 1]},
                   'movetime_ms': 500, 'depth': 18, 'multipv': 2, 'root_moves': None, 'pv_plies': 10}
        line = {'cp': 20, 'mate': None, 'depth': 15, 'pv_uci': ['e2e4']}
        self.assertIsNone(convert_exploration(chess.Board(), request, {'lines': [line]}))
        for patch in ({'pv_uci': ['e2e5']}, {'mate': 2}, {'cp': float('nan')}):
            self.assertIsNone(convert_exploration(chess.Board(), {**request, 'multipv': 1},
                                                  {'lines': [{**line, **patch}]}))
