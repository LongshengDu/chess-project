import unittest
from unittest.mock import Mock

import chess

from analysis.game.study import PgnAnalysisApi


class UniformPolicy:
    def probabilities(self, board):
        if board.is_game_over():
            return []
        moves = list(board.legal_moves)
        return [(move, 1 / len(moves)) for move in moves]


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.scorer = Mock()
        self.scorer.evaluate.return_value = ({"cp": 20, "mate": None, "label": "+0.20"}, 12)
        self.scorer.score.return_value = ({}, 12)
        self.analysis = PgnAnalysisApi(UniformPolicy(), self.scorer)

    def test_pgn_mainline_and_nested_variations(self):
        game = self.analysis.import_pgn('1. e4 (1. d4 d5 (1... Nf6) 2. c4) e5 2. Nf3 *')
        self.assertEqual(game['moves'], ['e4', 'e5', 'Nf3'])
        self.assertEqual(len(game['positions']), 4)
        self.assertEqual(len(game['variations']), 4)
        self.assertEqual(game['variations'][1]['moves'], ['d2d4', 'd7d5'])
        for branch in game['variations']:
            board = self.analysis._board_at(branch['ply'], branch['moves'])
            self.assertEqual(branch['position']['fullFen'], board.fen(en_passant='fen'))
        self.analysis.analyse_position(0, ['d2d4', 'g8f6'])

    def test_fen_preserves_black_turn_castling_counters(self):
        fen = 'r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R b KQkq - 9 23'
        game = self.analysis.import_fen(fen)
        position = game['positions'][0]
        self.assertEqual(position['fullFen'], fen)
        self.assertEqual(position['turn'], 'black')
        self.assertEqual(position['ply'], 45)
        step = self.analysis.play_variation_move(0, [], 'e8g8')
        self.assertIn('r4rk1', step['position']['fen'])
        self.assertEqual(step['san'], 'O-O')

    def test_fen_en_passant(self):
        self.analysis.import_fen('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 20')
        step = self.analysis.play_variation_move(0, [], 'e5d6')
        self.assertEqual(step['san'], 'exd6')
        self.assertEqual(step['position']['fen'], '4k3/8/3P4/8/8/8/8/4K3')

    def test_underpromotion(self):
        game = self.analysis.import_fen('7k/P7/8/8/8/8/8/7K w - - 0 1')
        self.assertEqual(set(game['positions'][0]['promotions']['a7a8']), set('qnrb'))
        step = self.analysis.play_variation_move(0, [], 'a7a8n')
        self.assertEqual(step['position']['fen'], 'N6k/8/8/8/8/8/8/7K')

    def test_invalid_import_does_not_replace_current_game(self):
        self.analysis.import_pgn('1. e4 e5 *')
        for fen in ['', 'invalid', '8/8/8/8/8/8/8/8 w - - 0 1', '8/8/8/8/8/8/8/K6k w - - 0']:
            with self.subTest(fen=fen), self.assertRaises(ValueError):
                self.analysis.import_fen(fen)
        for pgn in ('not a chess game', '1. e4 e5 2. Bh6 *'):
            with self.subTest(pgn=pgn), self.assertRaises(ValueError):
                self.analysis.import_pgn(pgn)
        self.assertEqual(self.analysis._board_at(2).peek().uci(), 'e7e5')

    def test_reject_multiple_games_and_variants(self):
        for pgn in ('1. e4 *\n\n1. d4 *', '[Variant "Atomic"]\n\n1. e4 *'):
            with self.subTest(pgn=pgn), self.assertRaises(ValueError):
                self.analysis.import_pgn(pgn)

    def test_setup_pgn_black_to_move(self):
        game = self.analysis.import_pgn('[SetUp "1"]\n[FEN "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R b KQkq - 9 23"]\n\n23... O-O *')
        self.assertEqual(game['positions'][0]['ply'], 45)
        self.assertEqual(game['moves'], ['O-O'])

    def test_variation_legality_and_position_bounds(self):
        with self.assertRaises(ValueError):
            self.analysis.analyse_position(0)
        self.analysis.import_fen(chess.STARTING_FEN)
        for ply, moves in ((1, []), (-1, []), (0, ['e2e5']), (0, ['bad'])):
            with self.subTest(ply=ply, moves=moves), self.assertRaises(ValueError):
                self.analysis.analyse_position(ply, moves)
        with self.assertRaises(ValueError):
            self.analysis.play_variation_move(0, [], 'e2e5')

    def test_cache_and_import_invalidation(self):
        self.analysis.import_fen(chess.STARTING_FEN)
        first = self.analysis.analyse_position(0)
        second = self.analysis.analyse_position(0)
        self.assertEqual(first, second)
        self.assertEqual(self.scorer.evaluate.call_count, 1)
        self.assertAlmostEqual(sum(move['probability'] for move in first['moves']) + first['otherProbability'], 1)
        self.analysis.import_fen(chess.STARTING_FEN)
        self.analysis.analyse_position(0)
        self.assertEqual(self.scorer.evaluate.call_count, 2)

    def test_terminal_fen(self):
        game = self.analysis.import_fen('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
        self.assertEqual(game['positions'][0]['status'], 'Checkmate')
        self.assertEqual(game['positions'][0]['dests'], {})
        analysis = self.analysis.analyse_position(0)
        self.assertEqual(analysis['moves'], [])
        self.assertEqual(analysis['otherProbability'], 0)

    def test_human_move_display_is_fixed_at_ten_and_retains_played_move(self):
        self.analysis.import_fen(chess.STARTING_FEN)
        result = self.analysis.analyse_position(0)
        self.assertEqual(len(result['moves']), 10)
        policy = UniformPolicy().probabilities(chess.Board())
        self.assertEqual([row['uci'] for row in result['moves']], [move.uci() for move, _ in policy[:10]])
        played = policy[-1][0]
        self.analysis.import_pgn(f'1. {chess.Board().san(played)} *')
        result = self.analysis.analyse_position(0)
        self.assertEqual(len(result['moves']), 11)
        self.assertEqual(result['moves'][-1]['uci'], played.uci())
        self.assertTrue(result['moves'][-1]['played'])

if __name__ == '__main__':
    unittest.main()
