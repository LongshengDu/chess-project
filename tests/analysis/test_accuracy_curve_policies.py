"""Curve policy preparation reuses supplied position measurements without a game cache."""
import io
import unittest

import chess.pgn

from analysis.accuracy.policies import prepare_policies
from analysis.accuracy.evidence import RATINGS


class AccuracyCurvePolicyTests(unittest.TestCase):
    @staticmethod
    def predictions(board):
        policy = {move.uci(): 1 / board.legal_moves.count() for move in board.legal_moves}
        return [{'policy': policy} for _ in RATINGS]

    def test_prepared_batches_attach_full_native_policies_in_game_order(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        records = [{'move':'e2e4','side':'White'}, {'move':'e7e5','side':'Black'}]
        predictions = [self.predictions(board) for board in (game.board(), game.variations[0].board())]
        prepare_policies(game, records, predictions)
        self.assertEqual(set(records[0]['policies']), set(RATINGS))
        self.assertEqual(set(records[1]['policies'][1600]),
                         {m.uci() for m in game.variations[0].board().legal_moves})

    def test_invalid_or_incomplete_supplied_policies_are_rejected(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 *'))
        records = [{'move':'e2e4','side':'White'}]
        with self.assertRaisesRegex(ValueError, 'complete legal policy'):
            prepare_policies(game, records, [[{'policy':{'e2e4':1.}} for _ in RATINGS]])
        with self.assertRaisesRegex(ValueError, 'mainline'):
            prepare_policies(game, [{'move':'d2d4'}], [self.predictions(game.board())])
        with self.assertRaisesRegex(ValueError, 'position batch'):
            prepare_policies(game, records, [])
        with self.assertRaisesRegex(ValueError, 'rating policies'):
            prepare_policies(game, records, [self.predictions(game.board())[:-1]])
        with self.assertRaisesRegex(ValueError, 'wrong player side'):
            prepare_policies(game, [{'move': 'e2e4', 'side': 'Black'}], [self.predictions(game.board())])

    def test_forced_position_needs_no_prediction(self):
        game = chess.pgn.read_game(io.StringIO(
            '[FEN "8/8/8/8/r7/2k5/K7/8 w - - 0 42"]\n\n42. Kb1 *'))
        records = [{'move':'a2b1','side':'White'}]
        prepare_policies(game, records, [None])
        self.assertEqual(records[0]['policies'][1600], {'a2b1':1.})


if __name__ == '__main__':
    unittest.main()
