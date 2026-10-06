"""Shared continuation searches preserve legal PVs and exact completed scores."""
import unittest
from unittest.mock import Mock

import chess
import chess.engine

from analysis.stockfish_exploration import explore


class ExplorationTests(unittest.TestCase):
    def test_exploration_limits_legal_lines_and_terminal(self):
        engine = Mock()
        info = {'score':chess.engine.PovScore(chess.engine.Cp(-23), chess.BLACK), 'depth':15,
                'pv':[chess.Move.from_uci('e7e5'), chess.Move.from_uci('g1f3')]}
        search = Mock()
        search.__enter__ = Mock(return_value=iter([info, {**info, 'depth':18, 'lowerbound':True}]))
        search.__exit__ = Mock(return_value=False)
        engine.analysis.return_value = search
        result = explore(engine, chess.Board(), move='e2e4', seconds=.4, depth=18)
        self.assertEqual(result['pv_san'], ['e4','e5','Nf3'])
        self.assertEqual(result['white_cp'],23)
        self.assertEqual(result['depth'],15) # Incomplete d18 must not replace exact d15.
        limit = engine.analysis.call_args.args[1]
        self.assertEqual((limit.depth, limit.time), (18,.4))
        with self.assertRaises(ValueError): explore(engine, chess.Board(), move='e2e5')
        board = chess.Board()
        for uci in ('f2f3','e7e5','g2g4','d8h4'): board.push_uci(uci)
        engine.reset_mock()
        self.assertTrue(explore(engine, board)['terminal'])
        engine.analysis.assert_not_called()


if __name__ == '__main__':
    unittest.main()
