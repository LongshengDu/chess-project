"""Shared continuation searches preserve legal PVs and exact completed scores."""
import unittest
from unittest.mock import Mock

import chess
import chess.engine

from analysis.stockfish_exploration import analysis_result, explore, search_lines, validate_search_lines
from tests.analysis.test_stockfish_search import FakeSearch


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

    def test_full_raw_pv_is_retained_while_presentation_length_changes(self):
        board = chess.Board()
        pv = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
        engine = Mock()
        engine.analysis.return_value = FakeSearch([{'score': chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
            'depth': 12, 'pv': [chess.Move.from_uci(move) for move in pv]}])
        raw = search_lines(engine, board, seconds=.1, depth=18)
        self.assertEqual(raw, {'lines': [{'cp': 20, 'mate': None, 'depth': 12, 'pv_uci': pv}]})
        for length in (1, 4):
            response = analysis_result(board, raw, movetime_ms=100, depth=18, pv_plies=length)
            self.assertEqual(response['lines'][0]['pv_uci'], pv[:length])
        self.assertEqual(raw['lines'][0]['pv_uci'], pv)
        self.assertEqual(engine.analysis.call_count, 1)

    def test_invalid_scores_or_illegal_pvs_are_not_accepted(self):
        valid = {'cp': 20, 'mate': None, 'depth': 12, 'pv_uci': ['e2e4', 'e7e5']}
        for changes in ({'cp': float('nan')}, {'cp': 20, 'mate': 3}, {'depth': True},
                        {'pv_uci': []}, {'pv_uci': ['e2e4', 'e2e3']}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_search_lines(chess.Board(), {'lines': [{**valid, **changes}]})

    def test_incomplete_multipv_is_not_a_complete_cached_measurement(self):
        engine = Mock()
        engine.analysis.return_value = FakeSearch([{'score': chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
            'depth': 12, 'pv': [chess.Move.from_uci('e2e4')], 'multipv': 1}])
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            search_lines(engine, chess.Board(), seconds=.1, depth=18, multipv=2)

    def test_terminal_web_score_matches_previous_white_perspective_for_both_winners(self):
        for moves, winner in ((['f2f3', 'e7e5', 'g2g4', 'd8h4'], 'black'),
                              (['e2e4', 'e7e5', 'd1h5', 'b8c6', 'f1c4', 'g8f6', 'h5f7'], 'white')):
            board = chess.Board()
            for uci in moves:
                board.push_uci(uci)
            with self.subTest(winner=winner):
                engine = Mock()
                web = explore(engine, board, seconds=.5, depth=18)
                previous = chess.engine.PovScore(chess.engine.Mate(0), board.turn).white()
                self.assertEqual((web['white_cp'], web['white_mate']), (previous.score(), previous.mate()))
                self.assertTrue(web['terminal'])
                self.assertEqual(web['pv_uci'], [])
                coach = analysis_result(board, {'lines': []}, movetime_ms=500, depth=18, pv_plies=10)
                self.assertEqual(coach['evaluation']['winner'], winner)
                engine.analysis.assert_not_called()


if __name__ == '__main__':
    unittest.main()
