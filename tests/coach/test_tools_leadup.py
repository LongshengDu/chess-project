"""Actual history and positional buildup, without model or engine calls."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn
from coach.tools_chess import ChessTools
from analysis.game.context import leadup_context, position_facts
from tests.coach.fixtures import FakeEngines


def analysis_for(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board, rows = game.board(), []
    for ply, move in enumerate(game.mainline_moves(),1):
        rows.append({'ply':ply, 'label':f'{board.fullmove_number}{"." if board.turn else "..."} {board.san(move)}',
            'fen':board.fen(), 'side':'white' if board.turn else 'black', 'position_eval':0.,
            'played':{'move':move.uci(),'eval':0.,'loss':0.}, 'flags':[]})
        board.push(move)
    return {'start_fen':game.board().fen(), 'moves':rows,
            'selected_player':{'side':'black','actual_elo':1270}}


class LeadupTests(unittest.TestCase):
    def test_example_setup_is_visible_before_error_and_overlaps_are_deduplicated(self):
        analysis = analysis_for('1. e4 e5 2. Qh5 Nc6 3. Qf5 h6 4. Bc4 d6 5. Qxf7# 1-0')
        original = json.dumps(analysis,sort_keys=True)
        with patch('analysis.move_hints.move_flags',side_effect=AssertionError('Never recalculate flags')):
            context = leadup_context(analysis,[8,6,2])
        self.assertEqual([m['ply'] for m in context['moves']],list(range(1,8)))
        self.assertEqual(context['positions']['8']['fen'],analysis['moves'][7]['fen'])
        self.assertEqual(context['windows'][0]['to_ply_exclusive'],8)
        setup = next(m for m in context['moves'] if m['ply']==7)
        self.assertEqual(setup['label'],'4. Bc4')
        self.assertIn('c4>f7', setup['changes']['black']['king_zone_attacks']['added'])
        developing = next(m for m in context['moves'] if m['ply']==4)
        self.assertEqual(developing['changes']['black']['minor_home_squares']['removed'],['b8'])
        self.assertTrue(any(m['ply']==6 and m['flags']==[] for m in context['moves']))
        pawn_move = next(m for m in context['moves'] if m['ply']==6)
        self.assertEqual(pawn_move['changes']['black']['pawn_squares'],{'added':['h6'],'removed':['h7']})
        self.assertEqual(json.dumps(analysis,sort_keys=True),original)

    def test_empty_first_move_final_board_and_backward_paging(self):
        analysis = analysis_for('1. e4 e5 2. Qh5 Nc6 3. Qf5 h6 4. Bc4 d6 5. Qxf7# 1-0')
        self.assertEqual(leadup_context(analysis,[1])['moves'],[])
        final = leadup_context(analysis,[10],2)
        self.assertEqual([m['ply'] for m in final['moves']],[8,9])
        self.assertTrue(chess.Board(final['positions']['10']['fen']).is_checkmate())
        previous = leadup_context(analysis,[final['windows'][0]['from_ply']],2)
        self.assertEqual([m['ply'] for m in previous['moves']],[6,7])
        for ply,count in [(0,8),(11,8),(True,8),(5,0),(5,17),(5,True)]:
            with self.assertRaises(ValueError): leadup_context(analysis,[ply],count)

    def test_castling_and_en_passant_replay_actual_legal_history(self):
        analysis = analysis_for('1. e4 a6 2. e5 d5 3. exd6 cxd6 4. Nf3 Nf6 5. Be2 e6 6. O-O *')
        context = leadup_context(analysis,[12],16)
        castling = context['moves'][-1]['changes']['white']
        self.assertEqual(castling['king'],{'from':'e1','to':'g1'})
        self.assertEqual(castling['castling_rights']['removed'],['kingside','queenside'])
        board = chess.Board(context['positions']['12']['fen'])
        self.assertEqual(board.piece_at(chess.F1),chess.Piece(chess.ROOK,chess.WHITE))
        self.assertNotIn('e5',position_facts(board)['white']['isolated_pawns'])

    def test_custom_start_and_promotion_do_not_assume_normal_start_position(self):
        analysis = analysis_for('[SetUp "1"]\n[FEN "7k/P7/8/8/8/8/8/7K w - - 0 1"]\n\n1. a8=Q+ *')
        result = leadup_context(analysis,[2])
        self.assertEqual(result['positions']['1']['fen'],analysis['start_fen'])
        self.assertEqual(chess.Board(result['positions']['2']['fen']).piece_at(chess.A8),chess.Piece(chess.QUEEN,chess.WHITE))

    def test_positional_features_are_concrete_squares_not_judgments(self):
        board = chess.Board('4r1k1/8/8/8/2P5/2P5/P3N3/4K3 w - - 0 1')
        facts = position_facts(board)
        self.assertEqual(facts['white']['doubled_pawn_files'],['c:2'])
        self.assertEqual(facts['white']['isolated_pawns'],['a2','c3','c4'])
        self.assertEqual(facts['white']['pinned_pieces'],['e2'])
        self.assertIn('b',facts['open_files'])

    def test_decorated_tool_can_be_batched_and_does_not_search(self):
        analysis = analysis_for('1. e4 e5 2. Nf3 Nc6 *')
        engines = FakeEngines()
        self.addCleanup(engines._temp.cleanup)
        with tempfile.TemporaryDirectory() as tmp:
            library = ChessTools(analysis,engines,Path(tmp))
            calls_before = len(engines.sf_calls),len(engines.human_calls)
            result = library.call('investigate_batch',{'requests':[
                {'tool':'get_leadup','arguments':{'ply':4,'lookback_plies':2}},
                {'tool':'get_leadup','arguments':{'ply':2}}]})
            self.assertEqual([r['result']['windows'][0]['target_ply'] for r in result['results']],[4,2])
            self.assertEqual((len(engines.sf_calls),len(engines.human_calls)),calls_before)


if __name__ == '__main__':
    unittest.main()
