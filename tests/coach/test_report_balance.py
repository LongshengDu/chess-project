"""General whole-game coverage, without favored moves or game-specific rules."""
import copy
import io
import unittest
from unittest.mock import patch

import chess.pgn

from analysis.game.summary import candidates_for_investigation, compact_summary
from tests.analysis.test_move_hints import cp_for, row_for


def game_analysis():
    game = chess.pgn.read_game(io.StringIO(
        '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 '
        '6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 '
        '11. Nbd2 Bb7 12. Bc2 Re8 *'))
    rows, board = [], game.board()
    for ply, move in enumerate(game.mainline_moves(),1):
        row = row_for(.5,.5,board.fen(),move.uci())
        row.update(ply=ply,label=board.san(move),stage='middlegame',flags=[])
        for rating, choices in row['maia'].items():
            row['maia'][rating] = [{k:c[k] for k in ('move','san','eval','loss')} | {'p':c['maia_p'][rating]} for c in choices]
        rows.append(row); board.push(move)
    return {'headers':{'WhiteElo':'1400','BlackElo':'1700'},'selected_player':{'side':'white','actual_elo':1400},
        'played_elo':{'white':{'estimate':1500},'black':{'estimate':1800}},'moves':rows}


class ReportBalanceTests(unittest.TestCase):
    def test_winning_pawn_loss_does_not_outrank_meaningful_outcome_loss(self):
        analysis = game_analysis()
        winning, meaningful = analysis['moves'][0], analysis['moves'][2]
        winning['flags'] = ['sacrifice']
        winning['position_eval'],winning['played']['eval'],winning['played']['loss'] = 20.,16.,4.
        meaningful['played']['eval'],meaningful['played']['loss'] = cp_for(.35),-cp_for(.35)
        meaningful['flags'] = ['mistake']
        for row in (winning,meaningful):
            next(c for c in row['candidate_moves'] if c['move']==row['played']['move']).update(row['played'])
        moments = {m['ply']:m for m in candidates_for_investigation(analysis['moves'],'white',1400,analysis['played_elo'],analysis['headers'])}
        self.assertNotIn('objective_loss',moments[1]['reasons'])
        self.assertNotIn('attainable_human_alternative',moments[1]['reasons'])
        self.assertIn('objective_loss',moments[3]['reasons'])
        self.assertGreater(moments[3]['priority'],moments[1]['priority'])

    def test_continuing_mate_does_not_get_a_new_decisive_bonus_each_move(self):
        analysis = game_analysis()
        for ply in (17,19,21,23):
            row = analysis['moves'][ply-1]
            row['position_eval'],row['played']['eval'],row['played']['loss'] = '#3','#4',None
            next(c for c in row['candidate_moves'] if c['move']==row['played']['move']).update(row['played'])
        analysis['moves'][8]['flags'] = ['sacrifice']
        moments = {m['ply']:m for m in candidates_for_investigation(analysis['moves'],'white',1400,analysis['played_elo'],analysis['headers'])}
        for ply in (17,19,21,23):
            self.assertLess(moments[ply]['priority'],moments[9]['priority'])

    def test_soft_spacing_keeps_major_neighbor_and_covers_stages_chronologically(self):
        analysis = game_analysis()
        analysis['moves'][0]['stage'] = 'opening'
        analysis['moves'][-2]['stage'] = 'endgame'
        moments = [{'ply':p,'priority':score,'stage':analysis['moves'][p-1]['stage']} for p,score in
                   [(9,10),(11,9),(13,8),(21,7)]]
        with patch('analysis.game.summary.candidates_for_investigation',return_value=moments):
            summary = compact_summary(analysis)
        plies = [m['ply'] for m in summary['critical_moments']]
        self.assertEqual(plies,sorted(plies))
        self.assertIn(21,plies)  # Spread out rather than only 9,11,13.
        self.assertEqual({m['stage'] for m in summary['critical_moments']},{'opening','middlegame','endgame'})
        moments[1]['priority'] = 30
        with patch('analysis.game.summary.candidates_for_investigation',return_value=moments):
            self.assertIn(11,[m['ply'] for m in compact_summary(analysis)['critical_moments']])

    def test_overview_exposes_saved_human_reply_even_for_unselected_moments(self):
        analysis = game_analysis()
        original = copy.deepcopy(analysis)
        with patch('analysis.game.summary.candidates_for_investigation',return_value=[]):
            summary = compact_summary(analysis)
        self.assertNotIn(19,[m['ply'] for m in summary['critical_moments']])
        expected = analysis['moves'][19]['maia']['1700'][0]
        self.assertEqual(summary['overview'][18][4],[1700,expected['san'],expected['p'],expected['eval']])
        self.assertIsNone(summary['overview'][-1][4])
        self.assertEqual(analysis,original)  # No flag/evidence mutation or searches.


if __name__ == '__main__':
    unittest.main()
