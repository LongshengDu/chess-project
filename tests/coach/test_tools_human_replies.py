"""Human reply evidence precedes strongest-defense analysis; no live models."""
import json
import tempfile
import unittest

from coach.tools_evidence import prepare_initial_evidence
from pathlib import Path
from unittest.mock import patch

import chess

from coach.tools_chess import ChessTools
from coach.tools_evidence import compact_evidence
from analysis.game.metadata import game_metadata
from tests.coach.fixtures import FakeAnalysisSession
from tests.analysis.test_move_hints import row_for


class HumanReplyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = json.loads((Path(__file__).with_name('data')/'game1-qxe4-replies.json').read_text(encoding='utf-8'))
        self.session = FakeAnalysisSession(fixture['fen'])
        self.addCleanup(self.session._temp.cleanup)
        first = row_for(.99, .99, fixture['fen'], 'c2e4')
        board = chess.Board(fixture['fen']); board.push_uci('c2e4')
        second = row_for(.01, .01, board.fen(), 'a5a4')
        second['ply'] = 2
        first.update(label='20. Qxe4',stage='middlegame')
        second.update(label='20... a4',stage='middlegame')
        second['candidate_moves'] = fixture['replies']
        second['maia'] = {rating: {
            'moves': [{k: c[k] for k in ('move', 'san', 'eval', 'loss')} | {'p': c['maia_p'][rating]}
                      for c in sorted(fixture['replies'], key=lambda c: -c['maia_p'][rating])[:5]],
            'expected_accuracy': 100., 'absolute_deviation': 0.}
                          for rating in ('1700','2000','2200','2600')}
        self.analysis = {'start_fen': fixture['fen'], 'moves': [first,second],
            'game': game_metadata({'WhiteElo': '1600', 'BlackElo': '1700'}),
            'coaching': {}}
        self.library = ChessTools(self.analysis, self.session, temporary.name, side='white')

    def test_qxe4_immediate_human_replies_preserve_likelihood_and_verify_mates_locally(self):
        with patch.object(self.session, 'sf', side_effect=AssertionError('Reuse saved reply scores')), \
             patch.object(self.session, 'human_pairs', side_effect=AssertionError('Reuse saved Maia')):
            result = self.library.human_replies(1, ['c2e4'], 'f8e8', 100)
        self.assertEqual(result['position'], {'ply':1,'line':['c2e4']})
        self.assertEqual(result['reference_elo'],1700)
        self.assertEqual(result['conditioning'],'equal_rating')
        knight = result['moves'][0]
        self.assertEqual(knight['move'],'f6e4')
        self.assertEqual(knight['maia_p']['1700'], .84087)
        self.assertEqual(knight['maia_p']['2000'], .5831)
        self.assertEqual(knight['eval'],'#1')
        self.assertEqual({m['san'] for m in knight['allows_mate_in_one']},{'Bh7#','Rh8#'})
        self.assertEqual(result['engine_reply_p']['1700'],.053755)
        self.assertEqual(result['engine_reply_p']['2600'],.54569)
        for move in result['moves']:
            for mate in move.get('allows_mate_in_one',[]):
                board = self.session.board(['c2e4',move['move'],mate['move']])
                self.assertTrue(board.is_checkmate())
        self.assertLess(result['covered_probability']['1700'],1)

    def test_hypothetical_choice_batches_equal_rating_policies_and_bounds_verification(self):
        result = self.library.human_replies(1,['g6e4'],'f8e8',100)
        self.assertLessEqual(len(result['moves']),5)
        self.assertEqual(len(self.session.pair_calls),1)
        history, own, opponents = self.session.pair_calls[0]
        self.assertEqual(history,['g6e4'])
        self.assertEqual(own,opponents)
        self.assertEqual(len(self.session.sf_calls),1)
        self.assertEqual(self.session.sf_calls[0][0],history)
        self.assertEqual(self.session.sf_calls[0][2],len(result['moves']))
        self.assertEqual(set(self.session.sf_calls[0][3]),{m['move'] for m in result['moves']})

    def test_both_comparison_branches_include_human_replies_and_compact_vectors(self):
        # Fake SF's arbitrary legal defense is deliberately not the main human reply.
        result = self.library.compare_played_vs_candidate(1,'g6e4')
        for key in ('played','candidate'):
            self.assertIsNotNone(result[key]['human_replies'])
            self.assertEqual(result[key]['human_replies']['position']['line'],[result[key]['stockfish']['move']])
        compact = compact_evidence(result,1600)
        human = compact['played']['human_replies']
        knight = next(m for m in human['moves'] if m['move']=='f6e4')
        self.assertEqual(knight['p'][human['ratings'].index(2000)],.5831)
        self.assertEqual(knight['allows_mate_in_one'],result['played']['human_replies']['moves'][0]['allows_mate_in_one'])
        self.assertLess(len(json.dumps(compact)),len(json.dumps(result)))

    def test_repeated_human_payload_is_referenced_but_full_result_remains_available(self):
        first = self.library.call('explore_candidate',{'ply':1,'candidate':'c2e4'})
        with patch.object(self.session, 'sf', side_effect=AssertionError('Do not repeat search')):
            second = self.library.call('explore_candidate',{'ply':1,'candidate':'c2e4'})
        self.assertEqual(second['human_replies'],{'reply_ref':first['human_replies']['reply_id']})
        full = self.library.results[second['result_id']]['human_replies']
        self.assertIn('moves',full)

    def test_terminal_branch_skips_maia_and_search(self):
        # Extend the fixture history only through supplied legal branches.
        with patch.object(self.session, 'sf', side_effect=AssertionError('Terminal position')), \
             patch.object(self.session, 'human_pairs', side_effect=AssertionError('Terminal position')):
            self.assertIsNone(self.library.human_replies(1,['c2e4','f6e4','g6h7'],None,100))

    def test_prepared_alternative_is_human_plausible_not_automatically_engine_best(self):
        row = self.analysis['moves'][0]
        for candidate in row['candidate_moves']:
            candidate['eval'] = 10.0 if candidate['move'] == 'g6e4' else 0.0
            candidate['maia_p'] = {str(r): .6 if candidate['move'] == 'e1f1' else .001
                                   for r in range(1000,2601,100)}
        with patch('coach.tools_evidence.compact_summary',return_value={'critical_moments':[{'ply':1}]}), \
             patch('coach.tools_evidence.leadup_context',return_value={}), \
             patch.object(self.library,'call',return_value={}) as call:
            prepare_initial_evidence(self.library)
        self.assertEqual(call.call_args.args,('compare_played_vs_candidate',{'ply':1,'candidate':'e1f1'}))


if __name__ == '__main__':
    unittest.main()
