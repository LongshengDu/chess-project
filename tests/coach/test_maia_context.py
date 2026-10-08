"""Source/display coordinates must never change native Maia investigations."""
from copy import deepcopy
import io
import tempfile
import unittest
from unittest.mock import patch

import chess.pgn

from analysis.game.pipeline import analyze_game
from analysis.accuracy.performance import refresh_performance
from analysis.game.summary import compact_summary
from analysis.move_hints import add_flags
from analysis.elo_convert import convert
from coach.report_performance import insert_performance_snapshot
from coach.tools_chess import ChessTools
from tests.coach.fixtures import FakeAnalysisSession


def from_native(value, scale):
    return convert(value, 'lb', scale, extrapolate=True)


class MaiaContextTests(unittest.TestCase):
    def setUp(self):
        session = FakeAnalysisSession()
        self.addCleanup(session._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO('[Site "lichess.org"]\n[TimeControl "180+2"]\n'
            '[WhiteElo "1600"]\n[BlackElo "1700"]\n\n1. e4 e5 2. Nf3 Nc6 *'))
        self.native = analyze_game(game, session, progress=lambda _: None)
        self.source = deepcopy(self.native)
        self.source['headers'].update(Site='Chess.com', TimeControl='600+0')
        # Exact numeric overrides avoid introducing integer PGN rounding error.
        for side, rating in (('white', 1600.), ('black', 1700.)):
            self.source['game'][side]['elo'] = from_native(rating, 'cr')
        self.source['game']['rating_scale'] = 'chess_com_rapid'

    def library(self, analysis):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session = FakeAnalysisSession()
        self.addCleanup(session._temp.cleanup)
        return ChessTools(analysis, session, directory.name, side='white')

    def test_summary_preserves_native_choices_and_labels_both_scales(self):
        native, source = compact_summary(self.native, 'white'), compact_summary(self.source, 'white')
        self.assertEqual(source['critical_moments'], native['critical_moments'])
        self.assertEqual(source['overview'], native['overview'])
        self.assertAlmostEqual(source['game']['white']['maia_elo'], 1600.)
        self.assertEqual(source['game']['rating_scale'], 'chess_com_rapid')
        self.assertEqual(source['game']['maia_rating_scale'], 'lichess_blitz')
        self.assertEqual(source['coaching'], {'side': 'white'})
        self.assertEqual(source['accuracy_curve'], native['accuracy_curve'])
        result = self.library(self.source).call('get_game_analysis', {})
        self.assertEqual(result['maia_rating_scale'], 'Lichess Blitz')
        report = insert_performance_snapshot('## Performance snapshot\n', self.source)
        self.assertEqual(report, insert_performance_snapshot('## Performance snapshot\n', self.native))

    def test_native_opponent_and_higher_rating_requests_are_coordinate_invariant(self):
        native, source = self.library(self.native), self.library(self.source)
        self.assertEqual(native.opponent_elo([]), 1700)
        self.assertEqual(source.opponent_elo([]), 1700)
        one = native.explore_candidate(1, 'd2d4')
        two = source.explore_candidate(1, 'd2d4')
        self.assertEqual(one, two)
        self.assertEqual(source.session.human_calls, native.session.human_calls)
        self.assertEqual(source.session.pair_calls, native.session.pair_calls)
        self.assertEqual(two['human_replies']['reference_elo'], 1700)

    def test_cached_rating_flags_refresh_without_changing_metrics_or_tactical_flags(self):
        source = deepcopy(self.source)
        row = source['moves'][0]
        row['flags'] = ['blunder', 'sacrifice', 'only_move']
        played = next(candidate for candidate in row['candidate_moves'] if candidate['move'] == row['played']['move'])
        played['maia_p'] = {str(rating): .2 if rating >= 1500 else .01 for rating in range(600, 2601, 100)}
        source['performance'].pop('rating_context_signature', None)
        original_metrics = deepcopy(source['performance'])
        with patch('analysis.accuracy.performance.game_accuracy', side_effect=AssertionError('Reuse saved statistics')):
            refresh_performance(source)
        self.assertIn('natural_but_bad', row['flags'])
        self.assertTrue({'blunder', 'sacrifice', 'only_move'} <= set(row['flags']))
        self.assertEqual({key: value for key, value in source['performance'].items() if key != 'rating_context_signature'}, original_metrics)
        signature = source['performance']['rating_context_signature']
        source['game']['white']['elo'] = from_native(1200., 'cr')
        refresh_performance(source)
        self.assertNotIn('natural_but_bad', row['flags'])
        self.assertNotEqual(source['performance']['rating_context_signature'], signature)
        self.assertTrue({'blunder', 'sacrifice', 'only_move'} <= set(row['flags']))
        stable = deepcopy(source)
        refresh_performance(source)
        self.assertEqual(source, stable)

    def test_full_move_hints_use_native_ratings_on_either_source_scale(self):
        native, source = deepcopy(self.native), deepcopy(self.source)
        for data in (native, source):
            add_flags(data, {})
        self.assertEqual([row['flags'] for row in native['moves']], [row['flags'] for row in source['moves']])


if __name__ == '__main__':
    unittest.main()
