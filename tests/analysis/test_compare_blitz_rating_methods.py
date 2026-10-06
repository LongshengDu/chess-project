"""Revised account ratings and honest shared-curve comparison boundaries."""
from copy import deepcopy
import unittest

import chess.pgn
import numpy as np

from tests.analysis.compare_blitz_rating_methods import current_pgn_context, intersection, intersection_ordering, metrics
from tests.analysis.test_player_rating_arithmetic_coverage import evidence_fixture


class BlitzComparisonTests(unittest.TestCase):
    def test_revised_pgn_ratings_replace_stale_overrides_without_mutating_inputs(self):
        game = chess.pgn.Game()
        game.headers.update(WhiteElo='1500', BlackElo='1600', WhiteEloEstimate='2100')
        case = {'name': 'example', 'game': game, 'evidence': evidence_fixture(),
                'analysis': {'headers': {'WhiteElo': '1900', 'BlackElo': '1800'},
                             'rating_account_overrides': {'White': 2000}}}
        before = deepcopy(case['evidence']), deepcopy(case['analysis'])
        contextual = current_pgn_context(case)
        self.assertEqual(contextual['White']['actual_rating'], 1500)
        self.assertEqual(contextual['Black']['actual_rating'], 1600)
        self.assertNotIn('WhiteEloEstimate', str(contextual))
        self.assertEqual((case['evidence'], case['analysis']), before)

    def test_native_crossing_and_extrapolation_are_actual_roots(self):
        knots = np.linspace(60, 90, 21)
        self.assertAlmostEqual(intersection(knots, 75)['estimate'], 1600)
        result = intersection(knots, 92)
        self.assertEqual(result['intersection_status'], 'extrapolated')
        self.assertGreater(result['estimate'], 2600)
        self.assertTrue(result['edge'])

    def test_unsupported_or_flat_observation_is_not_invented_endpoint_estimate(self):
        result = intersection(np.linspace(60, 90, 21), 99)
        self.assertIsNone(result['estimate'])
        self.assertEqual(result['intersection_status'], 'above_support')
        self.assertEqual(result['intersection_bound'], 3200)
        plateau = intersection(np.full(21, 90.), 90)
        self.assertIsNone(plateau['estimate'])
        self.assertEqual(plateau['intersection_interval'], [0, 3200])

    def test_native_plateau_returns_interval_instead_of_arbitrary_unique_root(self):
        knots = np.r_[np.linspace(60, 75, 10), 75., np.linspace(76, 90, 10)]
        result = intersection(knots, 75)
        self.assertIsNone(result['estimate'])
        self.assertEqual(result['intersection_status'], 'nonunique')
        self.assertEqual(result['intersection_interval'], [1500, 1600])

    def test_missing_crossings_do_not_enter_errors_or_pair_comparisons(self):
        row = {'game': 'game0', 'side': 'White', 'reference': 1500, 'actual': 1400,
               'estimate': 1510., 'display_estimate': 1510, 'edge': False}
        result = metrics([row, {**row, 'side': 'Black', 'estimate': None, 'display_estimate': None, 'edge': True}])
        self.assertEqual(result['mae'], 10)
        self.assertEqual(result['estimated_players'], 1)
        self.assertEqual(result['missing_players'], 1)
        self.assertEqual(result['white_black_comparable_games'], 0)
        self.assertEqual(metrics([row])['white_black_comparable_games'], 0)

    def test_ordering_separates_finite_small_gaps_from_unavailable_crossings(self):
        white = {'game': 'game0', 'side': 'White', 'method': 'shared_curve_intersection',
                 'reference': 1700, 'estimate': 1510., 'display_estimate': 1510,
                 'average_accuracy': 90., 'intersection_status': 'native'}
        black = {**white, 'side': 'Black', 'reference': 1500, 'estimate': 1500.,
                 'display_estimate': 1500, 'average_accuracy': 89.}
        check = intersection_ordering([white, black])
        self.assertEqual(check['finite_matches'], 1)  # A non-tie may have a gap <50.
        unavailable = {**white, 'estimate': None, 'display_estimate': None, 'intersection_status': 'above_support'}
        check = intersection_ordering([unavailable, black])
        self.assertEqual(check['finite_games'], 0)
        self.assertEqual(check['direction_only_matches'], 1)
        self.assertIsNone(check['games'][0]['intersection_gap'])
        self.assertIsNone(check['games'][0]['display_ordering_match'])
        check = intersection_ordering([{**unavailable, 'reference': 1500}, black])
        self.assertEqual(check['unresolved_games'], 1)  # Cannot infer a tie-size gap.


if __name__ == '__main__':
    unittest.main()
