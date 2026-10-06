"""Structural checks for the reference-free quality-distribution experiments."""
from copy import deepcopy
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import summarize
from tests.analysis import edge_distribution_transport as transport
from tests.analysis.test_edge_rating import make_case


class EdgeDistributionTransportTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]

    def predictions(self, case=None, calibration=None, *, edge_only=True):
        case = self.case if case is None else case
        calibration = self.calibration if calibration is None else calibration
        return transport.predict(case['evidence'], case['fit'], case['ratings'], calibration,
                                 edge_only=edge_only)

    def assert_predictions_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for method in first:
            for side in ('White', 'Black'):
                self.assertAlmostEqual(first[method][side], second[method][side], places=9,
                                       msg=f'{method}/{side}')

    def test_edge_gating_preserves_intersecting_estimate_exactly(self):
        self.assertTrue(transport.is_edge(self.case['fit'], 'White'))
        self.assertFalse(transport.is_edge(self.case['fit'], 'Black'))
        for values in self.predictions().values():
            self.assertEqual(values['Black'], self.case['fit']['players']['Black']['estimate'])
            self.assertTrue(np.isfinite(values['White']))
            self.assertGreaterEqual(values['White'], 600.)
            self.assertLessEqual(values['White'], 2600.)
        all_players = self.predictions(edge_only=False)
        self.assertTrue(any(pair['Black'] != self.case['fit']['players']['Black']['estimate']
                            for pair in all_players.values()))

    def test_inputs_are_immutable(self):
        snapshot = deepcopy((self.case, self.calibration))
        self.predictions()
        self.assertEqual((self.case, self.calibration), snapshot)

    def test_account_perturbations_have_exact_bounded_influence(self):
        original = self.predictions(edge_only=False)
        for side in ('White', 'Black'):
            other = 'Black' if side == 'White' else 'White'
            for shift in (-200., 200.):
                changed = deepcopy(self.case)
                changed['ratings'][side] += shift
                prediction = self.predictions(changed, edge_only=False)
                for method in original:
                    self.assertAlmostEqual(prediction[method][side] - original[method][side],
                                           .05 * shift, places=10)
                    self.assertEqual(prediction[method][other], original[method][other])

    def test_calibration_ignores_played_moves_ratings_and_references(self):
        original = self.predictions()
        changed = deepcopy(self.calibration)
        for case in changed:
            case['ratings'] = {'White': -10000., 'Black': 10000.}
            case['fit'] = {'reference': 'must not enter inference'}
            case['reference'] = {'White': 9999., 'Black': -9999.}
            for record in case['evidence'].values():
                record['reference'] = 'ignored'
                for row in record['observations']:
                    row['played_index'] = 'deliberately invalid, unused calibration label'
        self.assert_predictions_equal(original, self.predictions(calibration=changed))

    def test_target_reference_metadata_is_ignored(self):
        changed = deepcopy(self.case)
        changed['reference'] = {'White': 9999., 'Black': -9999.}
        changed['fit']['reference'] = 'ignored'
        for side in ('White', 'Black'):
            changed['evidence'][side]['reference'] = 12345
            changed['fit']['players'][side]['reference'] = -12345
        self.assert_predictions_equal(self.predictions(), self.predictions(changed))

    def test_calibration_game_and_side_permutation_is_invariant(self):
        changed = deepcopy(self.calibration[::-1])
        for case in changed:
            case['evidence'] = {'White': case['evidence']['Black'], 'Black': case['evidence']['White']}
        self.assert_predictions_equal(self.predictions(), self.predictions(calibration=changed))

    def test_target_side_swap_swaps_predictions(self):
        original = self.predictions()
        changed = deepcopy(self.case)
        for field in ('evidence', 'ratings'):
            changed[field] = {'White': self.case[field]['Black'], 'Black': self.case[field]['White']}
        changed['fit'] = summarize(changed['evidence'])
        swapped = self.predictions(changed)
        self.assert_predictions_equal(original, {
            method: {'White': pair['Black'], 'Black': pair['White']} for method, pair in swapped.items()})

    def test_move_order_is_invariant_for_target_and_calibration(self):
        changed, calibration = deepcopy((self.case, self.calibration))
        for case in [changed, *calibration]:
            for record in case['evidence'].values():
                record['observations'].reverse()
        changed['fit'] = summarize(changed['evidence'])
        self.assert_predictions_equal(self.predictions(), self.predictions(changed, calibration))

    def test_legal_move_permutation_is_invariant(self):
        changed, calibration = deepcopy((self.case, self.calibration))
        order = [2, 0, 1]
        for case in [changed, *calibration]:
            for record in case['evidence'].values():
                for row in record['observations']:
                    row['played_index'] = order.index(row['played_index'])
                    row['qualities'] = {view: [q[i] for i in order] for view, q in row['qualities'].items()}
                    row['maia_probabilities'] = [[p[i] for i in order] for p in row['maia_probabilities']]
        changed['fit'] = summarize(changed['evidence'])
        self.assert_predictions_equal(self.predictions(), self.predictions(changed, calibration))


if __name__ == '__main__':
    unittest.main()
