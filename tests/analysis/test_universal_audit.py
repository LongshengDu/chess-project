"""Scoring, probability-normalization and account-decision audit checks."""
from unittest.mock import patch
import unittest

import numpy as np

from tests.analysis.experiment_universal_rating import score
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass
from tests.analysis import universal_account_sensitivity


class UniversalAuditTests(unittest.TestCase):
    def test_strict_unrounded_target_is_distinct_from_displayed_integer_target(self):
        rows = [{'method': 'candidate', 'game': 'synthetic', 'side': 'White', 'estimate': 2099.95,
                 'reference': 2000., 'original_estimate': 2100., 'edge': True},
                {'method': 'candidate', 'game': 'synthetic', 'side': 'Black', 'estimate': 1599.95,
                 'reference': 1500., 'original_estimate': 1600., 'edge': False}]
        result = score(rows)[0]
        self.assertAlmostEqual(result['mae'], 99.95)
        self.assertEqual(result['rounded_mae'], 100.)
        self.assertTrue(result['meets_required_targets'])
        self.assertFalse(result['meets_rounded_error_targets'])

    def test_order_acceptance_uses_original_model_not_commercial_label(self):
        rows = [{'method': 'candidate', 'game': 'synthetic', 'side': 'White', 'estimate': 2100.,
                 'reference': 1000., 'original_estimate': 2000., 'edge': True},
                {'method': 'candidate', 'game': 'synthetic', 'side': 'Black', 'estimate': 1800.,
                 'reference': 2500., 'original_estimate': 1500., 'edge': False}]
        self.assertEqual(score(rows)[0]['ordering_matches'], 1)

    def test_gaussian_beta_measurement_bins_are_both_normalized(self):
        accuracy = np.linspace(0., 100., 10001)[:, None]
        means, variances = np.array([80., 90., 98.]), np.array([9., 25., 2.])
        for likelihood in (gaussian_accuracy_mass, beta_accuracy_mass):
            mass = likelihood(accuracy, means, variances)
            self.assertTrue(np.isfinite(mass).all())
            self.assertTrue(np.all(mass >= 0))
            np.testing.assert_allclose(mass.sum(axis=0), 1., atol=2e-10)

    def test_account_blend_is_applied_before_one_order_projection(self):
        fit = {'players': {'White': {'average_accuracy': 90.}, 'Black': {'average_accuracy': 80.}}}
        ratings = {'White': 1500., 'Black': 1700.}
        with patch.object(universal_account_sensitivity, 'unprojected_quality',
                          return_value={'White': 1800., 'Black': 2000.}):
            output = universal_account_sensitivity.predict({}, fit, ratings, [])
        #10% blend ->1770/1970, then nearest ordered pair1870.5/1869.5.
        self.assertEqual(output['universal_account_100'], {'White': 1870.5, 'Black': 1869.5})

    def test_own_and_opponent_rating_sensitivity_stays_bounded_even_at_projection_boundary(self):
        fit = {'players': {'White': {'average_accuracy': 90.}, 'Black': {'average_accuracy': 80.}}}
        ratings = {'White': 1500., 'Black': 1700.}
        for quality in ({'White': 2200., 'Black': 1800.}, {'White': 1800., 'Black': 1780.},
                        {'White': 1800., 'Black': 2000.}):
            with patch.object(universal_account_sensitivity, 'unprojected_quality', return_value=quality):
                baseline = universal_account_sensitivity.predict({}, fit, ratings, [])['universal_account_100']
                for side in ('White', 'Black'):
                    other = 'Black' if side == 'White' else 'White'
                    for shift in (-200., 200.):
                        changed = {**ratings, side: ratings[side]+shift}
                        result = universal_account_sensitivity.predict({}, fit, changed, [])['universal_account_100']
                        self.assertLessEqual(abs(result[side]-baseline[side]), 20.+1e-9)
                        self.assertLessEqual(abs(result[other]-baseline[other]), 10.+1e-9)


if __name__ == '__main__':
    unittest.main()
