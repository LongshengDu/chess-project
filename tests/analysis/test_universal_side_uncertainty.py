"""Numerical conditional variance and ensemble constraint checks."""
import unittest
from unittest.mock import patch

import numpy as np

from analysis.player_rating.bayesian_shared_curve import GRID
from tests.analysis.universal_side_uncertainty import measurement_variance
from tests.analysis.universal_uncertainty_consensus import predict


class SideUncertaintyTests(unittest.TestCase):
    def test_variance_matches_weighted_independent_mean(self):
        def row(probability, qualities, policy):
            return {'position_win_probability': probability,
                    'qualities': {'position': qualities},
                    'maia_probabilities': [policy]*len(GRID)}
        record = {'observations': [row(.5, [0., 100.], [.5, .5]),
                                   row(.2, [50., 100.], [.5, .5]),
                                   row(.5, [100.], [1.])]}
        # Competitiveness weights are 1 and .8; the forced move is excluded.
        expected = (2500.+.8**2*625.)/1.8**2
        np.testing.assert_allclose(measurement_variance(record, .5), expected)
        np.testing.assert_allclose(measurement_variance(record, 0.), (2500.+625.)/4)

    def test_pooling_precedes_account_and_order_constraint(self):
        base = 'tests.analysis.universal_uncertainty_consensus.'
        fit = {'players': {'White': {'average_accuracy': 90}, 'Black': {'average_accuracy': 80}}}
        with patch(base+'unprojected_quality', return_value={'White': 1800., 'Black': 2000.}), \
             patch(base+'quality_estimates', return_value={
                 'side_uncertainty_average': {'White': 2000., 'Black': 1900.}}):
            result = predict({}, fit, {'White': 2000., 'Black': 1000.}, [])
        self.assertEqual(result['uncertainty_consensus'], {'White': 1910., 'Black': 1855.})


if __name__ == '__main__':
    unittest.main()
