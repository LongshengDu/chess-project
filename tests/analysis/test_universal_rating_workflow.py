"""Acceptance and adapter regressions for the universal rating experiments."""
import unittest
from unittest.mock import patch

from tests.analysis.experiment_universal_rating import score
from tests.analysis.universal_coverage_consensus import predict


class UniversalRatingWorkflowTests(unittest.TestCase):
    def test_error_thresholds_do_not_hide_ordering_failure(self):
        rows = [dict(game='toy', side='White', method='reversed', estimate=1501,
                     reference=1503, original_estimate=1503, edge=True),
                dict(game='toy', side='Black', method='reversed', estimate=1502,
                     reference=1500, original_estimate=1500, edge=False)]
        result = score(rows)[0]
        self.assertLess(result['mae'], 100)
        self.assertFalse(result['meets_required_targets'])
        self.assertEqual(result['ordering_matches'], 0)

    def test_maximum_error_aim_is_reported_separately(self):
        rows = []
        for game in range(4):
            for side, value in (('White', 1800), ('Black', 1600)):
                rows.append(dict(game=str(game), side=side, method='candidate',
                                 estimate=value+(210 if game == 0 and side == 'White' else 0),
                                 reference=value, original_estimate=value,
                                 edge=game > 0))
        result = score(rows)[0]
        self.assertTrue(result['meets_required_targets'])
        self.assertFalse(result['meets_maximum_aim'])

    def test_consensus_adapts_unprefixed_native_coverage_api(self):
        fit = {'players': {'White': {'average_accuracy': 90}, 'Black': {'average_accuracy': 80}}}
        base = 'tests.analysis.universal_coverage_consensus.'
        with patch(base+'weighted_fit', return_value=fit), \
             patch(base+'universal_native_coverage.predict', side_effect=[
                 {'native_coverage_mean_account_5pct_all': {'White': 2000, 'Black': 1700}},
                 {'native_coverage_mean_account_5pct_all': {'White': 2100, 'Black': 1800}}]), \
             patch(base+'universal_competitiveness.predict', return_value={
                 'competitive_sqrt_mean_account': {'White': 1800, 'Black': 1600}}), \
             patch(base+'universal_posterior_decision.predict', return_value={
                 'mixture_mean_account': {'White': 1900, 'Black': 1500}}):
            result = predict({}, fit, {'White': 1600, 'Black': 1600}, [{'evidence': {}}])
        self.assertEqual(result['coverage_factorial_consensus'], {'White': 1950, 'Black': 1650})

    def test_rounding_cannot_hide_a_threshold_boundary(self):
        rows = [dict(game='toy', side=side, method='boundary', estimate=value+99.96,
                     reference=value, original_estimate=value, edge=True)
                for side, value in (('White', 1800), ('Black', 1600))]
        result = score(rows)[0]
        self.assertTrue(result['meets_required_targets'])
        self.assertEqual(result['rounded_mae'], 100.)
        self.assertFalse(result['meets_rounded_error_targets'])


if __name__ == '__main__':
    unittest.main()
