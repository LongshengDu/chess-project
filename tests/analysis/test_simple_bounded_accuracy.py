"""Monotonicity, bounded support and account behavior of the single Beta model."""
from copy import deepcopy
import unittest

import numpy as np

from tests.analysis import simple_bounded_accuracy as model
from tests.analysis.test_simple_curve_account import fixture


class SimpleBoundedAccuracyTests(unittest.TestCase):
    def test_posterior_is_finite_and_monotone_across_accuracy_endpoints(self):
        prepared = model.prepare(fixture())
        accuracy = np.r_[0., .001, np.linspace(.1, 99.9, 201), 99.999, 100.]
        points = np.array([model.posterior_mean(value, prepared) for value in accuracy])
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue(np.all((points >= 0) & (points <= 3200)))
        self.assertGreaterEqual(np.diff(points).min(), -1e-7)
        self.assertTrue(np.isfinite(model._log_bin_mass(0., np.array([1000.]), np.array([1000.]))).all())
        self.assertTrue(np.isfinite(model._log_bin_mass(100., np.array([1000.]), np.array([1000.]))).all())

    def test_shared_account_anchor_preserves_ties_and_has_ten_elo_sensitivity(self):
        evidence = fixture()
        evidence['Black'] = deepcopy(evidence['White'])
        accounts = {'White': 1600., 'Black': 1800.}
        before = model.predict(evidence, accounts)[model.METHOD]
        after = model.predict(evidence, {**accounts, 'White': 1800.})[model.METHOD]
        self.assertEqual(before['White'], before['Black'])
        self.assertEqual(after['White'], after['Black'])
        for side in model.SIDES:
            self.assertAlmostEqual(after[side]-before[side], 10.)

    def test_reference_metadata_and_missing_accounts_do_not_manufacture_ratings(self):
        evidence = fixture()
        expected = model.predict(evidence, {})
        for record in evidence.values():
            record['reference'] = 99999
        self.assertEqual(model.predict(evidence, {}), expected)
        evidence['Black']['observations'] = []
        result = model.predict(evidence, {})[model.METHOD]
        self.assertIsNotNone(result['White'])
        self.assertIsNone(result['Black'])


if __name__ == '__main__':
    unittest.main()
