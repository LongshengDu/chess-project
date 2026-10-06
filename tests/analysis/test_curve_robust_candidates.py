"""Invariant checks for the fixed Laplace accuracy-noise controls."""
from copy import deepcopy
import unittest

import numpy as np

from tests.analysis import curve_robust_candidates as candidate
from tests.analysis.rating_evidence_fixture import evidence_fixture


class RobustCurveTests(unittest.TestCase):
    def setUp(self):
        self.evidence = evidence_fixture()
        self.accounts = {'White': 1600., 'Black': 1500.}
        self.prepared = candidate.prepare(self.evidence, self.accounts)

    def test_monotone_maps_and_weak_account_sensitivity(self):
        maps = [candidate.points_at(accuracy, self.prepared, self.accounts)
                for accuracy in np.linspace(0., 100., 201)]
        for method in candidate.METHODS:
            points = np.array([row[method] for row in maps])
            self.assertTrue(np.all(np.diff(points) >= -1e-8))
            self.assertTrue(np.all((0 <= points) & (points <= 3200)))
        original = candidate.predict_prepared(self.prepared, self.accounts)
        method = 'curve_laplace_common_account5'
        for side in candidate.SIDES:
            changed = candidate.predict_prepared(self.prepared, dict(self.accounts, **{side: self.accounts[side]+200}))
            for player in candidate.SIDES:
                self.assertAlmostEqual(changed[method][player]-original[method][player], 5.)

    def test_noise_scale_preserves_variance_and_cancels_above_curve(self):
        variance = self.prepared['curve']['likelihood']['accuracy_variance']
        self.assertAlmostEqual(2*np.sqrt(variance/2)**2, variance)
        # A synthetic bounded local curve makes the exact tail cancellation
        # visible within physical observed accuracy support.
        prepared = deepcopy(self.prepared)
        prepared['curve']['shared_accuracy'] = np.linspace(20., 80., len(prepared['grid'])).tolist()
        for accounts in (self.accounts, {}):
            low = candidate.points_at(85., prepared, accounts)
            high = candidate.points_at(100., prepared, accounts)
            for method in candidate.METHODS:
                self.assertAlmostEqual(low[method], high[method])

    def test_account_changes_are_read_after_prepare_and_metadata_is_ignored(self):
        evidence = deepcopy(self.evidence)
        baseline = candidate.predict(evidence, self.accounts)
        self.assertEqual(evidence, self.evidence)
        for record in evidence.values():
            record.update(actual_rating=9999, commercial_reference=0, game='not a predictor')
        self.assertEqual(candidate.predict(evidence, self.accounts), baseline)
        changed = dict(self.accounts, White=1800.)
        self.assertEqual(candidate.predict_prepared(self.prepared, changed), candidate.predict(evidence, changed))


if __name__ == '__main__':
    unittest.main()
