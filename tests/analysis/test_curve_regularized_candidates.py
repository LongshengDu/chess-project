"""Monotonicity and exact declared-model checks for regularized shared curves."""
from copy import deepcopy
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON

import numpy as np

from tests.analysis import curve_discrepancy_candidates as original
from tests.analysis import curve_regularized_candidates as candidate
from tests.analysis.rating_evidence_fixture import evidence_fixture


@unittest.skip(WITHDRAWN_TEST_REASON)
class RegularizedCurveTests(unittest.TestCase):
    def setUp(self):
        self.evidence = evidence_fixture()
        self.accounts = {'White': 1600., 'Black': 1500.}
        self.prepared = candidate.prepare(self.evidence, self.accounts)

    def test_shrinkage_and_unanchored_points_match_declared_model(self):
        base = original.prepare(self.evidence)
        expected = base['models']['discrepancy_curve_shrinkage']
        np.testing.assert_array_equal(self.prepared['model']['curve'], expected['curve'])
        self.assertEqual(self.prepared['model']['variance'], expected['variance'])
        for accuracy in (0., 50., 90., 100.):
            self.assertAlmostEqual(candidate.point(accuracy, self.prepared, {}),
                                   original.points_at(accuracy, base)['discrepancy_curve_shrinkage'])

    def test_fixed_context_map_is_monotone_and_accounts_update_after_prepare(self):
        for accounts in (self.accounts, {}, dict(self.accounts, White=1800.)):
            points = np.array([candidate.point(accuracy, self.prepared, accounts)
                               for accuracy in np.linspace(0., 100., 201)])
            self.assertTrue(np.all(np.diff(points) >= -1e-8))
            self.assertTrue(np.all((0 <= points) & (points <= 3200)))
            self.assertEqual(candidate.predict_prepared(self.prepared, accounts),
                             candidate.predict(self.evidence, accounts))

    def test_inputs_and_reference_metadata_are_ignored(self):
        before = deepcopy(self.evidence)
        expected = candidate.predict(self.evidence, self.accounts)
        self.assertEqual(self.evidence, before)
        changed = deepcopy(self.evidence)
        for record in changed.values():
            record.update(actual_rating=9999, commercial_reference=0, game='ignored')
        self.assertEqual(candidate.predict(changed, self.accounts), expected)


if __name__ == '__main__':
    unittest.main()
