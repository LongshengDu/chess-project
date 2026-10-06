"""Reference-free shape, input and account-sensitivity checks."""
from copy import deepcopy
from dataclasses import replace
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON
from unittest.mock import patch

import numpy as np

from analysis.player_rating.calibration import load_calibration
from tests.analysis import curve_discrepancy_candidates as candidate
from tests.analysis.rating_evidence_fixture import evidence_fixture


@unittest.skip(WITHDRAWN_TEST_REASON)
class DiscrepancyTests(unittest.TestCase):
    def setUp(self):
        self.evidence = evidence_fixture()
        self.accounts = {'White': 1600., 'Black': 1500.}

    def test_inputs_are_immutable_and_metadata_cannot_change_points(self):
        original = deepcopy(self.evidence)
        prepared = candidate.prepare(self.evidence, self.accounts)
        points = candidate.predict_prepared(prepared, self.accounts)
        self.assertEqual(self.evidence, original)
        changed = deepcopy(self.evidence)
        for record in changed.values():
            record.update(actual_rating=9999, commercial_reference=9999, game='ignored')
        self.assertEqual(candidate.predict(changed, self.accounts), points)

    def test_every_mapping_is_nondecreasing_and_bounded(self):
        prepared = candidate.prepare(self.evidence, self.accounts)
        values = [candidate.points_at(accuracy, prepared) for accuracy in np.linspace(0, 100, 201)]
        for method in candidate.describe():
            points = np.array([value[method] for value in values])
            self.assertTrue(np.isfinite(points).all())
            self.assertTrue(np.all((0 <= points) & (points <= 3200)))
            self.assertTrue(np.all(np.diff(points) >= -1e-8))

    def test_common_account_anchor_preserves_order_with_bounded_sensitivity(self):
        prepared = candidate.prepare(self.evidence, self.accounts)
        baseline = candidate.predict_prepared(prepared, self.accounts)
        for side in candidate.SIDES:
            for change in (-200, 200):
                accounts = dict(self.accounts, **{side: self.accounts[side]+change})
                shifted = candidate.predict_prepared(prepared, accounts)
                for method, pair in shifted.items():
                    self.assertGreater(pair['White'], pair['Black'])
                    self.assertAlmostEqual(pair['White']-pair['Black'], baseline[method]['White']-baseline[method]['Black'])
                    for player in candidate.SIDES:
                        self.assertAlmostEqual(abs(pair[player]-baseline[method][player]), 5.)

    def test_competitive_population_fields_are_not_used(self):
        corpus = load_calibration()
        missing = replace(corpus, records=tuple(replace(record, competitive=None) for record in corpus.records))
        baseline = candidate.predict(self.evidence, self.accounts)
        with patch.object(candidate, 'load_calibration', return_value=missing):
            self.assertEqual(candidate.predict(self.evidence, self.accounts), baseline)


if __name__ == '__main__':
    unittest.main()
