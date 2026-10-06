"""Analytic-Jacobian and monotonic decision checks without commercial data."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import GRID, SharedCurve
from tests.analysis import curve_information_candidates as candidate
from tests.analysis.rating_evidence_fixture import evidence_fixture


class InformationPriorTests(unittest.TestCase):
    def test_analytic_derivative_matches_native_and_tail_finite_differences(self):
        model = SharedCurve(65.+25.*((GRID-GRID[0])/(GRID[-1]-GRID[0]))**1.2)
        ratings = np.array([300., 599., 650., 850., 1234., 2575., 2601., 2900.])
        h = .001
        expected = (model(ratings+h)-model(ratings-h))/(2*h)
        np.testing.assert_allclose(candidate.curve_derivative(model, ratings), expected, rtol=1e-6, atol=1e-10)
        derivative = candidate.curve_derivative(model, np.linspace(0., 3200., 1000))
        self.assertTrue(np.all(derivative >= 0))

    def test_flat_curve_and_forced_only_data_produce_no_estimate(self):
        model = SharedCurve(np.full(len(GRID), 75.))
        np.testing.assert_array_equal(candidate.curve_derivative(model, np.linspace(0., 3200., 101)), 0.)
        evidence = evidence_fixture()
        for record in evidence.values():
            for row in record['observations']:
                row['qualities'] = {'position': [75., 75., 75.], 'root': [75., 75., 75.]}
        prepared = candidate.prepare(evidence)
        self.assertFalse(prepared['identifiable'])
        self.assertEqual(candidate.predict_prepared(prepared, {}), {candidate.METHOD: {'White': None, 'Black': None}})
        for record in evidence.values():
            record['observations'] = [{'played_index': 0, 'qualities': {'position': [100.], 'root': [100.]},
                                       'maia_probabilities': [[1.]]*21}]
        self.assertEqual(candidate.predict(evidence, {}), {candidate.METHOD: {'White': None, 'Black': None}})

    def test_prior_normalized_monotonic_and_account_bound(self):
        prepared = candidate.prepare(evidence_fixture())
        accounts = {'White': 1600., 'Black': 1500.}
        self.assertAlmostEqual(trapezoid(prepared['prior'], prepared['grid']), 1.)
        points = np.array([candidate.point(accuracy, prepared, accounts) for accuracy in np.linspace(0., 100., 201)])
        self.assertTrue(np.all(np.diff(points) >= -1e-8))
        self.assertTrue(np.all((0 <= points) & (points <= 3200)))
        original = candidate.predict_prepared(prepared, accounts)[candidate.METHOD]
        changed = candidate.predict_prepared(prepared, dict(accounts, White=1800.))[candidate.METHOD]
        for side in candidate.SIDES:
            self.assertAlmostEqual(changed[side]-original[side], 5.)

    def test_inputs_and_metadata_do_not_affect_reference_free_estimation(self):
        evidence = evidence_fixture()
        before = deepcopy(evidence)
        expected = candidate.predict(evidence, {})
        self.assertEqual(evidence, before)
        for record in evidence.values():
            record.update(commercial_reference=9999, actual_rating=9999, game='ignored')
        self.assertEqual(candidate.predict(evidence, {}), expected)


if __name__ == '__main__':
    unittest.main()
