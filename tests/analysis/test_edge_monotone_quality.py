"""Reference-independent shape and interpolation guarantees for projected fits."""
from copy import deepcopy
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_monotone_quality import (
    ACCURACY_STEP, _cached_mapping, accuracy_mapping, interpolate, predict,
)
from tests.analysis.test_edge_rating import make_case


class MonotoneQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = ARGS.grid
        cls.mapping = accuracy_mapping(60.+cls.grid*.01, 10., 65.+cls.grid*.01,
                                       np.full_like(cls.grid, 40.), cls.grid)
        cls.case = make_case()
        cls.calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]

    def test_entire_accuracy_domain_is_finite_and_nondecreasing(self):
        accuracy, raw, projected = self.mapping
        self.assertEqual(accuracy[0], 0.)
        self.assertEqual(accuracy[-1], 100.)
        np.testing.assert_allclose(np.diff(accuracy), ACCURACY_STEP, atol=1e-12)
        self.assertTrue(np.isfinite(raw).all() and np.isfinite(projected).all())
        self.assertTrue(np.all(np.diff(projected) >= 0))
        self.assertTrue(np.all(np.diff(interpolate(self.mapping, np.linspace(0., 100., 10001))) >= 0))

    def test_interpolation_preserves_nodes_midpoints_and_endpoints(self):
        accuracy, _, projected = self.mapping
        np.testing.assert_array_equal(interpolate(self.mapping, accuracy), projected)
        middle = (accuracy[:-1]+accuracy[1:])/2
        np.testing.assert_allclose(interpolate(self.mapping, middle), (projected[:-1]+projected[1:])/2,
                                   atol=1e-10, rtol=0)

    def test_projection_cannot_increase_squared_error_against_a_constant_fit(self):
        _, raw, projected = self.mapping
        self.assertLessEqual(float(np.sum((projected-raw)**2)), float(np.sum((raw.mean()-raw)**2))+1e-8)
        self.assertAlmostEqual(float(raw.mean()), float(projected.mean()), places=10)

    def test_base_ignores_accounts_and_blend_has_fixed_sensitivity(self):
        before = predict(self.case['evidence'], self.case['fit'], self.case['ratings'], self.calibration)
        for shift in (-200, 200):
            ratings = dict(self.case['ratings'], White=self.case['ratings']['White']+shift)
            after = predict(self.case['evidence'], self.case['fit'], ratings, self.calibration)
            self.assertEqual(before['monotone_predictive_mixture_all'], after['monotone_predictive_mixture_all'])
            self.assertAlmostEqual(after['monotone_predictive_mixture_account_5pct_all']['White']-
                                   before['monotone_predictive_mixture_account_5pct_all']['White'], shift*.05)

    def test_numeric_cache_reused_and_inputs_unchanged(self):
        original = deepcopy((self.case, self.calibration))
        predict(self.case['evidence'], self.case['fit'], self.case['ratings'], self.calibration)
        hits = _cached_mapping.cache_info().hits
        predict(self.case['evidence'], self.case['fit'], self.case['ratings'], self.calibration)
        self.assertGreater(_cached_mapping.cache_info().hits, hits)
        self.assertEqual((self.case, self.calibration), original)
        self.assertTrue(all(not values.flags.writeable for values in self.mapping))

    def test_out_of_domain_accuracy_is_rejected(self):
        for accuracy in (-.1, 100.1, np.nan):
            with self.assertRaises(ValueError):
                interpolate(self.mapping, accuracy)


if __name__ == '__main__':
    unittest.main()
