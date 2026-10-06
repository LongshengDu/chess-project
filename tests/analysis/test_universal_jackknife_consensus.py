"""Constrained GLS covariance/weight invariants independent of references."""
import unittest

import numpy as np

from tests.analysis.universal_jackknife_consensus import simplex_gls_weight


class JackknifeConsensusTests(unittest.TestCase):
    def test_independent_model_variances_give_inverse_variance_weights(self):
        first = np.array([1., -1., 1., -1.])
        second = np.array([2., 2., -2., -2.])
        points = np.repeat(np.stack((first, second), axis=-1)[:, None, :], 2, axis=1)
        weight, covariance = simplex_gls_weight(points)
        self.assertAlmostEqual(weight, .8, places=12)
        self.assertAlmostEqual(covariance[0, 1], 0., places=12)
        self.assertAlmostEqual(covariance[1, 1]/covariance[0, 0], 4., places=12)

    def test_perfectly_identical_uncertainty_keeps_equal_opinions(self):
        points = np.repeat(np.arange(5.)[:, None, None], 2, axis=1)
        points = np.repeat(points, 2, axis=2)
        self.assertEqual(simplex_gls_weight(points)[0], .5)

    def test_interchanging_models_complements_weight(self):
        generator = np.random.default_rng(41)
        points = generator.normal(size=(15, 2, 2))*np.array([1., 2.])
        original, covariance = simplex_gls_weight(points)
        swapped, swapped_covariance = simplex_gls_weight(points[:, :, ::-1])
        self.assertAlmostEqual(original+swapped, 1., places=12)
        np.testing.assert_allclose(covariance, swapped_covariance[::-1, ::-1], atol=1e-12)
        self.assertTrue(0 <= original <= 1)

    def test_side_interchange_and_constant_rating_offsets_do_not_change_weight(self):
        generator = np.random.default_rng(15)
        points = generator.normal(size=(15, 2, 2))
        original, _ = simplex_gls_weight(points)
        transformed, _ = simplex_gls_weight(points[:, ::-1]+np.array([[1500., 1000.], [2000., 1700.]]))
        self.assertAlmostEqual(original, transformed, places=10)


if __name__ == '__main__':
    unittest.main()
