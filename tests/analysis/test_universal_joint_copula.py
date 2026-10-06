"""Numerical normalization and tail checks for the bounded copula experiment."""
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

from tests.analysis.universal_joint_copula import copula_log_density, likelihood


class JointCopulaTests(unittest.TestCase):
    def test_zero_correlation_is_independence(self):
        first = np.array([-5., -1., 0., 1., 5.])
        second = np.array([3., 2., 0., -2., -3.])
        np.testing.assert_allclose(copula_log_density(first, second, 0.), 0., atol=1e-14)

    def test_conditional_copula_integrates_to_one(self):
        # du=phi(z)dz; integrating c(u,v)phi(z) over z verifies each conditional
        # integrates to1. This also proves the joint copula normalizes to1.
        for rho in (-.7, 0., .5, .9, .99):
            for v in (.05, .5, .95):
                z_second = ndtri(v)
                result, error = quad(lambda z: float(np.exp(copula_log_density(z, z_second, rho)
                                                          -.5*z*z)/np.sqrt(2*np.pi)),
                                     -12., 12., epsabs=1e-9, points=[rho*z_second])
                self.assertAlmostEqual(result, 1., places=8, msg=f'{rho=}, {v=}, {error=}')

    def test_swapping_marginals_preserves_density(self):
        first = np.array([-3., -1., 0., 1., 3.])
        second = np.array([2., -2., 0., .5, -.5])
        np.testing.assert_allclose(copula_log_density(first, second, .97),
                                   copula_log_density(second, first, .97), rtol=1e-12, atol=1e-12)

    def test_bounded_observation_endpoints_have_finite_log_likelihood(self):
        means = np.array([[.85, .83], [.9, .88], [.97, .96]])
        covariance = np.tile(np.array([[.0025, .002], [.002, .003]]), (3, 1, 1))
        for observed in ([0., 0.], [1., 1.], [.95, .91]):
            result = likelihood(np.array(observed), means, covariance)
            self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
