"""Robust pooling invariants independent of chess-reference labels."""
import unittest

import numpy as np

from tests.analysis.universal_robust_consensus import geometric_median, robust_points


class RobustConsensusTests(unittest.TestCase):
    def test_median_and_trimmed_mean_reject_one_remote_opinion(self):
        points = np.array([[0., 0.], [1., 1.], [2., 2.], [100., 100.]])
        output = robust_points(points)
        np.testing.assert_allclose(output['robust_coordinate_median'], [1.5, 1.5])
        np.testing.assert_allclose(output['robust_trimmed_three_mean'], [1., 1.])
        self.assertTrue(np.all((output['robust_geometric_median'] >= 1)
                               & (output['robust_geometric_median'] <= 2)))

    def test_geometric_median_handles_coincident_opinions(self):
        points = np.array([[0., 0.], [0., 0.], [1., 0.], [-1., 0.]])
        np.testing.assert_allclose(geometric_median(points), [0., 0.], atol=1e-10)

    def test_translation_and_scale_equivariance(self):
        points = np.array([[2000., 1900.], [2050., 1820.], [2130., 2000.], [2300., 1900.]])
        original = robust_points(points)
        transformed = robust_points(.9*points+np.array([155., 170.]))
        for name in original:
            np.testing.assert_allclose(transformed[name], .9*original[name]+[155., 170.], atol=1e-7)

    def test_opinion_order_and_player_swap_have_no_arbitrary_effect(self):
        points = np.array([[2000., 1900.], [2050., 1820.], [2130., 2000.], [2300., 1900.]])
        original = robust_points(points)
        transformed = robust_points(points[::-1, ::-1])
        for name in original:
            np.testing.assert_allclose(transformed[name], original[name][::-1], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
