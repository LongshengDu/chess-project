"""Proper Student-t measurement and controlled Gaussian-noise comparisons."""
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass, posterior
from tests.analysis.universal_variance_uncertainty import MODELS, posterior_mean, student_accuracy_mass


class VarianceUncertaintyTests(unittest.TestCase):
    def test_student_measurement_normalizes_over_complete_accuracy_bins(self):
        centers = np.arange(.005, 100., .01)[:, None]
        mass = student_accuracy_mass(centers, np.array([0., 50., 100.]), np.array([20., 30., 40.]))
        self.assertTrue(np.isfinite(mass).all())
        self.assertTrue(np.all(mass >= 0))
        np.testing.assert_allclose(mass.sum(axis=0), 1., atol=1e-10, rtol=0)

    def test_student_has_heavier_variance_matched_remote_tails(self):
        self.assertGreater(float(student_accuracy_mass(80., 50., 25.)),
                           float(gaussian_accuracy_mass(80., 50., 25.)))

    def test_gaussian_control_reproduces_original_mixture_mean(self):
        grid = ARGS.grid
        target, population = 60.+grid*.01, 65.+grid*.01
        between, conditional = np.full_like(grid, 20.), 10.
        original = posterior(85., target, 15., population, between+conditional, grid)['mean']
        current = posterior_mean(85., target, 15., population, conditional, between, MODELS[0], grid)
        self.assertAlmostEqual(float(current), float(original), places=10)

    def test_extra_prediction_noise_is_exactly_one_conditional_variance(self):
        grid = ARGS.grid
        target, population = 60.+grid*.01, 65.+grid*.01
        between = np.full_like(grid, 20.)
        expected = posterior(85., target, 30., population, between+20., grid)['mean']
        current = posterior_mean(85., target, 15., population, 10., between, MODELS[2], grid)
        self.assertAlmostEqual(float(current), float(expected), places=10)

    def test_endpoint_bins_are_finite(self):
        for accuracy in (0., 100.):
            self.assertTrue(np.isfinite(student_accuracy_mass(accuracy, [0., 50., 100.], [0., 5., 0.])).all())


if __name__ == '__main__':
    unittest.main()
