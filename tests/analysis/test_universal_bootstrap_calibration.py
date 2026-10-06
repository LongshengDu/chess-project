"""Synthetic-distribution and bias-correction invariants, with no reference labels."""
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.universal_bootstrap_calibration import correct_estimate, measurement_cells, response_curve


class BootstrapCalibrationTests(unittest.TestCase):
    def test_measurement_cells_form_complete_probability_distributions(self):
        grid = np.linspace(0., 100., 401)
        cells = measurement_cells(grid, [0., 50., 100.], 20., [10., 60., 95.], [20., 30., 15.])
        self.assertTrue(np.isfinite(cells).all())
        self.assertTrue(np.all(cells >= 0))
        np.testing.assert_allclose(cells.sum(axis=0), 1., atol=1e-12)

    def test_inverse_and_onestep_have_the_standard_affine_form(self):
        grid = np.arange(0., 3201., 5.)
        response = 800.+.5*grid
        inverse, onestep = correct_estimate(1800., grid, response)
        self.assertEqual(inverse, 2000.)
        self.assertEqual(onestep, 1900.)

    def test_inverse_plateau_uses_middle_and_extremes_are_bounded(self):
        grid = np.array([0., 1000., 2000., 3000.])
        response = np.array([1000., 1500., 1500., 2000.])
        self.assertEqual(correct_estimate(1500., grid, response)[0], 1500.)
        self.assertEqual(correct_estimate(500., grid, response)[0], 0.)
        self.assertEqual(correct_estimate(2500., grid, response)[0], 3000.)

    def test_estimator_response_is_finite_nondecreasing_and_cached(self):
        target = 60.+ARGS.grid*.01
        population = 65.+ARGS.grid*.01
        variance = np.full_like(ARGS.grid, 40.)
        result = response_curve(target, 10., population, variance, ARGS.grid)
        self.assertTrue(np.isfinite(result['response']).all())
        self.assertTrue(np.all(np.diff(result['response']) >= 0))
        self.assertIs(result, response_curve(target, 10., population, variance, ARGS.grid))


if __name__ == '__main__':
    unittest.main()
