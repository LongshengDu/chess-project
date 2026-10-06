"""Probability and interface checks for the continuous predictive model mixture."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_bounded_accuracy import ACCURACY_BIN_WIDTH
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass, posterior, predict
from tests.analysis.test_edge_rating import make_case


class PredictiveMixtureTests(unittest.TestCase):
    def test_truncated_gaussian_measurements_normalize(self):
        centers = np.arange(ACCURACY_BIN_WIDTH/2, 100., ACCURACY_BIN_WIDTH)[:, None]
        masses = gaussian_accuracy_mass(centers, np.array([0., 50., 100.]), np.array([10., 40., 100.]))
        np.testing.assert_allclose(masses.sum(axis=0), 1., atol=1e-10, rtol=0)
        self.assertTrue(np.isfinite(masses).all())

    def test_posterior_normalization_and_finite_endpoints(self):
        grid = ARGS.grid
        target = 60.+grid*.01
        population = 65.+grid*.01
        result = posterior(np.array([0., 50., 100.])[:, None], target, 10., population, 40., grid)
        np.testing.assert_allclose(trapezoid(result['density'], grid, axis=-1), 1., atol=1e-12)
        self.assertTrue(np.isfinite(result['mean']).all())
        self.assertTrue(np.isfinite(result['median']).all())
        self.assertTrue(np.all((result['target_model_probability'] >= 0) & (result['target_model_probability'] <= 1)))

    def test_model_weight_is_updated_by_integrated_evidence(self):
        grid = ARGS.grid
        result = posterior(90., np.full_like(grid, 90.), 1., np.full_like(grid, 40.), 1., grid)
        self.assertGreater(result['target_model_probability'], .999)

    def test_inputs_unchanged_and_account_blend_has_fixed_sensitivity(self):
        case = make_case()
        calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]
        original = deepcopy((case, calibration))
        before = predict(case['evidence'], case['fit'], case['ratings'], calibration)
        self.assertEqual((case, calibration), original)
        ratings = dict(case['ratings'], White=case['ratings']['White']+200)
        after = predict(case['evidence'], case['fit'], ratings, calibration)
        self.assertEqual(before['predictive_mixture_all'], after['predictive_mixture_all'])
        self.assertAlmostEqual(after['predictive_mixture_account_5pct_all']['White']-
                               before['predictive_mixture_account_5pct_all']['White'], 10.)

    def test_empty_calibration_rejected_and_missing_player_retained(self):
        case = make_case()
        with self.assertRaisesRegex(ValueError, 'calibration'):
            predict(case['evidence'], case['fit'], case['ratings'], [])
        case['fit']['players']['White'].update(estimate=None, average_accuracy=None)
        results = predict(case['evidence'], case['fit'], case['ratings'], [make_case()])
        self.assertTrue(all(values['White'] is None for values in results.values()))


if __name__ == '__main__':
    unittest.main()
