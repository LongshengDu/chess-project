"""Mathematical domain and interface checks for bounded accuracy experiments."""
from copy import deepcopy
import unittest

import numpy as np

from tests.analysis.edge_bounded_accuracy import (
    ACCURACY_BIN_WIDTH, beta_accuracy_mass, beta_parameters, gamma_accuracy_mass, predict,
)
from tests.analysis.test_edge_rating import make_case


class BoundedAccuracyTests(unittest.TestCase):
    def test_interval_probabilities_normalize_on_bounded_domain(self):
        centers = np.arange(ACCURACY_BIN_WIDTH/2, 100., ACCURACY_BIN_WIDTH)[:, None]
        for mass in (beta_accuracy_mass, gamma_accuracy_mass):
            probabilities = mass(centers, np.array([20., 80., 95.]), np.array([100., 40., 20.]))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(np.all(probabilities >= 0))
            np.testing.assert_allclose(probabilities.sum(axis=0), 1., atol=1e-10, rtol=0)

    def test_beta_variance_cap_produces_positive_finite_shapes(self):
        alpha, beta = beta_parameters(np.array([0., 50., 100.]), np.array([0., 1e9, 0.]))
        self.assertTrue(np.isfinite(alpha).all() and np.isfinite(beta).all())
        self.assertTrue(np.all(alpha > 0) and np.all(beta > 0))

    def test_exact_perfect_and_zero_accuracy_remain_finite(self):
        for mass in (beta_accuracy_mass, gamma_accuracy_mass):
            for observation in (0., 100.):
                probability = mass(observation, np.array([0., 50., 100.]), np.array([0., 10., 0.]))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(np.all((probability >= 0) & (probability <= 1)))

    def test_invalid_moments_and_observations_are_rejected(self):
        for mass in (beta_accuracy_mass, gamma_accuracy_mass):
            for observation, mean, variance in ((-1., 80., 2.), (101., 80., 2.), (90., 101., 2.),
                                                (90., 80., -1.), (float('nan'), 80., 2.)):
                with self.assertRaises(ValueError):
                    mass(observation, mean, variance)

    def test_predictions_are_finite_and_inputs_unchanged(self):
        case = make_case()
        calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]
        original = deepcopy((case, calibration))
        result = predict(case['evidence'], case['fit'], case['ratings'], calibration)
        self.assertEqual(len(result), 4)
        for values in result.values():
            self.assertTrue(all(np.isfinite(value) and 0 <= value <= 3200 for value in values.values()))
            self.assertEqual(values['Black'], case['fit']['players']['Black']['estimate'])
        self.assertEqual((case, calibration), original)

    def test_empty_calibration_is_rejected_and_missing_player_is_preserved(self):
        case = make_case()
        with self.assertRaisesRegex(ValueError, 'calibration'):
            predict(case['evidence'], case['fit'], case['ratings'], [])
        case['fit']['players']['White'].update(estimate=None, average_accuracy=None)
        result = predict(case['evidence'], case['fit'], case['ratings'], [make_case()])
        self.assertTrue(all(values['White'] is None for values in result.values()))

    def test_target_sigma_does_not_affect_population_quality(self):
        case = make_case()
        calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]
        before = predict(case['evidence'], case['fit'], case['ratings'], calibration)
        changed = deepcopy(case['fit'])
        changed['diagnostics']['curve']['likelihood']['accuracy_variance'] *= 100
        changed['diagnostics']['curve']['likelihood']['accuracy_sigma'] *= 10
        self.assertEqual(before, predict(case['evidence'], changed, case['ratings'], calibration))


if __name__ == '__main__':
    unittest.main()
