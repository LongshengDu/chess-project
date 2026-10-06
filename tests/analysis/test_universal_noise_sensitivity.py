"""Checks isolating conditional noise from context variance and reference labels."""
from copy import deepcopy
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior
from tests.analysis.test_edge_rating import make_case
from tests.analysis.universal_noise_sensitivity import NoiseArgs, predict, scaled_variances


class NoiseSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]

    def test_squared_sigma_multiplier_leaves_between_context_variance_unchanged(self):
        target, population = scaled_variances(16., [4., 12.], [10., 20.], NoiseArgs(.5))
        self.assertEqual(target, 4.)
        np.testing.assert_array_equal(population, [12., 22.])

    def test_scale_one_reproduces_the_original_proper_mixture_mean(self):
        case = self.case
        results = predict(case['evidence'], case['fit'], case['ratings'], self.calibration)
        _, variances, pooled, between = _population(self.calibration, ARGS.grid)
        diagnostic = case['fit']['diagnostics']['curve']
        for side in ('White', 'Black'):
            original = posterior(case['fit']['players'][side]['average_accuracy'],
                                 np.asarray(diagnostic['shared_accuracy']), diagnostic['likelihood']['accuracy_variance'],
                                 pooled, between+variances.mean(), ARGS.grid)
            self.assertEqual(results['mixture_noise_100_mean_all'][side], float(original['mean']))

    def test_no_mutation_and_fixed_account_effect(self):
        original = deepcopy((self.case, self.calibration))
        before = predict(self.case['evidence'], self.case['fit'], self.case['ratings'], self.calibration)
        ratings = dict(self.case['ratings'], White=self.case['ratings']['White']+200)
        after = predict(self.case['evidence'], self.case['fit'], ratings, self.calibration)
        self.assertEqual((self.case, self.calibration), original)
        for name in before:
            self.assertAlmostEqual(after[name]['White']-before[name]['White'], 10. if '_account_5pct_' in name else 0.)

    def test_invalid_sigma_scale_rejected(self):
        for scale in (0., -1., np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                NoiseArgs(scale)


if __name__ == '__main__':
    unittest.main()
