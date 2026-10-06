"""Reference-independent invariants for shared latent curve uncertainty."""
from copy import deepcopy
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS, curve_posterior
from tests.analysis import universal_curve_uncertainty as experiment
from tests.analysis.test_edge_rating import make_case


class LatentCurveUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]

    def test_equal_curves_equal_variances_recover_single_curve(self):
        grid = ARGS.grid
        curve = 70+25*grid/3200
        observed = np.array([85., 90.])
        estimates, reliability = experiment._shared_reliability(observed, curve, 9., curve, 9., grid)
        expected = [curve_posterior(value, curve, 9.)['unrounded_estimate'] for value in observed]
        np.testing.assert_allclose(estimates, expected, atol=1e-9)
        self.assertAlmostEqual(reliability, .5, places=12)

    def test_side_exchange_preserves_shared_inference(self):
        original = experiment.predict(**self.case, calibration_cases=self.calibration)
        changed = deepcopy(self.case)
        changed['fit']['players'] = {'White': self.case['fit']['players']['Black'],
                                     'Black': self.case['fit']['players']['White']}
        changed['ratings'] = {'White': self.case['ratings']['Black'], 'Black': self.case['ratings']['White']}
        swapped = experiment.predict(**changed, calibration_cases=self.calibration)
        for method in original:
            self.assertAlmostEqual(original[method]['White'], swapped[method]['Black'], places=9)
            self.assertAlmostEqual(original[method]['Black'], swapped[method]['White'], places=9)

    def test_finite_ordered_immutable_and_bounded_account_sensitivity(self):
        before = deepcopy((self.case, self.calibration))
        base = experiment.predict(**self.case, calibration_cases=self.calibration)
        changed = deepcopy(self.case)
        changed['ratings']['White'] += 200
        shifted = experiment.predict(**changed, calibration_cases=self.calibration)
        for method, pair in base.items():
            self.assertTrue(all(np.isfinite(v) and 200 < v < 3000 for v in pair.values()))
            self.assertGreater(pair['White'], pair['Black'])
            expected = 10 if method.endswith('_account') else 0
            self.assertAlmostEqual(shifted[method]['White']-pair['White'], expected, places=9)
            self.assertAlmostEqual(shifted[method]['Black'], pair['Black'], places=9)
        self.assertEqual((self.case, self.calibration), before)

    def test_calibration_labels_actuals_played_means_are_unused(self):
        original = experiment.predict(**self.case, calibration_cases=self.calibration)
        changed = deepcopy(self.calibration)
        for case in changed:
            case['ratings'] = {'White': -1000, 'Black': 10000}
            case['evidence'] = {'not': 'used'}
            for player in case['fit']['players'].values():
                player.update(estimate=8000, average_accuracy=0, commercial_reference=2000)
        self.assertEqual(original, experiment.predict(**self.case, calibration_cases=changed))


if __name__ == '__main__':
    unittest.main()
