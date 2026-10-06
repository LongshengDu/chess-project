"""Reference-free CDF inversion tests."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.test_edge_rating import make_case
from tests.analysis.universal_accuracy_cdf import beta_mid_cdf, invert_cdf, normal_mid_cdf, predict


class AccuracyCdfTests(unittest.TestCase):
    def test_bounded_mid_cdfs_and_normalized_confidence_density(self):
        means = 60.+ARGS.grid*.01
        for cdf in (normal_mid_cdf(85., means, 20.), beta_mid_cdf(85., means, 20.)):
            self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))
            result = invert_cdf(cdf, ARGS.grid)
            self.assertTrue(np.isfinite(result['estimate']))
            self.assertAlmostEqual(float(trapezoid(result['density'], ARGS.grid)), 1., places=12)

    def test_same_linear_gaussian_location_gives_monotone_inversion(self):
        means = 60.+ARGS.grid*.01
        predictions = [invert_cdf(normal_mid_cdf(accuracy, means, 20.), ARGS.grid)['estimate']
                       for accuracy in np.linspace(60., 95., 36)]
        self.assertTrue(np.all(np.diff(predictions) >= 0))

    def test_flat_cdf_is_unidentifiable(self):
        with self.assertRaisesRegex(ValueError, 'identifiable'):
            invert_cdf(np.full_like(ARGS.grid, .5), ARGS.grid)

    def test_predictions_do_not_mutate_or_use_reference_labels(self):
        case = make_case()
        calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]
        original = deepcopy((case, calibration))
        before = predict(case['evidence'], case['fit'], case['ratings'], calibration)
        self.assertEqual((case, calibration), original)
        changed = deepcopy(calibration)
        for row in changed:
            row['fit']['reference'] = 9000
            for player in row['fit']['players'].values():
                player.update(average_accuracy=0., estimate=-1000)
        self.assertEqual(before, predict(case['evidence'], case['fit'], case['ratings'], changed))


if __name__ == '__main__':
    unittest.main()
