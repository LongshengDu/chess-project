"""Isolation and measurement invariants for the hurdle-quality experiment."""
from copy import deepcopy
import unittest

import numpy as np

from tests.analysis import universal_hurdle_quality as experiment
from tests.analysis.test_edge_rating import make_case


class HurdleQualityTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]

    def test_frequency_distinguishes_equal_average_quality(self):
        first = {'observations': [{'qualities': {'position': [100., 0.]}, 'played_index': 0},
                                   {'qualities': {'position': [80., 0.]}, 'played_index': 0}]}
        second = {'observations': [{'qualities': {'position': [90., 0.]}, 'played_index': 0}]*2}
        a, _ = experiment._observed(first, 5.)
        b, _ = experiment._observed(second, 5.)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], .5)
        self.assertEqual(b[1], 1.)

    def test_moments_are_symmetric_psd(self):
        mean, covariance = experiment._extend(*experiment._game_moments(self.case['evidence'], 5.))
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.all((0 <= mean) & (mean <= 1)))
        np.testing.assert_allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-14)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))

    def test_calibration_ignores_actuals_references_and_played_indices(self):
        expected = experiment.predict(**self.case, calibration_cases=self.calibration)
        changed = deepcopy(self.calibration)
        for case in changed:
            case['ratings'] = {'White': -5000, 'Black': 9000}
            case['fit'] = {'commercial': 8000}
            for side in ('White', 'Black'):
                for row in case['evidence'][side]['observations']:
                    row['played_index'] = 0
        self.assertEqual(expected, experiment.predict(**self.case, calibration_cases=changed))

    def test_side_swap_finite_immutable_and_account_invariant(self):
        before = deepcopy((self.case, self.calibration))
        expected = experiment.predict(**self.case, calibration_cases=self.calibration)
        changed = deepcopy(self.case)
        changed['evidence'] = {'White': self.case['evidence']['Black'], 'Black': self.case['evidence']['White']}
        changed['ratings'] = {'White': 1900, 'Black': 1300}
        swapped = experiment.predict(**changed, calibration_cases=self.calibration)
        for method, pair in expected.items():
            self.assertTrue(all(np.isfinite(v) and 200 < v < 3000 for v in pair.values()))
            self.assertAlmostEqual(pair['White'], swapped[method]['Black'], places=9)
            self.assertAlmostEqual(pair['Black'], swapped[method]['White'], places=9)
        self.assertEqual((self.case, self.calibration), before)


if __name__ == '__main__':
    unittest.main()
