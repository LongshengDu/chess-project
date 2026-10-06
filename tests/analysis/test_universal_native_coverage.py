"""Continuous reliability, shape and sensitivity checks for native coverage."""
from copy import deepcopy
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.test_edge_rating import make_case
from tests.analysis.universal_native_coverage import accuracy_mapping, native_reliability, predict


class NativeCoverageTests(unittest.TestCase):
    def test_reliability_is_bounded_symmetric_and_continuous(self):
        values = native_reliability(np.array([69., 75., 80., 85., 91.]), 70., 90., 4.)
        self.assertTrue(np.all((values >= 0) & (values <= 1)))
        np.testing.assert_allclose(values, values[::-1])
        adjacent = native_reliability(np.array([90.-1e-7, 90.+1e-7]), 70., 90., 4.)
        self.assertLess(abs(adjacent[1]-adjacent[0]), 1e-6)

    def test_projected_maps_are_finite_and_nondecreasing_on_whole_domain(self):
        mapping = accuracy_mapping(60.+ARGS.grid*.01, 10., 65.+ARGS.grid*.01,
                                   np.full_like(ARGS.grid, 40.), [66., 86.], ARGS.grid)
        self.assertEqual(mapping['accuracy_grid'][0], 0.)
        self.assertEqual(mapping['accuracy_grid'][-1], 100.)
        for point in ('mean', 'median'):
            self.assertTrue(np.isfinite(mapping[point]).all())
            self.assertTrue(np.all(np.diff(mapping[point]) >= 0))

    def test_no_mutation_and_fixed_account_effect(self):
        case = make_case()
        calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]
        original = deepcopy((case, calibration))
        before = predict(case['evidence'], case['fit'], case['ratings'], calibration)
        ratings = dict(case['ratings'], White=case['ratings']['White']+200)
        after = predict(case['evidence'], case['fit'], ratings, calibration)
        self.assertEqual((case, calibration), original)
        for name in before:
            self.assertAlmostEqual(after[name]['White']-before[name]['White'], 10. if '_account_5pct_' in name else 0.)


if __name__ == '__main__':
    unittest.main()
