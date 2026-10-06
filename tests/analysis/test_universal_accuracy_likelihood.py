"""Joint-context likelihood normalization, symmetry and calibration invariants."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.test_edge_rating import make_case
from tests.analysis.universal_accuracy_likelihood import context_models, joint_posterior, predict


class UniversalAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]

    def test_joint_posterior_is_normalized_and_symmetric(self):
        grid = ARGS.grid
        means = np.stack((60.+grid*.01, 65.+grid*.01))
        result = joint_posterior([80., 90.], means, 20., [.5, .5], grid)
        reverse = joint_posterior([90., 80.], means, 20., [.5, .5], grid)
        np.testing.assert_allclose(trapezoid(result['density'], grid, axis=-1), 1., atol=1e-12)
        np.testing.assert_allclose(result['estimates'], reverse['estimates'][::-1])
        np.testing.assert_allclose(result['posterior_context'], reverse['posterior_context'])

    def test_random_context_curves_are_bounded_and_nondecreasing(self):
        for means, variances, weights in context_models(self.case['fit'], self.calibration).values():
            self.assertTrue(np.isfinite(means).all())
            self.assertTrue(np.all((means >= 0) & (means <= 100)))
            self.assertTrue(np.all(np.diff(means, axis=-1) >= -1e-10))
            self.assertAlmostEqual(float(np.sum(weights)), 1.)

    def test_no_mutation_and_fixed_account_sensitivity(self):
        original = deepcopy((self.case, self.calibration))
        before = predict(self.case['evidence'], self.case['fit'], self.case['ratings'], self.calibration)
        changed_ratings = dict(self.case['ratings'], White=self.case['ratings']['White']+200)
        after = predict(self.case['evidence'], self.case['fit'], changed_ratings, self.calibration)
        self.assertEqual((self.case, self.calibration), original)
        for name in before:
            difference = after[name]['White']-before[name]['White']
            self.assertAlmostEqual(difference, 10. if '_account_5pct_' in name else 0.)

    def test_other_games_played_outcomes_and_ratings_are_ignored(self):
        before = predict(self.case['evidence'], self.case['fit'], self.case['ratings'], self.calibration)
        changed = deepcopy(self.calibration)
        for case in changed:
            case['ratings'] = {'White': -99999, 'Black': 99999}
            for player in case['fit']['players'].values():
                player.update(average_accuracy=0., estimate=-9999, reference=9999)
        self.assertEqual(before, predict(self.case['evidence'], self.case['fit'], self.case['ratings'], changed))


if __name__ == '__main__':
    unittest.main()
