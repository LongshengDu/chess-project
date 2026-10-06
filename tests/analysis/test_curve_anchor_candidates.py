"""Properties of the prespecified common-account likelihood candidates."""
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import Args, prior_density
from tests.analysis.curve_anchor_candidates import METHODS, point


class CommonAnchorTests(unittest.TestCase):
    def context(self):
        args = Args()
        grid = args.grid
        return {'grid': grid, 'prior': prior_density(grid, args=args),
                'curve': {'shared_accuracy': 70.+25.*grid/3200.,
                          'likelihood': {'accuracy_variance': 9.}},
                'anchor': 1550., 'slope': 25./3200., 'has_account': True}

    def test_every_fixed_map_is_finite_and_monotone_over_full_accuracy_domain(self):
        context = self.context()
        for method in METHODS:
            values = np.array([point(float(a), context, method) for a in np.linspace(0., 100., 401)])
            self.assertTrue(np.isfinite(values).all(), method)
            self.assertTrue(np.all((values > 200.) & (values < 3000.)), method)
            self.assertGreaterEqual(float(np.diff(values).min()), -1e-9, method)

    def test_missing_account_removes_all_account_model_differences(self):
        context = self.context()
        context['has_account'] = False
        values = [point(88., context, method) for method in METHODS]
        np.testing.assert_allclose(values, values[0], atol=1e-12, rtol=0.)

    def test_zero_slope_means_uninformative_local_noise_prior(self):
        context = self.context()
        context['slope'] = 0.
        local = point(88., context, 'curve_anchor_local_noise_sd')
        context['has_account'] = False
        self.assertAlmostEqual(local, point(88., context, METHODS[0]), places=12)


if __name__ == '__main__':
    unittest.main()
