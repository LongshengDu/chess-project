"""Label-free empirical-prior probability and numerical-update checks."""
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_weights
from tests.analysis.universal_empirical_prior import _component_logs, _m_step, fit_population


class EmpiricalPriorTests(unittest.TestCase):
    def test_learned_priors_normalize_and_keep_fixed_taper_support(self):
        grid = ARGS.grid
        likelihoods = np.exp(-.5*((grid[None, :]-np.array([1000., 1200., 1800., 2200., 2400.])[:, None])/150.)**2)
        for components in (1, 2):
            result = fit_population(likelihoods, components)
            self.assertTrue(np.isfinite(result['prior_mass']).all())
            self.assertAlmostEqual(float(result['prior_mass'].sum()), 1., places=12)
            self.assertTrue(np.all(result['prior_mass'][prior_weights(grid) == 0] == 0))
            self.assertTrue(np.all(np.diff(result['component_means']) >= 0))
            self.assertTrue(np.all(np.diff(result['objective_trace']) >= -1e-5))

    def test_tapered_component_update_improves_expected_log_probability(self):
        grid = ARGS.grid
        weights = prior_weights(grid)
        supported = weights > 0
        x, logs = grid[supported]/1000., np.log(weights[supported])
        counts = np.exp(-.5*((x-2.2)/.3)**2)
        initial = np.array([1.6, np.log(.7)])
        updated = _m_step(counts, initial, x, logs)
        self.assertGreaterEqual(float(counts@_component_logs(updated, x, logs)),
                                float(counts@_component_logs(initial, x, logs))-1e-8)

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_population(np.ones((2, len(ARGS.grid))), 3)
        with self.assertRaises(ValueError):
            fit_population(np.ones((0, len(ARGS.grid))), 1)


if __name__ == '__main__':
    unittest.main()
