"""Independent bounded-likelihood arithmetic, without any population asset."""
import unittest

import numpy as np

from analysis.player_rating.uncertainty_likelihood import beta_accuracy_mass, gaussian_accuracy_mass
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass as reference_beta
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass as reference_gaussian


class AccuracyLikelihoodTests(unittest.TestCase):
    def test_likelihoods_preserve_primitive_behavior_at_saturation(self):
        accuracies = np.array([0., .001, 50., 99.999, 100.])[:, None]
        means = np.array([0., 50., 90., 100.])
        variances = np.array([0., 3., 2000., 0.])
        for implementation, reference in ((beta_accuracy_mass, reference_beta),
                                           (gaussian_accuracy_mass, reference_gaussian)):
            with self.subTest(implementation=implementation.__name__):
                np.testing.assert_array_equal(implementation(accuracies, means, variances),
                                              reference(accuracies, means, variances))


if __name__ == '__main__':
    unittest.main()
