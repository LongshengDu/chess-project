"""Analytic prior-integrated competitiveness weights and fit invariants."""
from copy import deepcopy
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from tests.analysis import universal_competitive_marginal as experiment
from tests.analysis.test_universal_competitiveness import annotated_case


class MarginalCompetitivenessTests(unittest.TestCase):
    def test_analytic_weight_matches_uniform_prior_quadrature(self):
        probability = np.array([.001, .05, .2, .45, .5, .8, .99])
        nodes, weights = leggauss(64)
        base = 4*probability*(1-probability)
        numerical = (base[:, None]**((nodes+1)/2))@(weights/2)
        np.testing.assert_allclose(experiment.marginal_weight(probability), numerical, atol=1e-12)

    def test_endpoints_symmetry_and_between_geometric_arithmetic_means(self):
        probability = np.linspace(0, 1, 101)
        base = 4*probability*(1-probability)
        weights = experiment.marginal_weight(probability)
        self.assertEqual(weights[0], 0)
        self.assertEqual(weights[-1], 0)
        self.assertEqual(weights[50], 1)
        np.testing.assert_allclose(weights, weights[::-1], atol=1e-12)
        self.assertTrue(np.all(weights+1e-12 >= np.sqrt(base)))
        self.assertTrue(np.all(weights <= (1+base)/2+1e-12))

    def test_equal_position_chances_recover_unweighted_curve_without_mutation(self):
        case = annotated_case()
        before = deepcopy(case)
        fitted = experiment.marginal_fit(case['evidence'])
        np.testing.assert_allclose(fitted['diagnostics']['curve']['shared_accuracy'],
                                   case['fit']['diagnostics']['curve']['shared_accuracy'], atol=1e-12)
        self.assertEqual(case, before)


if __name__ == '__main__':
    unittest.main()
