"""Before-position weighting and constrained-decision invariants."""
from copy import deepcopy
import unittest

import numpy as np

from tests.analysis import universal_competitiveness as experiment
from tests.analysis.test_edge_rating import make_case


def annotated_case():
    case = make_case(count=2)
    for side in ('White', 'Black'):
        for row in case['evidence'][side]['observations']:
            row['position_win_probability'] = .5
    return case


class CompetitivePositionTests(unittest.TestCase):
    def test_annotation_uses_only_before_eval(self):
        case = make_case(count=2)
        before = deepcopy(case['evidence'])
        moves = [{'side': side.lower(), 'position_eval': 0., 'played': {'eval': 100.}}
                 for side in ('White', 'Black') for _ in case['evidence'][side]['observations']]
        adapted = experiment.annotate_probabilities(case['evidence'], moves)
        for side in ('White', 'Black'):
            self.assertTrue(all(row['position_win_probability'] == .5 for row in adapted[side]['observations']))
        for move in moves:
            move['played']['eval'] = -100.
        self.assertEqual(adapted, experiment.annotate_probabilities(case['evidence'], moves))
        self.assertEqual(case['evidence'], before)

    def test_variance_uses_squared_normalized_weights(self):
        case = annotated_case()
        baseline = experiment.weighted_fit(case['evidence'], 1.)
        changed = deepcopy(case['evidence'])
        changed['White']['observations'][1]['position_win_probability'] = .1
        weighted = experiment.weighted_fit(changed, 1.)
        base_variance = baseline['diagnostics']['curve']['shared_mean_variance']
        weighted_variance = weighted['diagnostics']['curve']['shared_mean_variance']
        expected_ratio = ((1+.36**2)/(1+.36)**2+1/3)/(1/2+1/3)
        self.assertAlmostEqual(weighted_variance/base_variance, expected_ratio, places=10)

    def test_equal_competitiveness_recovers_uniform_aggregation(self):
        case = annotated_case()
        fit = experiment.weighted_fit(case['evidence'], .5)
        for side in ('White', 'Black'):
            self.assertAlmostEqual(fit['players'][side]['average_accuracy'],
                                   case['fit']['players'][side]['average_accuracy'], places=12)
        np.testing.assert_allclose(fit['diagnostics']['curve']['shared_accuracy'],
                                   case['fit']['diagnostics']['curve']['shared_accuracy'], atol=1e-10)

    def test_projection_is_minimum_distance_and_idempotent(self):
        fit = {'players': {'White': {'average_accuracy': 90}, 'Black': {'average_accuracy': 80}}}
        pair = {'White': 1800., 'Black': 2000.}
        projected = experiment.project_order(pair, fit)
        self.assertEqual(projected, {'White': 1900.5, 'Black': 1899.5})
        self.assertEqual(experiment.project_order(projected, fit), projected)
        self.assertEqual(pair, {'White': 1800., 'Black': 2000.})

    def test_account_sensitivity_and_inputs_unchanged(self):
        case = annotated_case()
        calibration = [annotated_case(), annotated_case()]
        before = deepcopy((case, calibration))
        baseline = experiment.predict(**case, calibration_cases=calibration)
        changed = deepcopy(case)
        changed['ratings']['White'] += 200
        shifted = experiment.predict(**changed, calibration_cases=calibration)
        for method in baseline:
            self.assertLessEqual(abs(shifted[method]['White']-baseline[method]['White']), 10+1e-9)
            self.assertLessEqual(abs(shifted[method]['Black']-baseline[method]['Black']), 5+1e-9)
        self.assertEqual((case, calibration), before)


if __name__ == '__main__':
    unittest.main()
