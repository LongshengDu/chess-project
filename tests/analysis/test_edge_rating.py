"""Reference-independent invariants for the isolated edge-rating experiments."""
from copy import deepcopy
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import summarize
from tests.analysis import edge_global_quality, edge_model_discrepancy, edge_quality_likelihood
from tests.analysis.test_player_rating_bayesian_shared_curve_invariants import make_record


def make_case(qualities=(10., 50., 100.), count=4):
    evidence = {'White': make_record(2, qualities, count),
                'Black': make_record(1, qualities, count+1)}
    return {'evidence': evidence, 'fit': summarize(evidence),
            'ratings': {'White': 1500., 'Black': 1700.}}


class EdgeRatingExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]

    def predictions(self, case=None, calibration=None):
        case = self.case if case is None else case
        calibration = self.calibration if calibration is None else calibration
        output = {}
        for module in (edge_quality_likelihood, edge_model_discrepancy):
            output.update(module.predict(case['evidence'], case['fit'], case['ratings']))
        output.update(edge_global_quality.predict(
            case['evidence'], case['fit'], case['ratings'], calibration))
        return output

    def assert_predictions_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for method in first:
            for side in ('White', 'Black'):
                self.assertAlmostEqual(first[method][side], second[method][side], places=9,
                                       msg=f'{method}/{side}')

    def test_finite_outputs_keep_intersection(self):
        self.assertTrue(edge_quality_likelihood.is_edge(self.case['fit'], 'White'))
        self.assertFalse(edge_quality_likelihood.is_edge(self.case['fit'], 'Black'))
        outputs = self.predictions()
        self.assertIn('global_population_game_quality', outputs)
        self.assertIn('global_population_account_5pct', outputs)
        for values in outputs.values():
            for value in values.values():
                self.assertTrue(np.isfinite(value))
                self.assertGreaterEqual(value, 0.)
                self.assertLessEqual(value, 3200.)
            self.assertEqual(round(values['Black']), self.case['fit']['players']['Black']['estimate'])

    def test_inputs_are_not_mutated(self):
        before = deepcopy((self.case, self.calibration))
        self.predictions()
        self.assertEqual((self.case, self.calibration), before)

    def test_reference_and_game_metadata_are_ignored(self):
        expected = self.predictions()
        changed = deepcopy(self.case)
        changed['fit'].update(reference_elo=9000, game='different_game')
        for side in ('White', 'Black'):
            changed['evidence'][side].update(reference_elo=9999, game='commercial_label')
            changed['fit']['players'][side]['reference'] = -1000
        self.assert_predictions_equal(expected, self.predictions(changed))

    def test_calibration_only_uses_policy_curves_and_variances(self):
        expected = self.predictions()
        changed = deepcopy(self.calibration)
        for case in changed:
            case['ratings'] = {'White': -5000, 'Black': 5000}
            case['evidence'] = {'unexpected': 'not consulted'}
            for side in ('White', 'Black'):
                case['fit']['players'][side].update(average_accuracy=0., estimate=-1000, reference=8000)
        self.assert_predictions_equal(expected, self.predictions(calibration=changed))

    def test_calibration_order_does_not_change_results(self):
        self.assert_predictions_equal(self.predictions(), self.predictions(calibration=self.calibration[::-1]))

    def test_swapping_sides_swaps_predictions(self):
        original = self.predictions()
        changed = deepcopy(self.case)
        for field in ('evidence', 'ratings'):
            changed[field] = {'White': self.case[field]['Black'], 'Black': self.case[field]['White']}
        changed['fit'] = summarize(changed['evidence'])
        swapped = self.predictions(changed)
        self.assert_predictions_equal(original, {
            method: {'White': pair['Black'], 'Black': pair['White']} for method, pair in swapped.items()})

    def test_move_order_does_not_change_results(self):
        original = self.predictions()
        changed = deepcopy(self.case)
        for record in changed['evidence'].values():
            record['observations'].reverse()
        changed['fit'] = summarize(changed['evidence'])
        self.assert_predictions_equal(original, self.predictions(changed))

    def test_legal_move_permutation_does_not_change_results(self):
        original = self.predictions()
        changed = deepcopy(self.case)
        order = [2, 0, 1]
        for record in changed['evidence'].values():
            for row in record['observations']:
                row['played_index'] = order.index(row['played_index'])
                row['qualities'] = {view: [q[i] for i in order] for view, q in row['qualities'].items()}
                row['maia_probabilities'] = [[policy[i] for i in order] for policy in row['maia_probabilities']]
        changed['fit'] = summarize(changed['evidence'])
        self.assert_predictions_equal(original, self.predictions(changed))

    def test_five_percent_account_blends_have_declared_sensitivity(self):
        original = self.predictions()
        for shift in (-200, 200):
            changed = deepcopy(self.case)
            changed['ratings']['White'] += shift
            predictions = self.predictions(changed)
            for name in ('quality_bucket', 'quality_kernel', 'quality_bernoulli'):
                self.assertEqual(predictions[name], original[name])
                self.assertAlmostEqual(predictions[name+'_account']['White']-original[name+'_account']['White'],
                                       shift*.05, places=10)
            self.assertAlmostEqual(
                predictions['global_pooled_account_5pct']['White']-original['global_pooled_account_5pct']['White'],
                shift*.05, places=10)


if __name__ == '__main__':
    unittest.main()
