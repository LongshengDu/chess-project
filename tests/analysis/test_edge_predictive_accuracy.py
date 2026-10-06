"""Distribution and isolation invariants for the quantized predictive experiment."""
from copy import deepcopy
import unittest

import numpy as np

from analysis.player_rating.bayesian_shared_curve import GRID, summarize
from tests.analysis import edge_predictive_accuracy as experiment
from tests.analysis.test_player_rating_bayesian_shared_curve_invariants import make_record


def make_case(qualities=(0., 60., 100.), count=4):
    evidence = {'White': make_record(2, qualities, count),
                'Black': make_record(1, qualities, count+1)}
    return {'evidence': evidence, 'fit': summarize(evidence),
            'ratings': {'White': 1500., 'Black': 1700.}}


class PredictiveAccuracyTests(unittest.TestCase):
    def test_fft_matches_exact_two_position_distribution(self):
        probabilities = np.tile([.25, .75], (len(GRID), 1))
        positions = [(np.array([0, 100]), probabilities)]*2
        result = experiment._side_distribution(positions)
        expected = np.zeros(201)
        expected[[0, 100, 200]] = [.25**2, 2*.25*.75, .75**2]
        np.testing.assert_allclose(result['pmf'], np.tile(expected, (len(GRID), 1)), atol=1e-13)
        np.testing.assert_allclose(result['pmf'].sum(axis=1), 1., atol=1e-13)

    def test_perfect_mass_is_not_smoothed_away(self):
        probabilities = np.tile([.1, .9], (len(GRID), 1))
        result = experiment._side_distribution([(np.array([0, 100]), probabilities)]*4)
        np.testing.assert_allclose(experiment._bin_probabilities(result, 100.), .9**4, atol=1e-13)

    def test_quality_rounding_bound_and_forced_exclusion(self):
        record = make_record(0, (10.49, 50.51, 99.99), 2)
        record['observations'].append({'qualities': {'position': [100.]}})
        positions = experiment._quantized_positions(record)
        self.assertEqual(len(positions), 2)
        np.testing.assert_array_equal(positions[0][0], [10, 51, 100])

    def test_cache_ignores_labels_but_changes_with_quality_or_policy(self):
        evidence = make_case()['evidence']
        original = experiment._game_distributions(evidence)
        changed = deepcopy(evidence)
        for side in changed.values():
            side.update(actual_elo=5000, commercial_reference=-1000)
            for observation in side['observations']:
                observation['played_index'] = 0
        self.assertIs(original, experiment._game_distributions(changed))
        changed['White']['observations'][0]['qualities']['position'][0] = 20.
        self.assertIsNot(original, experiment._game_distributions(changed))
        changed = deepcopy(evidence)
        changed['White']['observations'][0]['maia_probabilities'][0] = [.1, .2, .7]
        self.assertIsNot(original, experiment._game_distributions(changed))

    def test_preserves_intersection_and_has_fixed_account_sensitivity(self):
        case = make_case()
        calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]
        before = deepcopy((case, calibration))
        base = experiment.predict(**case, calibration_cases=calibration)
        shifted = experiment.predict(**{**case, 'ratings': {'White': 1700., 'Black': 1900.}},
                                     calibration_cases=calibration)
        for values in base.values():
            self.assertEqual(values['Black'], case['fit']['players']['Black']['estimate'])
            self.assertGreater(values['White'], 600)
            self.assertLess(values['White'], 2600)
        self.assertEqual(base['predictive_accuracy_fft']['White'], shifted['predictive_accuracy_fft']['White'])
        self.assertAlmostEqual(shifted['predictive_accuracy_fft_account_5pct']['White']
                               - base['predictive_accuracy_fft_account_5pct']['White'], 10., places=9)
        self.assertEqual((case, calibration), before)

    def test_mixture_is_equal_game_and_side_not_length_weighted(self):
        case = make_case()
        calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]
        first = experiment.predict(**case, calibration_cases=calibration, edge_only=False)
        second = experiment.predict(**case, calibration_cases=calibration[::-1], edge_only=False)
        self.assertEqual(first, second)
        doubled = experiment.predict(**case, calibration_cases=calibration*2, edge_only=False)
        for method in first:
            for side in ('White', 'Black'):
                self.assertAlmostEqual(first[method][side], doubled[method][side], places=9)


if __name__ == '__main__':
    unittest.main()
