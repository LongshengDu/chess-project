"""Checks for the isolated probability-weighted Lichess rating experiment."""
from copy import deepcopy
from itertools import product
import unittest

import numpy as np

from analysis.lichess_accuracy import game_accuracy, move_metrics, win_percent
from analysis.player_rating.parameters import RATINGS
from tests.analysis.shared_curve_lichess import ARGS, side_moments, summarize


def observation(qualities, probabilities, *, played=0, weight=1.):
    q = list(qualities)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = np.tile(p, (len(RATINGS), 1))
    return {'qualities': {'position': q, 'root': q.copy()},
            'maia_probabilities': p.tolist(), 'played_index': played,
            'weight': weight}


def record(rows):
    return {'schema_version': 1, 'conditioning': 'equal_opponent',
            'rating_grid': list(RATINGS), 'observations': rows}


def evidence():
    probabilities = [[.55+.4*t, .45-.4*t] for t in np.linspace(0, 1, len(RATINGS))]
    return {'White': record([observation([100., 40.], probabilities, played=i, weight=w)
                             for i, w in [(0, 1.), (1, 2.), (0, 3.)]]),
            'Black': record([observation([100., 40.], probabilities, played=i, weight=w)
                             for i, w in [(1, 1.), (0, 2.)]])}


class LichessMomentsTests(unittest.TestCase):
    def test_probabilities_weight_arithmetic_and_reciprocal_components(self):
        # Each retained distribution is (.6, .4). The low-probability third
        # move is outside top 99%, but still counts when actually played.
        rows = [observation([100., 50., 0.], [.597, .398, .005], played=2, weight=2.),
                observation([80., 20., 0.], [.597, .398, .005], played=0, weight=1.)]
        result = side_moments(record(rows))
        weighted = (2*80+56)/3
        harmonic = 2/(.6/100+.4/50+.6/80+.4/20)
        np.testing.assert_allclose(result['weighted_mean'], weighted)
        np.testing.assert_allclose(result['harmonic_mean'], harmonic)
        np.testing.assert_allclose(result['mean'], (weighted+harmonic)/2)
        self.assertNotAlmostEqual(harmonic, 2/(1/80+1/56))
        self.assertAlmostEqual(result['accuracy'], ((2*0+80)/3+2/(1+1/80))/2)

    def test_deterministic_policies_reproduce_lichess_game_accuracy(self):
        scores = [20, 40, -500, -600, 1000, -1000]
        wins = [win_percent(score) for score in [15, *scores]]
        records = {'white': [], 'black': []}
        for i, row in enumerate(move_metrics(scores)):
            # With six plies Lichess uses two-point volatility windows.
            weight = max(.5, min(12., abs(wins[i+1]-wins[i])/2))
            records[row['side']].append(observation([row['accuracy'], 10.], [1., 0.], weight=weight))
        expected = game_accuracy(scores)
        for side, rows in records.items():
            result = side_moments(record(rows))
            self.assertAlmostEqual(result['accuracy'], expected[side], places=12)
            np.testing.assert_allclose(result['mean'], expected[side], atol=1e-12)
            np.testing.assert_allclose(result['mean_variance'], 0., atol=1e-10)

    def test_forced_moves_are_included_in_both_components(self):
        rows = [observation([40., 90.], [1., 0.]), observation([100.], [1.])]
        result = side_moments(record(rows))
        self.assertEqual(result['moves'], 2)
        expected = (70+2/(1/40+1/100))/2
        self.assertAlmostEqual(result['accuracy'], expected)
        np.testing.assert_allclose(result['mean'], expected)

    def test_zero_accuracy_floor_applies_only_to_harmonic_part(self):
        result = side_moments(record([observation([0.], [1.])]))
        self.assertEqual(result['accuracy'], .5)
        np.testing.assert_allclose(result['weighted_mean'], 0.)
        np.testing.assert_allclose(result['harmonic_mean'], 1.)
        np.testing.assert_allclose(result['mean'], .5)
        np.testing.assert_allclose(result['mean_variance'], 0.)

    def test_delta_variance_matches_numerically_linearized_joint_distribution(self):
        rows = [observation([20., 90.], [.4, .6], weight=2.),
                observation([40., 100.], [.3, .7], weight=1.)]
        qualities = [np.asarray(row['qualities']['position']) for row in rows]
        policies = [np.asarray(row['maia_probabilities'][0]) for row in rows]
        reciprocal = [1/q for q in qualities]
        means = np.array([p@q for p, q in zip(policies, qualities)] +
                         [p@h for p, h in zip(policies, reciprocal)])

        def aggregate(values):
            return ((2*values[0]+values[1])/3+2/(values[2]+values[3]))/2

        gradient = []
        for i in range(4):
            step = np.zeros(4)
            step[i] = 1e-7
            gradient.append((aggregate(means+step)-aggregate(means-step))/(2e-7))
        gradient = np.asarray(gradient)
        expected = 0.
        for choices in product(range(2), repeat=2):
            chance = np.prod([policies[i][choice] for i, choice in enumerate(choices)])
            point = np.array([qualities[i][choice] for i, choice in enumerate(choices)] +
                             [reciprocal[i][choice] for i, choice in enumerate(choices)])
            expected += chance*float(gradient@(point-means))**2
        actual = side_moments(record(rows))['mean_variance']
        np.testing.assert_allclose(actual, expected, rtol=2e-7)

    def test_repeated_positions_reduce_sampling_variance_without_fixed_sigma(self):
        original = evidence()['White']
        repeated = deepcopy(original)
        repeated['observations'] *= 4
        before, after = side_moments(original), side_moments(repeated)
        np.testing.assert_allclose(after['mean'], before['mean'])
        np.testing.assert_allclose(after['mean_variance'], before['mean_variance']/4)
        self.assertAlmostEqual(after['accuracy'], before['accuracy'])


class LichessFitTests(unittest.TestCase):
    def test_experiment_keeps_explicit_top99_with_current_production_prior(self):
        args = ARGS
        self.assertEqual(args.top_probability, .99)
        self.assertEqual(args.accuracy_sigma_scale, 1.)
        self.assertEqual(args.prior_range, (200., 3000.))
        self.assertEqual(args.central_interval, .20)

    def test_account_reference_metadata_cannot_affect_fit_or_mutate_evidence(self):
        original = evidence()
        saved = deepcopy(original)
        baseline = summarize(original)
        self.assertEqual(original, saved)
        for side in original.values():
            side.update(actual_elo=400, commercial_elo=3000)
            for row in side['observations']:
                row.update(actual_elo=3200, commercial_elo=600)
        self.assertEqual(summarize(original), baseline)

    def test_swapping_colors_preserves_shared_curve_and_swaps_estimates(self):
        original = evidence()
        baseline = summarize(original)
        swapped = summarize({'White': original['Black'], 'Black': original['White']})
        self.assertEqual(baseline['players']['White'], swapped['players']['Black'])
        self.assertEqual(baseline['players']['Black'], swapped['players']['White'])
        for key in ('maia_expected_accuracy', 'shared_mean_variance'):
            self.assertEqual(baseline['diagnostics']['curve'][key], swapped['diagnostics']['curve'][key])

    def test_full_summary_rejects_bad_observation_weights(self):
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                game = evidence()
                game['White']['observations'][0]['weight'] = value
                summarize(game)


if __name__ == '__main__':
    unittest.main()
