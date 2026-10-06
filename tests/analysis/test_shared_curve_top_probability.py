"""Label-independent checks for shared top-probability selection and compatibility."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.parameters import RATINGS
from tests.analysis.shared_curve_top_probability import (
    Args, TopProbabilityRating, prior_density, prior_weights, top_probability_policy,
)


def record(played=(0, 0, 0, 1)):
    """An improving Maia curve with real choices, including singleton nuclei."""
    low, high = np.array([.52, .30, .18]), np.array([.83, .12, .05])
    policy = [(1-t)*low+t*high for t in np.linspace(0., 1., len(RATINGS))]
    return {
        'schema_version': 1, 'conditioning': 'equal_opponent',
        'rating_grid': list(RATINGS),
        'observations': [
            {'qualities': {'position': [100., 60., 0.], 'root': [100., 60., 0.]},
             'maia_probabilities': np.asarray(policy).tolist(),
             'played_index': index, 'weight': 1.}
            for index in played
        ],
    }


def forced_row():
    return {'qualities': {'position': [100.], 'root': [100.]},
            'maia_probabilities': [[1.]]*len(RATINGS), 'played_index': 0,
            'weight': 1.}


def evidence():
    return {'White': record(), 'Black': record((0, 0, 1))}


class TopProbabilityPolicyTests(unittest.TestCase):
    def test_smallest_prefix_above_threshold_is_renormalized(self):
        policy = np.array([[.1, .4, .2, .3], [.05, .8, .1, .05]])
        original = policy.copy()
        selected = top_probability_policy(policy, .68)
        np.testing.assert_allclose(selected, [[0., 4/7, 0., 3/7], [0., 1., 0., 0.]])
        np.testing.assert_array_equal(policy, original)
        np.testing.assert_allclose(selected.sum(axis=1), 1.)

    def test_threshold_is_strict_and_equal_probability_boundary_moves_are_kept(self):
        policy = np.array([[.5, .25, .125, .125]])
        np.testing.assert_allclose(top_probability_policy(policy, .5), [[2/3, 1/3, 0., 0.]])
        np.testing.assert_allclose(top_probability_policy(policy, .75), policy)
        tied = np.array([[.4, .2, .2, .1, .1]])
        np.testing.assert_allclose(top_probability_policy(tied, .5), [[.5, .25, .25, 0., 0.]])

    def test_move_order_does_not_break_cutoff_ties(self):
        policy = np.array([[.4, .2, .2, .1, .1], [.1, .1, .2, .3, .3]])
        permutation = [3, 1, 4, 0, 2]
        expected = top_probability_policy(policy, .68)[:, permutation]
        np.testing.assert_allclose(top_probability_policy(policy[:, permutation], .68), expected)

    def test_full_probability_recovers_original_including_zero_probability_moves(self):
        policy = np.array([[.5, .3, .2, 0.], [0., 0., 1., 0.]])
        np.testing.assert_allclose(top_probability_policy(policy, 1.), policy)

    def test_invalid_probability_and_policy_fail_instead_of_silently_changing_data(self):
        for probability in (0., -.1, 1.01, float('nan'), float('inf'), True):
            with self.subTest(probability=probability), self.assertRaises((ValueError, TypeError)):
                top_probability_policy(np.array([[.8, .2]]), probability)
        for policy in (np.array([.8, .2]), np.array([[.8, .3]]),
                       np.array([[1.1, -.1]]), np.array([[float('nan'), .2]])):
            with self.subTest(policy=policy), self.assertRaises((ValueError, TypeError)):
                top_probability_policy(policy, .68)


class TopProbabilityFitTests(unittest.TestCase):
    def test_actual_played_move_is_counted_even_when_outside_every_nucleus(self):
        game = {'White': record((2, 2, 2)), 'Black': record()}
        fitted = TopProbabilityRating().fit(game)
        self.assertEqual(fitted['players']['White']['average_accuracy'], 0.)
        self.assertEqual(fitted['players']['White']['moves_used'], 3)
        self.assertGreater(min(fitted['diagnostics']['curve']['maia_expected_accuracy']), 0.)

    def test_one_legal_move_is_excluded_from_both_observed_and_expected_accuracy(self):
        game = evidence()
        expected = TopProbabilityRating().fit(game)
        game['White']['observations'].extend([forced_row() for _ in range(9)])
        game['Black']['observations'].insert(0, forced_row())
        actual = TopProbabilityRating().fit(game)
        for side in ('White', 'Black'):
            self.assertEqual(actual['players'][side], expected['players'][side])
        for key in ('maia_expected_accuracy', 'monotone_expected_accuracy',
                    'shared_mean_variance', 'posterior_densities'):
            self.assertEqual(actual['diagnostics']['curve'][key], expected['diagnostics']['curve'][key])

    def test_singleton_nucleus_is_still_a_choice_when_other_legal_moves_exist(self):
        game = evidence()
        # High-rating policies select only the 100-accuracy move, but these
        # positions are not legally forced and must remain in the game mean.
        weights = top_probability_policy(
            np.asarray(game['White']['observations'][0]['maia_probabilities']), .68)
        self.assertEqual(np.count_nonzero(weights[-1]), 1)
        fit = TopProbabilityRating().fit(game)
        self.assertEqual(fit['players']['White']['moves_used'], 4)
        self.assertEqual(fit['players']['White']['average_accuracy'], 90.)

    def test_all_forced_side_is_unobserved_and_other_side_can_still_be_fitted(self):
        game = evidence()
        game['Black']['observations'] = [forced_row() for _ in range(4)]
        fit = TopProbabilityRating().fit(game)
        self.assertIsNotNone(fit['players']['White']['estimate'])
        self.assertIsNone(fit['players']['Black']['estimate'])
        self.assertEqual(fit['players']['Black']['moves_used'], 0)
        self.assertIsNone(fit['players']['Black']['average_accuracy'])
        game['White']['observations'] = [forced_row()]
        fit = TopProbabilityRating().fit(game)
        self.assertTrue(all(player['estimate'] is None for player in fit['players'].values()))

    def test_repeating_observations_reduces_variance_instead_of_using_fixed_sigma(self):
        game = evidence()
        baseline = TopProbabilityRating().fit(game)['diagnostics']['curve']
        for side in game:
            game[side]['observations'] *= 4
        repeated = TopProbabilityRating().fit(game)['diagnostics']['curve']
        self.assertGreater(baseline['shared_mean_variance'], 0.)
        self.assertAlmostEqual(repeated['shared_mean_variance'], baseline['shared_mean_variance']/4)
        self.assertAlmostEqual(repeated['likelihood']['accuracy_sigma'],
                               baseline['likelihood']['accuracy_sigma']/2)
        self.assertEqual(repeated['likelihood']['sigma_scale'], .5)
        self.assertAlmostEqual(baseline['likelihood']['accuracy_sigma'],
                               np.sqrt(baseline['shared_mean_variance'])*.5)

    def test_account_and_commercial_metadata_do_not_enter_inference_or_change_evidence(self):
        game = evidence()
        original = deepcopy(game)
        expected = TopProbabilityRating().fit(game)
        self.assertEqual(game, original)
        for side in game:
            game[side].update(actual_elo=100, opponent_elo=3200, reference_elo=700)
            for row in game[side]['observations']:
                row.update(actual_elo=3000, reference_elo=1000)
        self.assertEqual(TopProbabilityRating().fit(game), expected)

    def test_color_swap_preserves_curve_and_swaps_player_results(self):
        game = evidence()
        normal = TopProbabilityRating().fit(game)
        swapped = TopProbabilityRating().fit({'White': game['Black'], 'Black': game['White']})
        self.assertEqual(normal['players']['White'], swapped['players']['Black'])
        self.assertEqual(normal['players']['Black'], swapped['players']['White'])
        for key in ('maia_expected_accuracy', 'shared_mean_variance'):
            self.assertEqual(normal['diagnostics']['curve'][key], swapped['diagnostics']['curve'][key])
        self.assertEqual(normal['diagnostics']['curve']['posterior_densities']['White'],
                         swapped['diagnostics']['curve']['posterior_densities']['Black'])


class TopProbabilityPriorTests(unittest.TestCase):
    def test_experiment_uses_current_prior_with_its_original_threshold_and_sigma(self):
        args = Args()
        self.assertEqual(args.top_probability, .68)
        self.assertEqual(args.accuracy_sigma_scale, .5)
        self.assertEqual(args.prior_range, (200., 3000.))
        self.assertEqual(args.flat_prior_range, (800., 2400.))
        grid = args.grid
        self.assertAlmostEqual(trapezoid(prior_density(grid, args=args), grid), 1., places=8)
        points = np.array([-1., 0., 200., 400., 600., 1600., 2600., 2800., 3000., 3200., 3201.])
        density = prior_density(points, args=args)
        np.testing.assert_allclose(density[[4, 6]]/density[5], 16/17, atol=1e-14)
        np.testing.assert_allclose(density[[3, 7]]/density[5], 1/17, atol=1e-14)
        np.testing.assert_allclose(prior_weights(points, args=args), density/density[5], atol=1e-14)
        np.testing.assert_array_equal(density[[0, 1, 2, 8, 9, 10]], 0.)


if __name__ == '__main__':
    unittest.main()
