"""Reference-independent invariants for the production shared-game curve."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from analysis.player_rating.bayesian_shared_curve import (
    ARGS, Args, GRID, SharedCurve, prior_density, prior_weights, curve_posterior, fit_pair, side_moments,
)
from scipy.integrate import cumulative_trapezoid, trapezoid

FINE = ARGS.grid


def make_record(played=1, qualities=(10., 50., 100.), count=4):
    low = np.array([.6, .3, .1])
    high = np.array([.05, .2, .75])
    policy = np.array([(1-t)*low+t*high for t in np.linspace(0., 1., 21)])
    row = {'played_index': played, 'qualities': {'root': list(qualities), 'position': list(qualities)},
           'maia_probabilities': policy.tolist()}
    return {'observations': [deepcopy(row) for _ in range(count)]}


class SharedCurveTests(unittest.TestCase):
    def test_reference_and_account_metadata_cannot_change_inference(self):
        white, black = make_record(2), make_record(1)
        expected = fit_pair(white, black)
        for change in (-200, 0, 200):
            changed_white, changed_black = deepcopy(white), deepcopy(black)
            changed_white.update(actual_elo=1600+change, opponent_elo=2100-change,
                                 reference_elo=600, game='unrelated', side='Black')
            changed_black.update(actual_elo=2100-change, opponent_elo=1600+change,
                                 reference_elo=2600, game='different', side='White')
            self.assertEqual(fit_pair(changed_white, changed_black), expected)

    def test_swapping_players_only_swaps_estimates(self):
        white, black = make_record(2), make_record(1, count=3)
        original, swapped = fit_pair(white, black), fit_pair(black, white)
        self.assertEqual(original['players'], list(reversed(swapped['players'])))
        self.assertEqual(original['maia_expected_accuracy'], swapped['maia_expected_accuracy'])
        self.assertEqual(original['shared_mean_variance'], swapped['shared_mean_variance'])
        self.assertEqual(original['white_minus_black'], -swapped['white_minus_black'])

    def test_improving_played_choice_preserves_curve_and_improves_estimate(self):
        white, black = make_record(0), make_record(1)
        original = fit_pair(white, black)
        white['observations'][0]['played_index'] = 2
        improved = fit_pair(white, black)
        self.assertEqual(original['monotone_expected_accuracy'], improved['monotone_expected_accuracy'])
        self.assertEqual(original['shared_mean_variance'], improved['shared_mean_variance'])
        self.assertGreater(improved['players'][0]['unrounded_estimate'],
                           original['players'][0]['unrounded_estimate'])
        self.assertEqual(original['players'][1], improved['players'][1])

    def test_posterior_is_monotone_for_fixed_curve_and_variance(self):
        curve = 10. + .025*FINE
        values = [curve_posterior(accuracy, curve, 50.)['unrounded_estimate']
                  for accuracy in np.linspace(0., 100., 41)]
        self.assertEqual(values, sorted(values))

    def test_flat_curve_reports_no_estimate_and_full_supported_range(self):
        record = make_record(qualities=(100., 100., 100.))
        result = fit_pair(record, record)
        self.assertFalse(result['identifiable'])
        self.assertIsNone(result['white_minus_black'])
        for player in result['players']:
            self.assertIsNone(player['estimate'])
            np.testing.assert_allclose(player['conditional_interval'], [0., 3200.])

    def test_boundaries_and_intervals_remain_on_supported_grid(self):
        curve = 10. + .025*FINE
        for accuracy in (0., 50., 100.):
            result = curve_posterior(accuracy, curve, 50.)
            lo, hi = result['conditional_interval']
            self.assertTrue(0 <= lo <= result['unrounded_estimate'] <= hi <= 3200)
            self.assertTrue(0 <= result['estimate'] <= 3200)
            density = np.asarray(result['posterior_density'])
            self.assertTrue(np.isfinite(density).all())
            self.assertEqual(density[0], 0.)
            self.assertEqual(density[-1], 0.)
            self.assertAlmostEqual(trapezoid(density, FINE), 1., places=12)
        above = curve_posterior(95., curve, 50.)
        higher = curve_posterior(100., curve, 50.)
        self.assertGreater(higher['unrounded_estimate'], above['unrounded_estimate'])
        self.assertNotIn('likelihood_anchor', above)

    def test_other_games_cannot_affect_pair_result(self):
        first = (make_record(2), make_record(1))
        before = fit_pair(*first)
        fit_pair(make_record(0, count=20), make_record(2, count=1))
        self.assertEqual(fit_pair(*first), before)

    def test_input_records_are_not_modified(self):
        records = [make_record(2), make_record(1)]
        original = deepcopy(records)
        fit_pair(*records)
        self.assertEqual(records, original)

    def test_rounding_tie_preserves_distinct_unrounded_estimates(self):
        curve = 10. + .025*FINE
        white = curve_posterior(50.000001, curve, 50.)
        black = curve_posterior(50., curve, 50.)
        self.assertGreater(white['unrounded_estimate'], black['unrounded_estimate'])
        self.assertEqual(white['estimate'], black['estimate'])

    def test_equal_sides_receive_equal_estimates(self):
        record = make_record()
        result = fit_pair(record, deepcopy(record))
        self.assertEqual(result['players'][0], result['players'][1])
        self.assertEqual(result['white_minus_black'], 0)

    def test_empty_observations_are_unobserved_and_unnormalized_policies_are_rejected(self):
        self.assertIsNone(side_moments({'observations': []}))
        record = make_record()
        record['observations'][0]['maia_probabilities'][0][0] += .1
        with self.assertRaises(ValueError):
            side_moments(record)

    def test_explicit_nucleus_conditions_both_moments_but_keeps_actual_played_choice(self):
        row = {'played_index': 2, 'qualities': {'position': [100., 50., 0.]},
               'maia_probabilities': [[.597, .398, .005]]*len(GRID)}
        forced = {'played_index': 0, 'qualities': {'position': [100.]},
                  'maia_probabilities': [[1.]]*len(GRID)}
        result = side_moments({'observations': [row, forced]}, args=Args(top_probability=.99))
        self.assertEqual(result['moves'], 1)
        self.assertEqual(result['accuracy'], 0.)
        np.testing.assert_allclose(result['mean'], 80.)
        np.testing.assert_allclose(result['mean_variance'], 600.)
        selection = result['selection']
        self.assertEqual(selection['forced_positions_removed'], 1)
        np.testing.assert_allclose(selection['mean_retained_mass'], .995)
        np.testing.assert_allclose(selection['played_move_retained_fraction'], 0.)

    def test_forced_rows_are_validated_before_omission(self):
        forced = {'played_index': 0, 'qualities': {'position': [100.]},
                  'maia_probabilities': [[.5]]*len(GRID)}
        with self.assertRaisesRegex(ValueError, 'normalized'):
            side_moments({'observations': [forced]})

    def test_full_probability_uses_all_moves_but_still_excludes_forced_positions(self):
        record = make_record()
        row = record['observations'][0]
        quality = np.asarray(row['qualities']['position'])
        policy = np.asarray(row['maia_probabilities'])
        mean = policy @ quality
        expected = {'mean': mean,
                    'mean_variance': (policy @ (quality**2)-mean**2)/4,
                    'accuracy': 50., 'moves': 4}
        record['observations'].append({'played_index': 0, 'qualities': {'position': [100.]},
                                       'maia_probabilities': [[1.]]*len(GRID)})
        result = side_moments(record)
        self.assertEqual(ARGS.top_probability, 1.)
        for key in ('mean', 'mean_variance', 'accuracy', 'moves'):
            np.testing.assert_allclose(result[key], expected[key], atol=1e-12)

    def test_prior_arguments_change_only_posterior_and_prior_diagnostics(self):
        records = [make_record(2), make_record(1)]
        original = fit_pair(*records)
        changed = fit_pair(*records, args=Args(flat_prior_range=(1000., 2200.)))
        for key in ('rating_grid', 'maia_expected_accuracy', 'monotone_expected_accuracy',
                    'shared_mean_variance', 'fine_ratings', 'shared_accuracy',
                    'likelihood', 'top_probability', 'selection'):
            with self.subTest(key=key):
                self.assertEqual(changed[key], original[key])
        for before, after in zip(original['players'], changed['players'], strict=True):
            self.assertEqual(before['average_accuracy'], after['average_accuracy'])
            self.assertEqual(before['moves'], after['moves'])
        self.assertNotEqual(changed['posterior_densities'], original['posterior_densities'])

    def test_played_index_must_be_an_actual_legal_move_index(self):
        for index in (-1, True, 3, 1.0):
            record = make_record()
            record['observations'][0]['played_index'] = index
            with self.assertRaisesRegex(ValueError, 'Played index'):
                side_moments(record)

    def test_tails_preserve_measured_values_and_endpoint_derivatives(self):
        measured = np.linspace(30., 90., len(GRID))
        curve = SharedCurve(measured)
        np.testing.assert_allclose(curve(GRID), measured, atol=1e-12)
        values = curve(np.arange(-10000., 10001.))
        self.assertTrue(np.all((values >= 0) & (values <= 100)))
        self.assertTrue(np.all(np.diff(values) >= -1e-12))
        for boundary, slope in zip((600., 2600.), curve.slopes):
            for direction in (-1, 1):
                difference = (curve(boundary+direction*.001)-curve(boundary))/(direction*.001)
                self.assertAlmostEqual(float(difference), slope, places=7)
        self.assertEqual(float(SharedCurve(np.zeros(len(GRID)))(-500)), 0.)
        self.assertEqual(float(SharedCurve(np.full(len(GRID), 100.))(4000)), 100.)

    def test_prior_raw_weights_match_requested_anchors_and_normalized_density(self):
        points = np.array([-1, 0, 200, 300, 400, 500, 600, 700, 800, 1600,
                           2400, 2500, 2600, 2700, 2800, 2900, 3000, 3200, 3201])
        weights = prior_weights(points)
        np.testing.assert_allclose(weights[2:9], [0, 1/626, 1/17, .5, 16/17, 625/626, 1],
                                   atol=1e-14, rtol=0)
        np.testing.assert_allclose(weights[10:17], weights[2:9][::-1], atol=1e-14, rtol=0)
        np.testing.assert_array_equal(weights[[8, 9, 10]], 1.)
        np.testing.assert_array_equal(weights[[0, 1, 2, 16, 17, 18]], 0.)
        density = prior_density(points)
        np.testing.assert_allclose(density/density[9], weights, atol=1e-14)
        np.testing.assert_allclose(density*2200, weights, atol=1e-14)
        self.assertAlmostEqual(trapezoid(prior_density(FINE), FINE), 1., places=12)

    def test_prior_symmetry_monotonic_shoulders_and_total_area(self):
        ratings = np.linspace(0., 3200., 32001)
        weights = prior_weights(ratings)
        self.assertTrue(np.all(np.diff(weights[ratings <= 800.]) >= 0))
        self.assertTrue(np.all(np.diff(weights[ratings >= 2400.]) <= 0))
        np.testing.assert_allclose(weights, weights[::-1], atol=1e-14, rtol=0)
        self.assertAlmostEqual(trapezoid(weights, ratings), 2200., places=8)

    def test_prior_support_and_power_midpoints_with_asymmetric_shoulders(self):
        args = Args(rating_range=(-100., 3600.), flat_prior_range=(600., 2600.),
                    prior_range=(0., 3400.))
        points = np.array([-100, 0, 300, 600, 2600, 3000, 3400, 3600])
        np.testing.assert_allclose(prior_weights(points, args=args), [0, 0, .5, 1, 1, .5, 0, 0], atol=1e-14)
        density = prior_density(args.grid, args=args)
        self.assertAlmostEqual(trapezoid(density, args.grid), 1., places=10)
        weights = prior_weights(args.grid, args=args)
        self.assertTrue(np.all(np.diff(weights[args.grid <= 600]) >= 0))
        self.assertTrue(np.all(np.diff(weights[args.grid >= 2600]) <= 0))

    def test_power_prior_is_smooth_at_support_plateau_and_shoulder_midpoints(self):
        delta = .001
        for point, expected in ((200, 0.), (500, .5), (800, 1.),
                                (2400, 1.), (2700, .5), (3000, 0.)):
            values = prior_weights(np.array([point-delta, point, point+delta]))
            self.assertAlmostEqual(values[1], expected, places=14)
            left, right = np.diff(values)/delta
            self.assertAlmostEqual(left, right, places=7)
            if point in (200, 800, 2400, 3000):
                self.assertAlmostEqual(left, 0., places=7)
                self.assertAlmostEqual(right, 0., places=7)

    def test_posterior_matches_direct_gaussian_integration_and_prior_rescaling(self):
        curve = SharedCurve(np.linspace(65., 95., len(GRID)))(FINE)
        for accuracy in (62., 81., 98.):
            with self.subTest(accuracy=accuracy):
                actual = curve_posterior(accuracy, curve, 9.)
                density = prior_weights(FINE)*np.exp(-.5*(accuracy-curve)**2/9.)
                density /= trapezoid(density, FINE)
                cdf = cumulative_trapezoid(density, FINE, initial=0.)
                quantiles = np.interp([.4, .5, .6], cdf, FINE)
                np.testing.assert_allclose(actual['posterior_density'], density, atol=1e-14)
                np.testing.assert_allclose(actual['conditional_interval'], quantiles[[0, 2]], atol=1e-10)
                self.assertAlmostEqual(actual['unrounded_estimate'], quantiles[1], places=10)
                self.assertEqual(actual['estimate'], round(quantiles[1]))
                np.testing.assert_array_equal(np.asarray(actual['posterior_density'])[prior_weights(FINE) == 0], 0.)
                with patch('analysis.player_rating.bayesian_shared_curve.prior_density',
                           return_value=prior_weights(FINE)):
                    scaled = curve_posterior(accuracy, curve, 9.)
                np.testing.assert_allclose(scaled['posterior_density'], density, atol=1e-14)

    def test_posterior_grid_refinement_preserves_estimate(self):
        class FineArgs(Args):
            @property
            def grid(self):
                return np.arange(self.rating_range[0], self.rating_range[1]+1)

        fine_args = FineArgs()
        shared = SharedCurve(np.linspace(65., 95., len(GRID)))
        for accuracy in (62., 81., 98.):
            with self.subTest(accuracy=accuracy):
                coarse = curve_posterior(accuracy, shared(FINE), 9.)
                fine = curve_posterior(accuracy, shared(fine_args.grid), 9., args=fine_args)
                self.assertLess(abs(coarse['unrounded_estimate']-fine['unrounded_estimate']), .1)

    def test_default_interval_is_quantiles_40_and_60_percent(self):
        result = curve_posterior(65., 10.+.025*FINE, 50.)
        cdf = cumulative_trapezoid(result['posterior_density'], FINE, initial=0)
        self.assertAlmostEqual(cdf[-1], 1., places=12)
        np.testing.assert_allclose(np.interp(result['conditional_interval'], FINE, cdf), [.4, .6], atol=1e-12)

    def test_method_argument_bounds_and_probability_are_validated(self):
        for kwargs in ({'rating_range': (600, 2600)}, {'flat_prior_range': (2600, 600)},
                       {'rating_range': (.5, 3200.5)},
                       {'rating_range': (0, float('inf'))}, {'rating_range': (False, 3200)},
                       {'central_interval': True}, {'central_interval': 0},
                       {'central_interval': 1}, {'central_interval': float('nan')},
                       {'accuracy_sigma_scale': 0}, {'accuracy_sigma_scale': -1},
                       {'accuracy_sigma_scale': True}, {'accuracy_sigma_scale': float('inf')},
                       {'accuracy_sigma_scale': float('nan')},
                       {'top_probability': 0}, {'top_probability': 1.01},
                       {'top_probability': True}, {'top_probability': float('nan')},
                       {'prior_range': (800, 3000)}, {'prior_range': (200, 2400)},
                       {'prior_range': (3000, 200)}, {'prior_range': (False, 3000)},
                       {'prior_range': (-1, 3000)}, {'prior_range': (200, 3201)},
                       {'prior_range': (200, float('inf'))},
                       {'prior_range': (float('nan'), 3000)},
                       {'rating_range': (400, 3200)}, {'rating_range': (0, 2800)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Args(**kwargs)

    def test_extrapolated_estimate_can_exceed_last_maia_rating(self):
        result = curve_posterior(100., SharedCurve(np.linspace(40., 90., len(GRID)))(FINE), 5.)
        self.assertGreater(result['estimate'], 2600)
        self.assertLess(result['estimate'], 3200)

    def test_sigma_multiplier_only_changes_gaussian_variance(self):
        curve = 10.+.025*FINE
        for scale in (.25, .5, 1.):
            args = Args(accuracy_sigma_scale=scale)
            actual = curve_posterior(65., curve, 50., args=args)
            expected = curve_posterior(65., curve, 50.*scale**2,
                                       args=Args(accuracy_sigma_scale=1.))
            self.assertEqual(actual, expected)
        self.assertEqual(ARGS.accuracy_sigma_scale, 1.)

    def test_plateau_retains_all_supported_intersections_and_broadens_posterior(self):
        plateau_curve = np.interp(FINE, [0., 1200., 2000., 3200.], [10., 50., 50., 90.])
        args = Args(central_interval=.68)
        plateau = curve_posterior(50., plateau_curve, 50., args=args)
        unique = curve_posterior(50., 10.+.025*FINE, 50., args=args)
        self.assertAlmostEqual(plateau['unrounded_estimate'], 1600., places=8)
        density = np.asarray(plateau['posterior_density'])
        flat = density[(FINE >= 1200.) & (FINE <= 2000.)]
        np.testing.assert_allclose(flat, flat[0], atol=1e-12)
        self.assertGreater(np.ptp(plateau['conditional_interval']),
                           np.ptp(unique['conditional_interval']))

    def test_curve_slope_still_controls_rating_resolution(self):
        shallow = curve_posterior(70., 70.+.001*(FINE-1600.), 50.)
        steep = curve_posterior(70., 70.+.01*(FINE-1600.), 50.)
        self.assertLess(np.ptp(steep['conditional_interval']), np.ptp(shallow['conditional_interval']))

    def test_more_moves_still_reduce_variance_and_narrow_posterior(self):
        white, black = make_record(1), make_record(1)
        original = fit_pair(white, black)
        for record in (white, black):
            record['observations'] *= 4
        repeated = fit_pair(white, black)
        self.assertAlmostEqual(repeated['shared_mean_variance'],
                               original['shared_mean_variance']/4, places=10)
        self.assertAlmostEqual(original['likelihood']['accuracy_variance'],
                               original['shared_mean_variance'])
        for first, second in zip(original['players'], repeated['players'], strict=True):
            self.assertLess(np.ptp(second['conditional_interval']), np.ptp(first['conditional_interval']))


if __name__ == '__main__':
    unittest.main()
