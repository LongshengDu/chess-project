"""Cohort and scoring checks for the isolated arithmetic shared-curve sweep."""
from copy import deepcopy
import math
import unittest

from tests.analysis.experiment_shared_curve_sweep import build_variants, eligibility, rank_variants


class SweepGridTests(unittest.TestCase):
    def test_custom_sigma_grid_preserves_every_requested_value(self):
        scales = (.6, .65, .7, .75, .8, .85, .9, .95, 1.)
        variants = build_variants((.99, 1.), scales)
        self.assertEqual(len(variants), 18)
        self.assertEqual({(args.top_probability, args.accuracy_sigma_scale) for args in variants.values()},
                         {(probability, scale) for probability in (.99, 1.) for scale in scales})
        self.assertEqual(variants['top-099-sigma-0.65'].accuracy_sigma_scale, .65)
        self.assertTrue(all(args.prior_range == (200., 3000.) for args in variants.values()))
        self.assertEqual(len(build_variants()), 12)

    def test_invalid_empty_and_duplicate_grids_are_rejected(self):
        for probabilities, scales in (((), (1.,)), ((.99,), ()), ((.99, .99), (1.,)),
                                      ((.99,), (.7, .7)), ((0.,), (1.,)), ((1.1,), (1.,)),
                                      ((.99,), (0.,)), ((.99,), (float('inf'),))):
            with self.subTest(probabilities=probabilities, scales=scales), self.assertRaises(ValueError):
                build_variants(probabilities, scales)


def fit(white=85., black=90.):
    return {
        'players': {
            'White': {'average_accuracy': white, 'estimate': 1500},
            'Black': {'average_accuracy': black, 'estimate': 1800},
        },
        'diagnostics': {
            'curve': {'ratings': [600, 1600, 2600],
                      'monotone_expected_accuracy': [80., 88., 95.]},
        },
    }


def row(variant, game, side, reference, estimate):
    return {'variant': variant, 'game': game, 'side': side,
            'reference': reference, 'estimate': estimate,
            'top_probability': .99, 'sigma_scale': .5 if variant == 'a' else 1.}


def pair(variant, game, white, black):
    return [row(variant, game, 'White', *white),
            row(variant, game, 'Black', *black)]


class IntersectionEligibilityTests(unittest.TestCase):
    def test_both_sides_must_intersect_the_supported_curve(self):
        result = eligibility(fit())
        self.assertTrue(result['included'])
        for side in ('White', 'Black'):
            self.assertEqual(result['sides'][side]['position'], 'inside')
            self.assertEqual(result['sides'][side]['curve_low'], 80.)
            self.assertEqual(result['sides'][side]['curve_high'], 95.)
        self.assertEqual(result['sides']['White']['average_accuracy'], 85.)

    def test_a_single_nonintersecting_side_excludes_the_whole_game(self):
        for white, black, side, position in [
            (79., 90., 'White', 'below'),
            (85., 96., 'Black', 'above'),
        ]:
            with self.subTest(side=side):
                result = eligibility(fit(white, black))
                self.assertFalse(result['included'])
                self.assertEqual(result['sides'][side]['position'], position)

    def test_endpoints_and_floating_point_tolerance_are_inclusive(self):
        for white, black in [(80., 95.), (80.-5e-11, 95.+5e-11)]:
            with self.subTest(white=white, black=black):
                self.assertTrue(eligibility(fit(white, black))['included'])
        self.assertFalse(eligibility(fit(80.-1e-8, 90.))['included'])
        self.assertFalse(eligibility(fit(85., 95.+1e-8))['included'])

    def test_missing_estimate_is_unavailable_even_with_intersecting_accuracy(self):
        for side in ('White', 'Black'):
            value = fit()
            value['players'][side]['estimate'] = None
            result = eligibility(value)
            self.assertFalse(result['included'])
            self.assertEqual(result['sides'][side]['position'], 'unavailable')

    def test_prior_tail_and_posterior_width_cannot_make_game_eligible(self):
        original = fit(96., 90.)
        changed = deepcopy(original)
        changed['parameters'] = {'accuracy_sigma_scale': .5}
        changed['diagnostics']['extended_curve'] = {
            'ratings': [0, 3200], 'accuracy': [0., 100.]}
        changed['diagnostics']['prior_density'] = [1., 0.]
        for player in changed['players'].values():
            player.update(interval=[0, 3200], uncertainty=1600)
        self.assertEqual(eligibility(original), eligibility(changed))

    def test_commercial_references_and_supplied_ratings_cannot_affect_filter(self):
        original = fit()
        changed = deepcopy(original)
        changed.update(WhiteEloEstimate=50, BlackEloEstimate=5000)
        for side in changed['players'].values():
            side.update(commercial_elo=0, actual_elo=5000)
        self.assertEqual(eligibility(original), eligibility(changed))


class SweepRankingTests(unittest.TestCase):
    def test_primary_ranking_uses_one_common_game_cohort(self):
        rows = (
            pair('a', 'game2', (1000, 1010), (2000, 2010))
            + pair('a', 'game10', (1000, 1000), (2000, 2000))
            + pair('b', 'game2', (1000, 1020), (2000, 2020))
            + pair('b', 'game3', (1000, 3000), (2000, 4000))
        )
        result = rank_variants(rows, {'a': {'game2', 'game10'},
                                     'b': {'game2', 'game3'}})
        self.assertEqual(result['common_games'], ['game2'])
        self.assertEqual([x['variant'] for x in result['common_ranking']], ['a', 'b'])
        common = {x['variant']: x for x in result['common_ranking']}
        self.assertEqual(common['a']['mean_absolute_error'], 10.)
        self.assertEqual(common['b']['mean_absolute_error'], 20.)
        self.assertEqual(common['a']['games'], 1)
        self.assertEqual(common['a']['players'], 2)
        individual = {x['variant']: x for x in result['individual_ranking']}
        self.assertEqual(individual['a']['mean_absolute_error'], 5.)
        self.assertEqual(individual['b']['mean_absolute_error'], 1010.)

    def test_common_games_are_sorted_by_numeric_game_number(self):
        rows = [item for game in ('game10', 'game2', 'game1')
                for item in pair('a', game, (1000, 1000), (2000, 2000))]
        result = rank_variants(rows, {'a': {'game10', 'game1', 'game2'}})
        self.assertEqual(result['common_games'], ['game1', 'game2', 'game10'])

    def test_scores_use_rounded_estimates_and_include_both_players(self):
        rows = pair('a', 'game1', (1000, 1000.4), (2000, 1998.6))
        result = rank_variants(rows, {'a': {'game1'}})['common_ranking'][0]
        self.assertEqual(result['rank'], 1)
        self.assertEqual(result['top_probability'], .99)
        self.assertEqual(result['sigma_scale'], .5)
        self.assertEqual(result['players'], 2)
        self.assertEqual(result['mean_absolute_error'], .5)
        self.assertAlmostEqual(result['root_mean_square_error'], math.sqrt(.5))
        self.assertEqual(result['maximum_absolute_error'], 1)
        self.assertEqual(result['ordering_matches'], 1)

    def test_ordering_reports_a_reversal_and_does_not_count_it_as_matching(self):
        rows = pair('a', 'game1', (2000, 1800), (1800, 2000))
        result = rank_variants(rows, {'a': {'game1'}})['common_ranking'][0]
        self.assertEqual(result['ordering_matches'], 0)

    def test_ties_are_resolved_by_rmse_then_maximum_then_variant(self):
        # Equal MAE: constant errors beat more variable errors by RMSE.
        rows = pair('a', 'game1', (1000, 1000), (2000, 2004))
        rows += pair('c', 'game1', (1000, 1002), (2000, 2002))
        rows += pair('b', 'game1', (1000, 1002), (2000, 2002))
        result = rank_variants(rows, {name: {'game1'} for name in ('a', 'b', 'c')})
        self.assertEqual([x['variant'] for x in result['common_ranking']], ['b', 'c', 'a'])
        self.assertEqual([x['rank'] for x in result['common_ranking']], [1, 2, 3])
        # Equal MAE and RMSE: maximum error breaks the tie.
        rows = pair('a', 'game1', (1000, 1000), (2000, 2003))
        rows += pair('a', 'game2', (1000, 1003), (2000, 2000))
        rows += pair('b', 'game1', (1000, 1001), (2000, 2001))
        rows += pair('b', 'game2', (1000, 1004), (2000, 2000))
        result = rank_variants(rows, {name: {'game1', 'game2'} for name in ('a', 'b')})
        self.assertEqual([x['variant'] for x in result['common_ranking']], ['a', 'b'])

    def test_empty_common_cohort_is_reported_without_invented_scores(self):
        rows = pair('a', 'game1', (1000, 1001), (2000, 2001))
        rows += pair('b', 'game2', (1000, 1002), (2000, 2002))
        result = rank_variants(rows, {'a': {'game1'}, 'b': {'game2'}})
        self.assertEqual(result['common_games'], [])
        for stats in result['common_ranking']:
            self.assertEqual(stats['games'], 0)
            self.assertEqual(stats['players'], 0)
            self.assertIsNone(stats['mean_absolute_error'])
            self.assertIsNone(stats['root_mean_square_error'])
            self.assertIsNone(stats['maximum_absolute_error'])


if __name__ == '__main__':
    unittest.main()
