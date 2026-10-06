"""Matching observed/policy aggregate statistics and numerical integration checks."""
from copy import deepcopy
import itertools
import unittest

import numpy as np

from analysis.lichess_accuracy import INITIAL_CP, game_accuracy, move_metrics, win_percent
from analysis.player_rating.bayesian_shared_curve import Args
from analysis.player_rating.uncertainty_measurement import measure as arithmetic_measure
from tests.analysis import lichess_measurement as experiment
from tests.analysis.shared_curve_lichess import side_moments as historical_moments
from tests.analysis.rating_evidence_fixture import evidence_fixture


LICHESS_METHODS = tuple(method for method in experiment.METHODS if method.startswith('lichess_'))


def discrete_record(values=((0., 100.), (0., 100.)), weights=(1., 1.)):
    return {'observations': [{'qualities': {'position': list(q), 'root': list(q)},
                              'played_index': 0, 'weight': weight,
                              'maia_probabilities': [[1/len(q)]*len(q) for _ in range(21)]}
                             for q, weight in zip(values, weights, strict=True)]}


class LichessMeasurementTests(unittest.TestCase):
    def test_arithmetic_is_the_unchanged_production_measurement(self):
        evidence = evidence_fixture()
        original = arithmetic_measure(evidence)
        actual = experiment.measure(evidence)
        self.assertEqual(actual['curve'], original['curve'])
        for side in experiment.SIDES:
            for key, value in original['sides'][side].items():
                np.testing.assert_equal(actual['sides'][side][key], value)

    def test_the_three_references_are_distinct_and_sampling_matches_exact_enumeration(self):
        record = discrete_record()
        values = np.array(list(itertools.product((0., 100.), repeat=2)))
        scores = experiment.aggregate(values, [1., 1.])
        expected, variance = scores.mean(), scores.var()
        plugin = experiment.side_moments(record, 'lichess_plugin', samples=4096)
        reciprocal = experiment.side_moments(record, 'lichess_reciprocal_plugin', samples=4096)
        sampled = experiment.side_moments(record, 'lichess_policy_expectation', samples=4096)
        self.assertAlmostEqual(expected, 38.120049504950494)
        np.testing.assert_allclose(plugin['mean'], 50., atol=1e-12)
        np.testing.assert_allclose(reciprocal['mean'], 25.99009900990099, atol=1e-12)
        np.testing.assert_allclose(sampled['mean'], expected, atol=1e-12)
        np.testing.assert_allclose(sampled['variance'], variance, atol=1e-10)
        self.assertGreater(variance, 1000.)
        self.assertTrue(np.all(sampled['variance'] > sampled['numerical_mean_se']**2))
        np.testing.assert_allclose(sampled['numerical_mean_se'], 0., atol=1e-12)
        # The denominator belongs to the aggregate distribution, not its sampled mean.
        self.assertNotAlmostEqual(sampled['variance'][0], variance/4096.)

    def test_reciprocal_jensen_bound_and_positive_quality_plugin_upper_bound(self):
        record = discrete_record(((20., 100.), (40., 100.)), (1., 3.))
        results = {name: experiment.side_moments(record, name, samples=4096) for name in LICHESS_METHODS}
        self.assertTrue(np.all(results['lichess_reciprocal_plugin']['mean'] <= results['lichess_policy_expectation']['mean']))
        self.assertTrue(np.all(results['lichess_policy_expectation']['mean'] <= results['lichess_plugin']['mean']))
        for row in results.values():
            self.assertAlmostEqual(row['accuracy'], float(experiment.aggregate([20., 40.], [1., 3.])))

    def test_historical_reciprocal_approximation_preserved_with_full_legal_policies(self):
        record = evidence_fixture()['White']
        expected = historical_moments(record, args=Args(top_probability=1.))
        actual = experiment.side_moments(record, 'lichess_reciprocal_plugin')
        np.testing.assert_allclose(actual['mean'], expected['mean'], atol=1e-12)
        np.testing.assert_allclose(actual['variance'], expected['mean_variance'], atol=1e-12)
        self.assertAlmostEqual(actual['accuracy'], expected['accuracy'])

    def test_observed_matches_local_lichess_game_score_including_forced_positions(self):
        scores = [10., 80., -50., 130.]
        wins = [win_percent(score) for score in [INITIAL_CP, *scores]]
        evidence = {side: {'observations': []} for side in experiment.SIDES}
        for index, row in enumerate(move_metrics(scores)):
            # For four plies Lichess uses two-position windows; pstdev is half their gap.
            weight = max(.5, min(12., abs(wins[index+1]-wins[index])/2))
            q = [row['accuracy']] if index == 0 else [row['accuracy'], 50.]
            evidence[row['side'].title()]['observations'].append(
                {'qualities': {'position': q, 'root': q}, 'played_index': 0, 'weight': weight,
                 'maia_probabilities': [[1/len(q)]*len(q)]*21})
        expected = game_accuracy(scores)
        for method in LICHESS_METHODS:
            result = experiment.measure(evidence, method, samples=4096)
            for side in experiment.SIDES:
                self.assertAlmostEqual(result['sides'][side]['accuracy'], expected[side.lower()], places=12)
                self.assertEqual(result['sides'][side]['moves'], 2)
            self.assertEqual(result['sides']['White']['forced_positions_included'], 1)

    def test_deterministic_policy_has_zero_aggregate_variance_and_exact_floor(self):
        record = discrete_record(((0.,), (100.,)), (1., 2.))
        expected = experiment.aggregate([0., 100.], [1., 2.])
        for method in LICHESS_METHODS:
            result = experiment.side_moments(record, method, samples=4096)
            np.testing.assert_allclose(result['mean'], expected, atol=1e-12)
            np.testing.assert_allclose(result['variance'], 0., atol=1e-10)
            self.assertEqual(result['forced_positions_included'], 2)
        self.assertEqual(experiment.aggregate([0.], [1.]), .5)

    def test_plugin_delta_variance_matches_gradient_of_aggregate(self):
        record = evidence_fixture()['White']
        rows, weights, _ = experiment._rows(record)
        means, _, variances, _, _ = experiment._policy_moments(rows)
        rating = 10
        gradient = []
        for index in range(len(rows)):
            plus, minus = means[:, rating].copy(), means[:, rating].copy()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            gradient.append((experiment.aggregate(plus, weights)-experiment.aggregate(minus, weights))/2e-4)
        expected = np.sum(np.asarray(gradient)**2*variances[:, rating])
        actual = experiment.side_moments(record, 'lichess_plugin')['variance'][rating]
        self.assertAlmostEqual(actual, expected, places=7)

    def test_volatility_weighting_without_harmonic_has_exact_linear_moments(self):
        record = discrete_record(((20., 100.), (40., 100.)), (1., 3.))
        choices = np.asarray(list(itertools.product((20., 100.), (40., 100.))))
        values = choices @ np.asarray([.25, .75])
        actual = experiment.side_moments(record, 'volatility_weighted')
        np.testing.assert_allclose(actual['mean'], values.mean(), atol=1e-12)
        np.testing.assert_allclose(actual['variance'], values.var(), atol=1e-12)
        self.assertEqual(actual['accuracy'], 35.)
        self.assertIsNone(actual['harmonic_mean'])
        forced = experiment.side_moments(discrete_record(((100.,), (0.,)), (1., 3.)), 'volatility_weighted')
        self.assertEqual(forced['accuracy'], 25.)
        self.assertEqual(forced['moves'], 2)
        self.assertEqual(forced['forced_positions_included'], 2)

    def test_joint_covariance_matches_exact_distribution_and_preserves_scalar_measurement(self):
        record = discrete_record()
        choices = np.array(list(itertools.product((0., 100.), repeat=2)))
        paired = np.column_stack((choices.mean(axis=1), experiment.aggregate(choices, [1., 1.])))
        result = experiment.side_moments(record, 'lichess_policy_expectation', samples=4096)
        joint = result['joint']
        np.testing.assert_allclose(joint['mean'], np.tile(paired.mean(axis=0), (21, 1)), atol=1e-12)
        exact_covariance = np.cov(paired, rowvar=False, ddof=0)
        np.testing.assert_allclose(joint['covariance'], np.tile(exact_covariance, (21, 1, 1)), atol=1e-10)
        self.assertGreater(joint['covariance'][0, 0, 1], 0.)
        self.assertTrue(np.all(np.linalg.eigvalsh(joint['covariance']) >= 0.))
        np.testing.assert_allclose(joint['mean'][:, 1], result['mean'], atol=1e-12)
        np.testing.assert_allclose(joint['covariance'][:, 1, 1], result['variance'], atol=1e-10)
        np.testing.assert_allclose(joint['exact_arithmetic_policy_mean'], paired[:, 0].mean(), atol=1e-12)
        np.testing.assert_allclose(joint['exact_arithmetic_policy_variance'], exact_covariance[0, 0], atol=1e-12)
        np.testing.assert_array_equal(joint['observed'], [0., .5])

    def test_joint_arithmetic_excludes_forced_while_lichess_retains_them(self):
        record = discrete_record(((100.,), (20., 100.)), (1., 1.))
        result = experiment.side_moments(record, 'lichess_policy_expectation', samples=4096)
        joint = result['joint']
        self.assertEqual(joint['arithmetic_moves'], 1)
        self.assertEqual(result['moves'], 2)
        self.assertEqual(joint['observed'][0], 20.)
        self.assertEqual(joint['observed'][1], experiment.aggregate([100., 20.], [1., 1.]))
        np.testing.assert_allclose(joint['mean'][:, 0], 60., atol=1e-12)
        np.testing.assert_allclose(joint['covariance'][:, 0, 0], 1600., atol=1e-10)
        forced = experiment.side_moments(discrete_record(((100.,), (100.,))), 'lichess_policy_expectation', samples=4096)
        self.assertIsNone(forced['joint'])

    def test_joint_common_sampling_is_color_symmetric_and_label_free(self):
        evidence = evidence_fixture()
        baseline = experiment.measure(evidence, 'lichess_policy_expectation', samples=4096, seed=17)
        altered = deepcopy(evidence)
        for record in altered.values():
            record['actual_rating'] = -99999.
            record['reference'] = 99999.
            for row in record['observations']:
                row['played_index'] = 2
        changed = experiment.measure({'White': altered['Black'], 'Black': altered['White']},
                                     'lichess_policy_expectation', samples=4096, seed=17)
        for left, right in zip(experiment.SIDES, reversed(experiment.SIDES)):
            for key in ('mean', 'covariance', 'replicate_means', 'numerical_mean_se',
                        'exact_arithmetic_policy_mean', 'exact_arithmetic_policy_variance'):
                np.testing.assert_array_equal(changed['sides'][left]['joint'][key], baseline['sides'][right]['joint'][key])

    def test_determinism_color_candidate_symmetry_and_unused_labels(self):
        evidence = evidence_fixture()
        before = deepcopy(evidence)
        result = experiment.measure(evidence, 'lichess_policy_expectation', samples=4096, seed=123)
        self.assertEqual(evidence, before)
        altered = deepcopy(evidence)
        for record in altered.values():
            record['actual_rating'] = 9999.
            record['commercial_reference'] = -9999.
            for row in record['observations']:
                row['qualities'] = {key: list(reversed(q)) for key, q in row['qualities'].items()}
                row['maia_probabilities'] = [list(reversed(p)) for p in row['maia_probabilities']]
                row['played_index'] = len(row['qualities']['position'])-1-row['played_index']
        changed = experiment.measure(altered, 'lichess_policy_expectation', samples=4096, seed=123)
        self.assertEqual(changed['curve'], result['curve'])
        for side in experiment.SIDES:
            np.testing.assert_array_equal(changed['sides'][side]['mean'], result['sides'][side]['mean'])
            self.assertEqual(changed['sides'][side]['accuracy'], result['sides'][side]['accuracy'])
        swapped = experiment.measure({'White': evidence['Black'], 'Black': evidence['White']},
                                     'lichess_policy_expectation', samples=4096, seed=123)
        self.assertEqual(swapped['curve']['shared_accuracy'], result['curve']['shared_accuracy'])
        for left, right in zip(experiment.SIDES, reversed(experiment.SIDES)):
            np.testing.assert_array_equal(swapped['sides'][left]['mean'], result['sides'][right]['mean'])
        for record in altered.values():
            for row in record['observations']:
                row['played_index'] = 0
        choices = experiment.measure(altered, 'lichess_policy_expectation', samples=4096, seed=123)
        self.assertEqual(choices['curve'], result['curve'])

    def test_empty_one_side_and_bad_inputs(self):
        empty = {side: {'observations': []} for side in experiment.SIDES}
        for method in experiment.METHODS[1:]:
            result = experiment.measure(empty, method, samples=4096)
            self.assertFalse(result['curve']['identifiable'])
            self.assertEqual(result['sides'], dict.fromkeys(experiment.SIDES))
        one = evidence_fixture()
        one['Black']['observations'] = []
        result = experiment.measure(one, 'lichess_plugin')
        self.assertIsNone(result['sides']['Black'])
        self.assertIsNotNone(result['sides']['White'])
        for kwargs in ({'method': 'unknown'}, {'samples': 1000}, {'seed': -1}, {'samples': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                experiment.measure(one, **kwargs)


if __name__ == '__main__':
    unittest.main()
