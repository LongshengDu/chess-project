"""Mathematical invariants for the two-feature hierarchical affine experiment."""
from copy import deepcopy
import unittest

import numpy as np

from analysis.player_rating.shared_curve_affine import affine_moments, translated_prior
from tests.analysis.curve_hierarchical_candidates import shrinkage
from analysis.player_rating.parameters import RATINGS
from tests.analysis.hierarchical_joint_accuracy import (
    CURVE_ARGS, fit_joint, joint_affine_moments, matrix_shrinkage, positive_semidefinite,
)


def measurement(shift=0.):
    knots = np.linspace(65., 93., len(RATINGS))
    return {'sides': {side: {'joint': {
        'mean': np.column_stack((knots+shift, knots*.9-3.+shift)),
        'covariance': np.tile([[9., 5.], [5., 16.]], (len(RATINGS), 1, 1)),
        'observed': np.array([value, value*.9-3.]),
    }} for side, value in [('White', 88.), ('Black', 78.)]}}


class HierarchicalJointAccuracyTests(unittest.TestCase):
    def test_duplicate_features_recover_scalar_shrinkage_and_affine_decision(self):
        grid = CURVE_ARGS.grid
        local = 70.+.008*grid
        population = 65.+.01*grid
        variance, between = 9., 16.
        expected_curve, _, expected_noise = shrinkage(local, population, between, variance)
        joint = matrix_shrinkage(np.column_stack((local, local)), np.column_stack((population, population)),
                                 np.full((2, 2), between), np.full((2, 2), variance))
        np.testing.assert_allclose(joint['model_curve'], np.column_stack((expected_curve, expected_curve)), atol=1e-12)
        np.testing.assert_allclose(joint['residual_covariance'], expected_noise, atol=1e-12)
        prior = translated_prior(1500.)
        scalar = affine_moments(expected_curve, expected_noise, prior)
        result = joint_affine_moments(joint['model_curve'], joint['residual_covariance'], prior)
        self.assertEqual(result['covariance_rank'], 1)
        for value in (60., 80., 95.):
            actual = prior['prior_mean']+result['affine_coefficients'] @ (np.array([value, value])-result['accuracy_mean'])
            expected = prior['prior_mean']+scalar['affine_slope']*(value-scalar['accuracy_mean'])
            self.assertAlmostEqual(actual, expected, places=9)

    def test_zero_feature_information_returns_prior_level(self):
        prior = translated_prior(1450.)
        result = joint_affine_moments(np.full((len(CURVE_ARGS.grid), 2), 75.), np.zeros((2, 2)), prior)
        np.testing.assert_allclose(result['affine_coefficients'], 0., atol=1e-9)

    def test_feature_permutation_preserves_decision(self):
        grid, prior = CURVE_ARGS.grid, translated_prior(1600.)
        curve = np.column_stack((60.+.012*grid, 50.+.014*grid+.000001*grid**2))
        noise, observed = np.array([[10., 6.], [6., 15.]]), np.array([85., 81.])
        first = joint_affine_moments(curve, noise, prior)
        swapped = joint_affine_moments(curve[:, ::-1], noise[::-1, ::-1], prior)
        one = first['affine_coefficients'] @ (observed-first['accuracy_mean'])
        two = swapped['affine_coefficients'] @ (observed[::-1]-swapped['accuracy_mean'])
        self.assertAlmostEqual(one, two, places=10)

    def test_positive_semidefinite_roundoff_is_not_a_variance_floor(self):
        np.testing.assert_allclose(positive_semidefinite([[1., 0.], [0., -1e-12]]), [[1., 0.], [0., 0.]], atol=1e-15)
        with self.assertRaises(ValueError):
            positive_semidefinite([[1., 2.], [2., 1.]])
        with self.assertRaises(ValueError):
            positive_semidefinite([[1., .5], [.4, 1.]])

    def test_noncommuting_covariances_still_produce_psd_remaining_noise(self):
        local, population = np.ones((3, 2))*80., np.ones((3, 2))*75.
        result = matrix_shrinkage(local, population, [[10., 6.], [6., 8.]], [[8., -2.], [-2., 4.]])
        self.assertGreaterEqual(np.linalg.eigvalsh(result['remaining_covariance']).min(), -1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['residual_covariance']).min(), -1e-12)
        self.assertFalse(np.allclose(result['gain'], result['gain'].T))

    def test_target_exclusion_color_symmetry_and_inputs_unchanged(self):
        local = measurement()
        records = {'target': measurement(5.), 'other1': measurement(-3.), 'other2': measurement(2.)}
        original = deepcopy((local, records))
        actual = {'White': 1500., 'Black': 1600.}
        first = fit_joint(local, actual, records, 'target')
        excluded = fit_joint(local, actual, {key: value for key, value in records.items() if key != 'target'}, 'target')
        self.assertEqual(first['players'], excluded['players'])
        swapped = deepcopy(local)
        swapped['sides'] = {'White': local['sides']['Black'], 'Black': local['sides']['White']}
        second = fit_joint(swapped, {'White': 1600., 'Black': 1500.}, records, 'target')
        self.assertEqual(first['players']['White'], second['players']['Black'])
        self.assertEqual(first['players']['Black'], second['players']['White'])
        for side in ('White', 'Black'):
            np.testing.assert_array_equal(local['sides'][side]['joint']['mean'], original[0]['sides'][side]['joint']['mean'])
        for key in records:
            np.testing.assert_array_equal(records[key]['sides']['White']['joint']['mean'], original[1][key]['sides']['White']['joint']['mean'])

    def test_one_context_has_zero_between_context_covariance(self):
        result = fit_joint(measurement(), {}, {'only': measurement(-2.)}, 'absent')
        np.testing.assert_allclose(result['hierarchy']['between_covariance'], 0.)
        np.testing.assert_allclose(result['hierarchy']['gain'], 0.)

    def test_missing_joint_observation_is_not_fabricated(self):
        local = measurement()
        local['sides']['Black']['joint'] = None
        result = fit_joint(local, {}, {'one': measurement(-2.), 'two': measurement(2.)}, 'none')
        self.assertIsNone(result['players']['Black'])
        self.assertIsNotNone(result['players']['White'])


if __name__ == '__main__':
    unittest.main()
