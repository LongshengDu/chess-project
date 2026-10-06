"""Pre-score moment identities, retained local context and common-map behavior."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON
from unittest.mock import patch

import numpy as np
from scipy.integrate import trapezoid

from tests.analysis import curve_hierarchical_candidates as candidate
from tests.analysis.rating_evidence_fixture import evidence_fixture


def fixture(accuracies=(58., 62.), noise=0.):
    grid = np.linspace(0., 3200., 641)
    curve = 40.+.01*grid
    return {'identifiable': True, 'grid': grid, 'local_curve': curve, 'population_curve': curve,
            'model_curve': curve, 'measurement_variance': noise, 'between_context_variance': 25.,
            'residual_variance': noise, 'local_weight': .5,
            'observed': dict(zip(('White', 'Black'), accuracies)), 'contexts_used': 16,
            'contexts_excluded': 0, 'calibration_hash': 'synthetic'}


class HierarchicalCandidateTests(unittest.TestCase):
    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_prepare_retains_target_curve_and_does_not_use_competitive_population(self):
        evidence = evidence_fixture()
        before = deepcopy(evidence)
        prepared = candidate.prepare(evidence)
        self.assertGreater(prepared['local_weight'], 0.)
        np.testing.assert_allclose(prepared['model_curve'], prepared['local_weight']*prepared['local_curve']+
                                   (1-prepared['local_weight'])*prepared['population_curve'])
        measurement = candidate.measure(evidence)
        measurement['curve']['shared_accuracy'] = (np.asarray(measurement['curve']['shared_accuracy'])-1.).tolist()
        with patch.object(candidate, 'measure', return_value=measurement):
            altered = candidate.prepare(evidence)
        np.testing.assert_allclose(altered['model_curve']-prepared['model_curve'], -prepared['local_weight'])
        corpus = candidate.load_calibration()
        stripped = replace(corpus, records=tuple(replace(record, competitive=None) for record in corpus.records))
        accounts = {'White': 1400., 'Black': 1600.}
        with patch.object(candidate, 'load_calibration', return_value=stripped):
            self.assertEqual(candidate.predict(evidence, accounts), candidate.predict_prepared(prepared, accounts))
        self.assertEqual(evidence, before)

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_prepare_matches_the_predeclared_discrepancy_shrinkage_exactly(self):
        from tests.analysis.curve_discrepancy_candidates import prepare as discrepancy_prepare
        evidence = evidence_fixture()
        prepared, discrepancy = candidate.prepare(evidence), discrepancy_prepare(evidence)
        model = discrepancy['models']['discrepancy_curve_shrinkage']
        self.assertEqual(prepared['between_context_variance'], discrepancy['between_context_variance'])
        self.assertEqual(prepared['local_weight'], discrepancy['local_weight'])
        self.assertEqual(prepared['residual_variance'], model['variance'])
        np.testing.assert_array_equal(prepared['model_curve'], model['curve'])

    def test_shrinkage_matches_declared_empirical_bayes_moments(self):
        local, population = np.array([60., 90.]), np.array([50., 80.])
        curve, weight, noise = candidate.shrinkage(local, population, 12., 4.)
        self.assertEqual(weight, .75)
        self.assertEqual(noise, 7.)
        np.testing.assert_array_equal(curve, .75*local+.25*population)
        np.testing.assert_array_equal(candidate.shrinkage(local, population, 12., 0.)[0], local)
        np.testing.assert_array_equal(candidate.shrinkage(local, population, 0., 4.)[0], population)
        self.assertEqual(candidate.shrinkage(local, population, 0., 0.)[1:], (1., 0.))

    def test_noiseless_linear_curve_is_inverted_exactly_under_both_priors(self):
        result = candidate.predict_prepared(fixture(), {'White': 1400., 'Black': 1600.})
        for pair in result.values():
            self.assertAlmostEqual(pair['White'], 1800.)
            self.assertAlmostEqual(pair['Black'], 2200.)

    def test_monotonic_bounds_color_symmetry_and_exact_ties(self):
        previous = dict.fromkeys(candidate.METHODS, -1.)
        for accuracy in np.linspace(0., 100., 41):
            prepared = fixture((accuracy, min(100., accuracy+1)), noise=20.)
            result = candidate.predict_prepared(prepared, {'White': 1400., 'Black': 1600.})
            reversed_data = {**prepared, 'observed': {'White': prepared['observed']['Black'],
                                                   'Black': prepared['observed']['White']}}
            swapped = candidate.predict_prepared(reversed_data, {'White': 1600., 'Black': 1400.})
            for method, pair in result.items():
                self.assertTrue(0 <= pair['White'] <= pair['Black'] <= 3200)
                self.assertGreaterEqual(pair['White'], previous[method])
                self.assertEqual(pair['White'], swapped[method]['Black'])
                previous[method] = pair['White']
        equal = candidate.predict_prepared(fixture((60., 60.), 20.), {'White': 1000., 'Black': 2000.})
        self.assertTrue(all(pair['White'] == pair['Black'] for pair in equal.values()))

    def test_metadata_contains_normalized_priors_and_does_not_mutate_inputs(self):
        prepared, actual = fixture(noise=10.), {'White': 1400., 'Black': 1600.}
        before = deepcopy(prepared)
        data = candidate.diagnostics(prepared, actual)
        json.dumps(data, allow_nan=False)
        self.assertEqual(data['account_anchor'], 1500.)
        for method, model in data['methods'].items():
            self.assertAlmostEqual(trapezoid(model['prior_density'], data['grid']), 1.)
            self.assertGreaterEqual(model['affine_slope'], 0.)
        self.assertEqual(data['methods'][candidate.METHODS[1]]['gaussian_sd'], candidate.ELO_LOGISTIC_SD)
        for key in ('grid', 'local_curve', 'population_curve', 'model_curve'):
            np.testing.assert_array_equal(prepared[key], before[key])
        self.assertEqual(prepared['observed'], before['observed'])
        self.assertEqual(actual, {'White': 1400., 'Black': 1600.})

    def test_missing_account_does_not_invent_a_gaussian_and_missing_player_stays_missing(self):
        prepared = fixture((60., None), 20.)
        result = candidate.predict_prepared(prepared, {})
        self.assertTrue(all(pair['Black'] is None for pair in result.values()))
        self.assertEqual(result[candidate.METHODS[0]], result[candidate.METHODS[1]])
        data = candidate.diagnostics(prepared, {})
        self.assertIsNone(data['methods'][candidate.METHODS[1]]['gaussian_sd'])
        none = {'identifiable': False, 'observed': {'White': None, 'Black': None}}
        self.assertTrue(all(pair == {'White': None, 'Black': None} for pair in
                            candidate.predict_prepared(none, {'White': 1600.}).values()))

    def test_bad_variances_and_accounts_are_rejected(self):
        with self.assertRaises(ValueError):
            candidate.shrinkage([60.], [60.], -1., 2.)
        for accounts in ({'White': True}, {'White': float('nan')}, {'Black': -1.}, {'white': 1600.}):
            with self.subTest(accounts=accounts), self.assertRaises(ValueError):
                candidate.predict_prepared(fixture(), accounts)


if __name__ == '__main__':
    unittest.main()
