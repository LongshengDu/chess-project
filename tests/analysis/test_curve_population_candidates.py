"""Affine moment identities and common-map invariants before benchmark scoring."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from tests.analysis.curve_population_candidates import Prepared, describe, moments, predict_prepared


def fixture(accuracies=(58., 62.), noise=0.):
    grid = np.linspace(0., 3200., 641)
    return Prepared(grid, 40.+.01*grid, np.zeros_like(grid), noise, accuracies, True, 'synthetic', 16)


class PopulationCandidateTests(unittest.TestCase):
    def test_noiseless_affine_model_recovers_its_exact_inverse(self):
        prepared = fixture()
        model = moments(prepared)
        self.assertAlmostEqual(model['coefficient'], 100.)
        result = predict_prepared(prepared, {})
        self.assertAlmostEqual(result['population_affine_common_account5']['White'], 1800.)
        self.assertAlmostEqual(result['population_affine_account_prior']['Black'], 2200.)

    def test_measurement_noise_reduces_affine_sensitivity(self):
        clear, noisy = moments(fixture()), moments(fixture(noise=100.))
        self.assertLess(noisy['coefficient'], clear['coefficient'])
        self.assertAlmostEqual(noisy['accuracy_variance']-clear['accuracy_variance'], 100.)

    def test_each_rule_is_monotone_bounded_and_color_symmetric(self):
        previous = dict.fromkeys(describe(), -1.)
        for accuracy in (0., 40., 55., 70., 100.):
            prepared = fixture((accuracy, min(100., accuracy+1)), noise=20.)
            result = predict_prepared(prepared, {'White': 1400., 'Black': 1600.})
            swapped = predict_prepared(replace(prepared, accuracies=tuple(reversed(prepared.accuracies))),
                                       {'White': 1600., 'Black': 1400.})
            for name, pair in result.items():
                self.assertLessEqual(pair['White'], pair['Black'])
                self.assertGreaterEqual(pair['White'], previous[name])
                self.assertTrue(all(0 <= value <= 3200 for value in pair.values()))
                self.assertEqual(pair['White'], swapped[name]['Black'])
                previous[name] = pair['White']

    def test_common_five_percent_anchor_has_exact_bounded_sensitivity(self):
        prepared = fixture(noise=10.)
        first = predict_prepared(prepared, {'White': 1400., 'Black': 1600.})
        changed = predict_prepared(prepared, {'White': 1600., 'Black': 1600.})
        for side in ('White', 'Black'):
            self.assertAlmostEqual(changed['population_affine_common_account5'][side]-
                                   first['population_affine_common_account5'][side], 5.)

    def test_missing_observation_and_accounts_remain_explicit(self):
        prepared = fixture((60., None))
        accounts = {'White': None, 'Black': None}
        before = deepcopy(accounts)
        result = predict_prepared(prepared, accounts)
        self.assertEqual(accounts, before)
        self.assertEqual(result, predict_prepared(prepared, {}))
        self.assertTrue(all(pair['Black'] is None for pair in result.values()))
        self.assertTrue(all(pair == {'White': None, 'Black': None} for pair in
                            predict_prepared(replace(prepared, identifiable=False), {}).values()))


if __name__ == '__main__':
    unittest.main()
