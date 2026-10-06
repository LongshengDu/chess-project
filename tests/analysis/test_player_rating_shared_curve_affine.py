"""Synthetic invariants for an affine fit that reads only the current game."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import trapezoid

from analysis import elo_convert
from analysis.player_rating import shared_curve_affine as model
from analysis.player_rating.evidence import CONDITIONING, SCHEMA_VERSION
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.service import fit_evidence, get_estimator
from analysis.settings import CONFIG


def evidence_fixture():
    evidence = {}
    for side, played in (('White', 0), ('Black', 1)):
        observations = []
        for index in range(6):
            q = [100., 75.+index, 30.+index]
            observations.append({'played_index': played, 'qualities': {'position': q, 'root': q.copy()},
                                 'maia_probabilities': [[.15+.02*r, .55-.015*r, .30-.005*r] for r in range(21)],
                                 'weight': 1., 'position_win_probability': .5})
        evidence[side] = {'schema_version': SCHEMA_VERSION, 'conditioning': CONDITIONING,
                          'rating_grid': list(RATINGS), 'actual_rating': 1500.375,
                          'observations': observations}
    return evidence


class SharedCurveAffineTests(unittest.TestCase):
    def setUp(self):
        self.evidence = evidence_fixture()
        self.accounts = {'White': 1600.375, 'Black': 1500.625}

    def test_filename_interface_and_fit_need_no_files_or_external_data(self):
        estimator = get_estimator('shared_curve_affine')
        self.assertIsInstance(estimator, model.Rating)
        with patch('builtins.open', side_effect=AssertionError('No external file reads are allowed.')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('No external asset reads are allowed.')), \
             patch.object(Path, 'read_text', side_effect=AssertionError('No external asset reads are allowed.')):
            result = estimator.fit(self.evidence)
            parameters = estimator.parameters
        self.assertEqual(estimator.name, 'Shared-curve affine fit')
        self.assertEqual(estimator.version, 1)
        self.assertNotIn('calibration_hash', parameters)
        self.assertNotIn('central_interval', parameters)
        self.assertNotIn('calibration', result['diagnostics'])
        self.assertNotIn('hierarchy', result['diagnostics'])
        self.assertIsNone(result['central_interval'])
        self.assertNotIn('posterior_densities', result['diagnostics']['curve'])
        for player in result['players'].values():
            self.assertIsNone(player['interval'])
            self.assertIsNone(player['uncertainty'])
        json.dumps(result, allow_nan=False)

    def test_current_curve_and_variance_are_used_without_population_modification(self):
        measured = model.measure(self.evidence)
        result = model.calculate(self.evidence, self.accounts)
        diagnostic = result['diagnostics']
        np.testing.assert_array_equal(diagnostic['model']['accuracy_curve'], measured['curve']['shared_accuracy'])
        self.assertEqual(diagnostic['model']['measurement_variance'], measured['curve']['likelihood']['accuracy_variance'])
        self.assertAlmostEqual(diagnostic['affine']['accuracy_variance'],
                               diagnostic['affine']['curve_accuracy_variance']+diagnostic['model']['measurement_variance'])
        for side, component in diagnostic['components'].items():
            self.assertEqual(component['observed_accuracy'], measured['sides'][side]['accuracy'])
            self.assertEqual(component['unbounded_estimate'],
                             diagnostic['affine']['prior_mean']+component['accuracy_adjustment'])
            self.assertEqual(result['players'][side]['unrounded_estimate'],
                             float(np.clip(component['unbounded_estimate'], 0., 3200.)))

    def test_linear_noiseless_inverse_and_conditional_variance_reliability(self):
        grid = model.CURVE_ARGS.grid
        prior = model.translated_prior(1400.)
        curve = 40.+.01*grid
        exact = model.affine_moments(curve, 0., prior)
        noisy = model.affine_moments(curve, 20., prior)
        self.assertAlmostEqual(prior['prior_mean']+exact['affine_slope']*(58.-exact['accuracy_mean']), 1800.)
        self.assertAlmostEqual(prior['prior_mean']+exact['affine_slope']*(62.-exact['accuracy_mean']), 2200.)
        self.assertGreater(noisy['affine_slope'], 0.)
        self.assertLess(noisy['affine_slope'], exact['affine_slope'])
        for anchor in (None, 0., 1600., 3200.):
            prior = model.translated_prior(anchor)
            self.assertAlmostEqual(trapezoid(prior['prior_density'], grid), 1.)
            if anchor in (0., 3200.):
                self.assertNotEqual(prior['prior_mean'], anchor)

    def test_color_order_ties_and_common_account_prior(self):
        result = model.calculate(self.evidence, self.accounts)
        swapped = model.calculate({'White': self.evidence['Black'], 'Black': self.evidence['White']},
                                  {'White': self.accounts['Black'], 'Black': self.accounts['White']})
        self.assertEqual(result['players']['White'], swapped['players']['Black'])
        self.assertEqual(result['players']['Black'], swapped['players']['White'])
        same_mean = model.calculate(self.evidence, {'White': 1550.5, 'Black': 1550.5})
        self.assertEqual(result['players'], same_mean['players'])
        tied_evidence = deepcopy(self.evidence)
        tied_evidence['Black'] = deepcopy(tied_evidence['White'])
        tied = model.calculate(tied_evidence, self.accounts)
        self.assertEqual(tied['players']['White']['unrounded_estimate'], tied['players']['Black']['unrounded_estimate'])
        affine = result['diagnostics']['affine']
        actions = np.clip(affine['prior_mean']+affine['affine_slope']*(np.linspace(0., 100., 1001)-affine['accuracy_mean']), 0., 3200.)
        self.assertTrue(np.all(np.diff(actions) >= 0.))

    def test_unrelated_games_reference_labels_and_position_weights_have_no_effect(self):
        before = deepcopy(self.evidence)
        expected = model.calculate(self.evidence, self.accounts)
        altered = deepcopy(self.evidence)
        for record in altered.values():
            record['population'] = [{'curve': [0.]*21, 'actual_elo': 99999.}]
            record['other_games'] = {'ratings': [0., 3200.], 'arbitrary': 'never read'}
            record['commercial_estimate'] = -99999.
            for index, row in enumerate(record['observations']):
                row['weight'] = 1e9 if index % 2 else 1e-9
                row['position_win_probability'] = float(index % 2)
        self.assertEqual(model.calculate(altered, self.accounts), expected)
        self.assertEqual(self.evidence, before)

    def test_empty_flat_forced_and_missing_account_behavior(self):
        empty = {side: {'observations': []} for side in model.SIDES}
        flat = deepcopy(self.evidence)
        forced = deepcopy(self.evidence)
        for side in model.SIDES:
            for row in flat[side]['observations']:
                row['maia_probabilities'] = [row['maia_probabilities'][0]]*21
            for row in forced[side]['observations']:
                row.update(played_index=0, qualities={'position': [100.], 'root': [100.]},
                           maia_probabilities=[[1.]]*21)
        for evidence in (empty, flat, forced):
            result = model.calculate(evidence, self.accounts)
            self.assertTrue(all(row['estimate'] is None for row in result['players'].values()))
            self.assertFalse(result['account_ratings_used'])
            json.dumps(result, allow_nan=False)
        one = deepcopy(self.evidence)
        one['Black']['observations'] = []
        one_result = model.calculate(one, {'White': 1500.})
        self.assertIsNotNone(one_result['players']['White']['estimate'])
        self.assertIsNone(one_result['players']['Black']['estimate'])
        absent = model.calculate(self.evidence)
        self.assertIsNone(absent['diagnostics']['affine']['account_anchor'])
        self.assertFalse(absent['account_ratings_used'])
        self.assertEqual(absent['diagnostics']['affine']['prior_shift'], 0.)

    def test_invalid_accounts_and_variance_are_rejected(self):
        for accounts in ({'white': 1500.}, {'White': True}, {'White': -1.}, {'White': 4001.}, {'Black': float('nan')}):
            with self.subTest(accounts=accounts), self.assertRaises(ValueError):
                model.calculate(self.evidence, accounts)
        for variance in (-1., float('nan'), True):
            with self.subTest(variance=variance), self.assertRaises(ValueError):
                model.affine_moments(np.linspace(50., 100., len(model.CURVE_ARGS.grid)), variance, model.translated_prior(None))
        with self.assertRaises(ValueError):
            model.translated_prior(True)

    def test_service_preserves_native_map_and_translates_prior_for_all_scales(self):
        expected = model.calculate(self.evidence, self.accounts)
        with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='shared_curve_affine'):
            for code in ('lb', 'lr', 'cb', 'cr'):
                evidence = deepcopy(self.evidence)
                for side in model.SIDES:
                    evidence[side]['actual_rating'] = elo_convert.convert(self.accounts[side], 'lb', code, extrapolate=True)
                actual = fit_evidence(evidence, rating_scale=code)
                self.assertEqual(actual['diagnostics']['rating_scale'], 'lb')
                for side in model.SIDES:
                    point = expected['players'][side]['unrounded_estimate']
                    self.assertAlmostEqual(actual['players'][side]['canonical_estimate'], point, places=7)
                    self.assertAlmostEqual(actual['players'][side]['unrounded_estimate'],
                                           elo_convert.convert(point, 'lb', code, extrapolate=True), places=7)
                    self.assertIsNone(actual['players'][side]['interval'])
                if code == 'lb':
                    self.assertEqual(actual['prior'], expected['prior'])
                else:
                    self.assertNotIn('shift', actual['prior'])
                    self.assertAlmostEqual(actual['prior']['native_prior']['shift'], expected['prior']['shift'], places=8)
                    self.assertAlmostEqual(actual['prior']['account_anchor'],
                                           elo_convert.convert(expected['prior']['account_anchor'], 'lb', code, extrapolate=True), places=8)
                    np.testing.assert_allclose(actual['prior']['truncated_to'],
                        [elo_convert.convert(value, 'lb', code, extrapolate=True) for value in expected['prior']['truncated_to']],
                        atol=1e-8, rtol=0.)


if __name__ == '__main__':
    unittest.main()
