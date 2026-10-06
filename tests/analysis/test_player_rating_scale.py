"""Scale-equivariant fits and scale-aware saved-fit invalidation without engines."""
from copy import deepcopy
import math
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from analysis import elo_convert
from analysis.player_rating import service
from analysis.player_rating.scale import rating_context
from analysis.settings import CONFIG
from tests.analysis.test_player_rating_arithmetic_coverage import evidence_fixture
from tests.analysis.test_player_rating_bayesian_shared_curve import evaluate, game_records


METHODS = ('bayesian_shared_curve', 'shared_curve_affine')
SCALES = ('lb', 'lr', 'cb', 'cr')
SIDES = ('White', 'Black')


def contextual_evidence(code, accounts=None):
    accounts = accounts or {'White': 1600.375, 'Black': 1500.625}
    evidence = evidence_fixture()
    for side in SIDES:
        value = accounts.get(side)
        evidence[side]['actual_rating'] = (None if value is None else
            elo_convert.convert(value, 'lb', code, extrapolate=True))
    return evidence


class ScaleFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fits = {}
        for method in METHODS:
            with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD=method):
                cls.fits[method] = {
                    code: service.fit_evidence(contextual_evidence(code), evidence_key='fixture', rating_scale=code)
                    for code in SCALES}

    def test_current_methods_are_equivariant_across_all_four_scales(self):
        for method, by_scale in self.fits.items():
            native = by_scale['lb']
            for code, fit in by_scale.items():
                for side in SIDES:
                    with self.subTest(method=method, scale=code, side=side):
                        expected = native['players'][side]['unrounded_estimate']
                        actual = fit['players'][side]
                        self.assertAlmostEqual(actual['canonical_estimate'], expected, places=6)
                        self.assertAlmostEqual(elo_convert.convert(actual['unrounded_estimate'], code, 'lb', extrapolate=True),
                                               expected, places=6)
                        displayed = elo_convert.convert(expected, 'lb', code, extrapolate=True)
                        self.assertAlmostEqual(actual['unrounded_estimate'], displayed, places=6)
                        self.assertLess(abs(actual['estimate']-displayed), 2.)

    def test_actual_inputs_keep_fractional_precision_and_canonical_diagnostics_stay_native(self):
        for method, by_scale in self.fits.items():
            native = by_scale['lb']
            for code, fit in by_scale.items():
                with self.subTest(method=method, scale=code):
                    self.assertEqual(fit['rating_scale']['scale'], code)
                    self.assertEqual(fit['rating_scale']['native_scale'], 'lb')
                    self.assertEqual(fit['parameters_rating_scale'], 'lb')
                    self.assertEqual(fit['diagnostics']['rating_scale'], 'lb')
                    self.assertEqual(fit['parameters'], native['parameters'])
                    curve, native_curve = deepcopy(fit['diagnostics']['curve']), deepcopy(native['diagnostics']['curve'])
                    # Inverse scale conversion differs by floating-point roundoff;
                    # an account-translated prior inherits that tiny shift.
                    for key in ('prior_weights', 'prior_density'):
                        np.testing.assert_allclose(curve.pop(key), native_curve.pop(key), atol=1e-12, rtol=0.)
                    self.assertEqual(curve, native_curve)
                    for side, actual in (('White', 1600.375), ('Black', 1500.625)):
                        self.assertAlmostEqual(fit['rating_scale']['native_actual_ratings'][side], actual, places=8)
                        expected = elo_convert.convert(actual, 'lb', code, extrapolate=True)
                        self.assertEqual(fit['rating_scale']['actual_ratings'][side], expected)
                        if method != 'bayesian_shared_curve':
                            self.assertAlmostEqual(fit['diagnostics']['components'][side]['actual_rating'], actual, places=8)

    def test_interval_endpoints_are_transformed_individually_and_points_are_not_rerounded_early(self):
        by_scale = self.fits['bayesian_shared_curve']
        for code in ('lr', 'cb', 'cr'):
            fit, native = by_scale[code], by_scale['lb']
            self.assertEqual(fit['central_interval'], native['central_interval'])
            for side in SIDES:
                canonical = native['players'][side]
                player = fit['players'][side]
                expected = [elo_convert.convert(value, 'lb', code, extrapolate=True) for value in canonical['interval']]
                self.assertEqual(player['canonical_interval'], canonical['interval'])
                self.assertEqual(player['interval'], [math.floor(expected[0]), math.ceil(expected[1])])
                transformed_point = elo_convert.convert(canonical['unrounded_estimate'], 'lb', code, extrapolate=True)
                self.assertAlmostEqual(player['unrounded_estimate'], transformed_point)
                rounded_first = elo_convert.convert(canonical['estimate'], 'lb', code, extrapolate=True)
                self.assertGreater(abs(transformed_point-rounded_first), 1e-6)
        for method in ('shared_curve_affine',):
            for fit in self.fits[method].values():
                self.assertIsNone(fit['central_interval'])
                for player in fit['players'].values():
                    self.assertIsNone(player['interval'])
                    self.assertIsNone(player['canonical_interval'])
                    self.assertIsNone(player['uncertainty'])

    def test_explicit_scale_context_preserves_inputs_and_extrapolation_flags(self):
        evidence = contextual_evidence('cr', {'White': 300., 'Black': 3100.})
        before = deepcopy(evidence)
        context = elo_convert.resolve_scale({'Site': 'Chess.com', 'TimeControl': '600+0'})
        with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'):
            fit = service.fit_evidence(evidence, rating_scale=context)
        self.assertEqual(evidence, before)
        self.assertEqual(fit['rating_scale']['source'], 'pgn_headers')
        self.assertTrue(fit['rating_scale']['curve_support_extrapolated'])
        self.assertEqual(fit['rating_scale']['actual_conversion_extrapolated'], {'White': True, 'Black': True})
        for side in SIDES:
            point = fit['players'][side]['canonical_estimate']
            self.assertEqual(fit['rating_scale']['estimate_conversion_extrapolated'][side],
                             not elo_convert.LB_MIN <= point <= elo_convert.LB_MAX)


class SavedScaleTests(unittest.TestCase):
    def test_header_changes_and_converter_version_refit_without_touching_evidence(self):
        game, rows = game_records('1. e4 e5 *')
        game.headers.update(Site='https://lichess.org/fixture', TimeControl='300+0', WhiteElo='1600', BlackElo='1500')
        with tempfile.TemporaryDirectory() as cache, \
             patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'):
            fit = service.fit_game(game, rows, evaluate, cache, 'scale-fixture')
            evidence_key = fit['rating_fit']['evidence_key']
            path = service.evidence_cache_path(cache, evidence_key)
            original_bytes = path.read_bytes()
            analysis = {'headers': dict(game.headers)}
            service.store_elo_fit(analysis, fit)
            # Evidence validation casts accounts to floats; PGN refresh reads
            # integers. Their identical numeric context must reuse the fit.
            with patch.object(service, 'fit_evidence', side_effect=AssertionError('Unchanged fit must be reused')):
                service.refresh_saved_rating(analysis, cache)
            for site, control, code in (('https://lichess.org/fixture', '600+0', 'lr'),
                                        ('Chess.com', '600+0', 'cr'), ('Chess.com', '300+0', 'cb')):
                previous = analysis['rating_fit']['context_signature']
                analysis['headers'].update(Site=site, TimeControl=control)
                with patch.object(service, 'collect_evidence', side_effect=AssertionError('Scale-only refits must reuse chess evidence')), \
                     patch.object(service, 'fit_evidence', wraps=service.fit_evidence) as refit:
                    service.refresh_saved_rating(analysis, cache)
                self.assertEqual(refit.call_count, 1)
                self.assertNotEqual(previous, analysis['rating_fit']['context_signature'])
                self.assertEqual(analysis['played_elo_scale']['scale'], code)
                self.assertEqual(analysis['rating_fit']['evidence_key'], evidence_key)
                self.assertEqual(path.read_bytes(), original_bytes)
                with patch.object(service, 'fit_evidence', side_effect=AssertionError('Unchanged fit must be reused')):
                    service.refresh_saved_rating(analysis, cache)
            previous = analysis['rating_fit']['context_signature']
            with patch.object(elo_convert, 'MODEL_VERSION', 'changed-conversion-model'), \
                 patch.object(service, 'fit_evidence', wraps=service.fit_evidence) as refit:
                service.refresh_saved_rating(analysis, cache)
            self.assertEqual(refit.call_count, 1)
            self.assertNotEqual(previous, analysis['rating_fit']['context_signature'])
            self.assertEqual(analysis['rating_fit']['conversion_version'], 'changed-conversion-model')
            self.assertEqual(analysis['rating_fit']['evidence_key'], evidence_key)
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_fit_game_infers_source_headers_keeps_full_precision_and_reuses_cache(self):
        game, rows = game_records('1. e4 e5 *')
        game.headers.update(Site='Chess.com', TimeControl='600+0', WhiteElo='1500', BlackElo='1400',
                            Link='https://lichess.org/irrelevant')
        with tempfile.TemporaryDirectory() as cache, \
             patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'):
            fit = service.fit_game(game, rows, evaluate, cache, 'scale-fixture')
            self.assertEqual(fit['rating_scale']['scale'], 'cr')
            self.assertEqual(fit['rating_scale']['source'], 'pgn_headers')
            expected = elo_convert.convert(1500., 'cr', 'lb')
            self.assertNotEqual(expected, int(expected))
            self.assertEqual(fit['rating_scale']['native_actual_ratings']['White'], expected)
            game.headers.update(Site='https://lichess.org/fixture', TimeControl='300+0')
            no_engine = Mock(side_effect=AssertionError('Cached chess measurements must be reused'))
            updated = service.fit_game(game, rows, no_engine, cache, 'scale-fixture')
            no_engine.assert_not_called()
            self.assertEqual(fit['rating_fit']['evidence_key'], updated['rating_fit']['evidence_key'])
            self.assertEqual(updated['rating_scale']['scale'], 'lb')
            self.assertEqual(updated['rating_scale']['native_actual_ratings']['White'], 1500.)
            game.headers.update(TimeControl='1500+0')
            with self.assertRaises(elo_convert.UnsupportedRatingScale):
                service.fit_game(game, rows, no_engine, cache, 'scale-fixture')
            overridden = service.fit_game(game, rows, no_engine, cache, 'scale-fixture', rating_scale='lb')
            self.assertEqual(overridden['rating_scale']['source'], 'override')
            self.assertEqual(overridden['rating_fit']['evidence_key'], updated['rating_fit']['evidence_key'])
            no_engine.assert_not_called()

    def test_legacy_missing_metadata_fallback_is_explicit_but_unsupported_metadata_is_not_reclassified(self):
        context = rating_context({'Site': '?'})
        self.assertEqual(context['scale'], 'lb')
        self.assertEqual(context['source'], 'assumed_native')
        self.assertIn('assumption', context)
        with self.assertRaises(elo_convert.UnsupportedRatingScale):
            rating_context({'Site': 'https://lichess.org/example', 'TimeControl': '60+0'})


if __name__ == '__main__':
    unittest.main()
