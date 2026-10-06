"""Declared account inputs, legacy numeric evidence and point-only result contracts."""
import copy
import tempfile
import unittest
from unittest.mock import patch

from analysis.player_rating.context import attach_context, account_ratings, account_signature
from analysis.player_rating.evidence import collect_evidence, validate_evidence
from analysis.player_rating.service import fit_evidence, fit_game, refresh_saved_rating, store_elo_fit
from analysis.settings import CONFIG
from tests.analysis.test_player_rating_bayesian_shared_curve import evaluate, game_records
from tests.analysis.test_player_rating_service import method_file, FIXTURE_SOURCE


class RatingContextTests(unittest.TestCase):
    def test_only_actual_headers_and_explicit_override_enter_context(self):
        headers = {'WhiteElo': '1700', 'BlackEloEstimate': '2800', 'WhiteEloEstimate': '10'}
        self.assertEqual(account_ratings(headers), {'White': 1700, 'Black': None})
        self.assertEqual(account_ratings(headers, {'White': 1800}), {'White': 1800, 'Black': None})
        self.assertEqual(account_signature({'White': 1800}), account_signature({'White': 1800., 'Black': None}))

    def test_optional_context_is_validated_and_arbitrary_metadata_removed(self):
        game, rows = game_records()
        evidence = collect_evidence(game, rows, evaluate)
        original = copy.deepcopy(evidence)
        contextual = attach_context(evidence, {'White': 1700, 'Black': None})
        contextual['White']['reference'] = 2500
        contextual['White']['observations'][0]['commercial_reference'] = 2600
        clean = validate_evidence(contextual)
        self.assertEqual(clean['White']['actual_rating'], 1700.)
        self.assertIn('position_win_probability', clean['White']['observations'][0])
        self.assertNotIn('reference', clean['White'])
        self.assertNotIn('commercial_reference', clean['White']['observations'][0])
        self.assertEqual(evidence, original)
        for rating in (True, float('nan'), -1, 4001):
            contextual['White']['actual_rating'] = rating
            with self.assertRaises(ValueError):
                validate_evidence(contextual)

    def test_legacy_rows_accept_added_before_position_scores(self):
        game, rows = game_records('1. e4 e5 *')
        evidence = collect_evidence(game, rows, evaluate)
        for side in evidence.values():
            for row in side['observations']:
                row.pop('position_win_probability')
        old = copy.deepcopy(evidence)
        with_scores = attach_context(evidence, {}, {'White': [0], 'Black': [100]})
        self.assertEqual(with_scores['White']['observations'][0]['position_win_probability'], .5)
        self.assertGreater(with_scores['Black']['observations'][0]['position_win_probability'], .5)
        self.assertEqual(evidence, old)
        with self.assertRaisesRegex(ValueError, 'cover every'):
            attach_context(evidence, {}, {'White': [], 'Black': [100]})

    def test_point_only_estimator_can_return_no_interval(self):
        source = FIXTURE_SOURCE.replace(
            "return fixed_fit(evidence, central_interval=self.args['central_interval'])",
            "result = fixed_fit(evidence)\n        result['central_interval'] = None\n"
            "        result['interval_scope'] = 'Point estimate; no calibrated interval.'\n"
            "        for player in result['players'].values():\n"
            "            player.update(interval=None, uncertainty=None)\n        return result")
        game, rows = game_records()
        with method_file('point_fixture', source), patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='point_fixture'):
            fit = fit_evidence(collect_evidence(game, rows, evaluate))
        self.assertIsNone(fit['central_interval'])
        self.assertIsNone(fit['players']['White']['interval'])
        self.assertEqual(fit['players']['White']['estimate'], 1300)
        missing = source.replace("result['central_interval'] = None", "result.pop('central_interval')")
        with method_file('missing_interval_fixture', missing), patch.dict(
                CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='missing_interval_fixture'):
            with self.assertRaisesRegex(ValueError, 'explicitly declare central_interval'):
                fit_evidence(collect_evidence(game, rows, evaluate))

    def test_account_edit_invalidates_saved_fit_without_engine_calls(self):
        game, rows = game_records('1. e4 e5 *')
        game.headers.update(WhiteElo='1700', BlackElo='1600')
        with tempfile.TemporaryDirectory() as cache, \
                patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'):
            fit = fit_game(game, rows, evaluate, cache, 'context-fixture')
            analysis = {'headers': dict(game.headers)}
            store_elo_fit(analysis, fit)
            initial = analysis['rating_fit']['context_signature']
            analysis['headers']['WhiteElo'] = '1900'
            refresh_saved_rating(analysis, cache)
            self.assertNotEqual(initial, analysis['rating_fit']['context_signature'])
            # The account-independent method itself is preserved.
            self.assertEqual(analysis['played_elo']['white']['estimate'], fit['players']['White']['estimate'])


if __name__ == '__main__':
    unittest.main()
