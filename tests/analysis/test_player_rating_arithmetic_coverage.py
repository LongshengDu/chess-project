"""Arithmetic-coverage promotion parity, missing inputs and decision invariants."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON
from unittest.mock import patch

from analysis.player_rating.arithmetic_coverage import Args, Rating, calculate
from analysis.player_rating.calibration import load_calibration
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.service import fit_evidence, get_estimator, rating_signature
from analysis.settings import CONFIG


ROOT = Path(__file__).resolve().parents[2]
SIDES = ('White', 'Black')
ARCHIVE = ROOT/'tests/analysis/output/rating-methods-games0-18'


def recorded_rating_cases(names):
    """Read hash-verified original inputs, not the mutable games collection.

    These experiments predate source-scale normalization. Their recorded
    numeric account values stay on the original coordinate convention; new
    integration tests separately verify today's PGN scale interpretation.
    """
    from analysis.player_rating.context import saved_context
    from analysis.player_rating.evidence import validate_evidence
    from analysis.player_rating.service import evidence_cache_path
    manifest_path = ARCHIVE/'comparison.json'
    if not manifest_path.is_file():
        raise unittest.SkipTest('Optional original rating-experiment manifest is absent.')
    manifest_bytes = manifest_path.read_bytes()
    manifest = {row['game']: row for row in json.loads(manifest_bytes)['games']}
    cases = []
    for name in names:
        recorded = manifest[name]
        snapshot = ARCHIVE/'original-analysis'/(name+'.json')
        cache = evidence_cache_path(CONFIG['ANALYSIS']['CACHE_DIR'], recorded['evidence_key'])
        inputs = {manifest_path: sha256(manifest_bytes).hexdigest()}
        content = []
        for path, suffix in ((snapshot, f'/games/output/{name}-full/analysis.json'),
                              (cache, '/'+cache.name)):
            if not path.is_file():
                raise unittest.SkipTest(f'Optional archived rating input is absent: {path}.')
            data = path.read_bytes()
            expected = [digest for original, digest in recorded['input_sha256'].items()
                        if original.replace('\\', '/').endswith(suffix)]
            if len(expected) != 1 or sha256(data).hexdigest() != expected[0]:
                raise AssertionError(f'Archived experiment input does not match its recorded hash: {path}.')
            inputs[path] = expected[0]
            content.append(json.loads(data))
        analysis, evidence = content
        if analysis['rating_fit']['evidence_key'] != recorded['evidence_key']:
            raise AssertionError('Archived analysis and evidence identities disagree.')
        evidence = validate_evidence(saved_context(validate_evidence(evidence), analysis))
        cases.append({'name': name, 'number': recorded['number'], 'analysis': analysis,
                      'evidence': evidence, 'inputs': inputs})
    return cases


def evidence_fixture():
    """Same candidate contexts, different played qualities; no reference labels."""
    evidence = {}
    for side, played in (('White', 0), ('Black', 1)):
        rows = []
        for move in range(6):
            quality = [100., 75.+move, 30.+move]
            rows.append({'played_index': played, 'qualities': {'position': quality[:], 'root': quality[:]},
                         'weight': 1., 'maia_probabilities': [
                             [.15+.02*r, .55-.015*r, .30-.005*r] for r in range(len(RATINGS))],
                         'position_win_probability': .45+.05*move})
        evidence[side] = {'conditioning': 'equal_opponent', 'rating_grid': list(RATINGS),
                          'observations': rows}
    return evidence


def points(result):
    return {side: result['players'][side]['unrounded_estimate'] for side in SIDES}


@unittest.skip(WITHDRAWN_TEST_REASON)
class ArithmeticCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = evidence_fixture()
        cls.corpus = load_calibration()
        cls.accounts = {'White': 1600., 'Black': 1500.}
        cls.result = calculate(cls.evidence, cls.accounts, cls.corpus)

    def test_filename_loader_and_public_service_preserve_point_only_contract(self):
        self.assertIsInstance(get_estimator('arithmetic_coverage'), Rating)
        evidence = deepcopy(self.evidence)
        for side in SIDES:
            evidence[side]['actual_rating'] = self.accounts[side]
        with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='arithmetic_coverage'):
            fit = fit_evidence(evidence, evidence_key='synthetic')
            changed = rating_signature(args=Args(account_weight=.1))
        self.assertEqual(points(fit), points(self.result))
        self.assertEqual(fit['rating_fit']['method'], 'arithmetic_coverage')
        self.assertNotEqual(fit['rating_fit']['model_signature'], changed['model_signature'])
        self.assertIsNone(fit['central_interval'])
        self.assertNotIn('posterior_densities', fit['diagnostics']['curve'])
        self.assertNotIn('central_interval', fit['parameters'])
        for player in fit['players'].values():
            self.assertIsNone(player['interval'])
            self.assertIsNone(player['uncertainty'])
        json.dumps(fit, allow_nan=False)

    def test_inputs_are_unchanged_and_reference_metadata_is_ignored(self):
        evidence, accounts = deepcopy(self.evidence), dict(self.accounts)
        for record in evidence.values():
            record['commercial_reference'] = 9999
            record['headers'] = {'WhiteEloEstimate': '0', 'BlackEloEstimate': '9999'}
        before = deepcopy(evidence)
        self.assertEqual(calculate(evidence, accounts, self.corpus), self.result)
        self.assertEqual(evidence, before)
        self.assertEqual(accounts, self.accounts)

    def test_missing_and_explicit_null_accounts_use_the_same_unanchored_decision(self):
        absent = calculate(self.evidence, {}, self.corpus)
        null = calculate(self.evidence, {'White': None, 'Black': None}, self.corpus)
        self.assertEqual(absent, null)
        self.assertEqual(absent, calculate(self.evidence, None, self.corpus))
        self.assertFalse(absent['account_ratings_used'])
        for side, row in absent['diagnostics']['components'].items():
            self.assertIsNone(row['common_account_rating'])
            self.assertEqual(row['account_weight'], 0.)
            self.assertEqual(row['coverage_mean'], absent['players'][side]['unrounded_estimate'])
        single = calculate(self.evidence, {'White': 3500.}, self.corpus)
        self.assertTrue(single['account_ratings_used'])
        self.assertEqual(single['diagnostics']['components']['White']['actual_rating'], 3500.)
        self.assertIsNone(single['diagnostics']['components']['Black']['actual_rating'])
        for side in SIDES:
            row = single['diagnostics']['components'][side]
            self.assertEqual(row['common_account_rating'], 3200.)
            self.assertEqual(row['account_weight'], .05)

    def test_common_anchor_changes_both_players_equally_without_changing_order(self):
        baseline = points(self.result)
        for shifted in (('White',), ('Black',), SIDES):
            for delta in (-200., 200.):
                accounts = dict(self.accounts)
                for side in shifted:
                    accounts[side] += delta
                changed = points(calculate(self.evidence, accounts, self.corpus))
                expected = .05*len(shifted)*delta/2
                for side in SIDES:
                    self.assertAlmostEqual(changed[side]-baseline[side], expected, places=9)
                self.assertAlmostEqual(changed['White']-changed['Black'], baseline['White']-baseline['Black'])
        single = points(calculate(self.evidence, {'White': 1600.}, self.corpus))
        changed = points(calculate(self.evidence, {'White': 1800.}, self.corpus))
        for side in SIDES:
            self.assertAlmostEqual(changed[side]-single[side], 10.)

    def test_color_swap_preserves_points_and_equal_accuracies_need_no_minimum_gap(self):
        swapped = {side: deepcopy(self.evidence[other]) for side, other in
                   (('White', 'Black'), ('Black', 'White'))}
        for record in swapped.values():
            for row in record['observations']:
                row['position_win_probability'] = 1-row['position_win_probability']
        result = calculate(swapped, {'White': 1500., 'Black': 1600.}, self.corpus)
        self.assertAlmostEqual(points(result)['White'], points(self.result)['Black'])
        self.assertAlmostEqual(points(result)['Black'], points(self.result)['White'])
        equal = deepcopy(self.evidence)
        equal['Black'] = deepcopy(equal['White'])
        result = calculate(equal, self.accounts, self.corpus)
        self.assertEqual(points(result)['White'], points(result)['Black'])

    def test_better_observed_quality_never_lowers_either_rating(self):
        previous = dict.fromkeys(SIDES, -1.)
        for played in (2, 1, 0):
            evidence = deepcopy(self.evidence)
            for record in evidence.values():
                for row in record['observations']:
                    row['played_index'] = played
            current = points(calculate(evidence, self.accounts, self.corpus))
            for side in SIDES:
                self.assertGreaterEqual(current[side], previous[side])
                self.assertTrue(0 <= current[side] <= 3200)
            previous = current

    def test_missing_and_uninformative_observations_do_not_invent_estimates(self):
        evidence = deepcopy(self.evidence)
        evidence['Black']['observations'] = []
        result = calculate(evidence, self.accounts, self.corpus)
        self.assertIsNotNone(result['players']['White']['estimate'])
        self.assertIsNone(result['players']['Black']['estimate'])
        evidence['White']['observations'] = []
        result = calculate(evidence, self.accounts, self.corpus)
        self.assertTrue(all(player['estimate'] is None for player in result['players'].values()))
        flat = deepcopy(self.evidence)
        for record in flat.values():
            for row in record['observations']:
                row['qualities'] = {'position': [100.]*3, 'root': [100.]*3}
        result = calculate(flat, self.accounts, self.corpus)
        self.assertTrue(all(player['estimate'] is None for player in result['players'].values()))

    def test_arguments_and_account_inputs_are_validated_and_asset_identifies_cache(self):
        for value in (True, float('nan'), -.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Args(account_weight=value)
        for accounts in ({'White': True}, {'White': float('nan')}, {'White': -1},
                         {'Black': 4001}, {'white': 1600}):
            with self.subTest(accounts=accounts), self.assertRaises(ValueError):
                calculate(self.evidence, accounts, self.corpus)
        changed = Rating(calibration=replace(self.corpus, content_hash='changed'))
        self.assertNotEqual(changed.parameters, Rating(calibration=self.corpus).parameters)


@unittest.skipUnless((ARCHIVE/'comparison.json').is_file(), 'Archived 19-game inputs are optional local fixtures.')
@unittest.skip(WITHDRAWN_TEST_REASON)
class ArithmeticCoverageSavedParityTests(unittest.TestCase):
    def test_all38_unrounded_points_match_selected_research_and_exclude_exact_targets(self):
        from analysis.player_rating.context import saved_ratings
        from tests.analysis.simple_arithmetic_likelihood import predict
        cases = recorded_rating_cases([f'game{number}' for number in range(19)])
        corpus = load_calibration()
        self.assertEqual(len(cases), 19)
        inputs = {}
        for case in cases:
            inputs.update(case['inputs'])
            actual = saved_ratings(case['analysis'])
            expected = predict(case['evidence'], actual)['arithmetic_coverage_common_account']
            result = calculate(case['evidence'], actual, corpus)
            calibration = result['diagnostics']['calibration']
            self.assertEqual(calibration['contexts_excluded'], int(case['number'] < 16))
            self.assertEqual(calibration['contexts_used'], 15 if case['number'] < 16 else 16)
            for side in SIDES:
                with self.subTest(game=case['name'], side=side):
                    self.assertAlmostEqual(result['players'][side]['unrounded_estimate'], expected[side], places=8)
        # Fitting and comparison are read-only even when local full-game fixtures exist.
        for path, digest in inputs.items():
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
