"""Account blending must precede a single color-order decision projection."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON
from unittest.mock import patch

from tests.analysis import universal_account_sensitivity as estimator
from tests.analysis.universal_competitiveness import project_order


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT/'tests/analysis/output/universal-rating-methods/account-coarse-sensitivity.json'


class AccountProjectionTests(unittest.TestCase):
    def setUp(self):
        self.fit = {'players': {'White': {'average_accuracy': 90.}, 'Black': {'average_accuracy': 80.}}}

    def test_opposed_raw_opinions_are_projected_only_after_account_blending(self):
        quality, ratings = {'White': 1000., 'Black': 2000.}, {'White': 2000., 'Black': 1000.}
        with patch.object(estimator, 'unprojected_quality', return_value=quality), \
                patch.object(estimator, 'project_order', wraps=project_order) as projected:
            result = estimator.predict({}, self.fit, ratings, [])
        self.assertEqual(projected.call_count, len(estimator.ACCOUNT_WEIGHTS))
        for pair in result.values():
            self.assertEqual(pair, {'White': 1500.5, 'Black': 1499.5})
        # Projecting the zero-account pair first would instead manufacture a
        # substantial account-driven gap in this deliberately opposed example.
        prematurely_projected = project_order({side: .95*value for side, value in quality.items()}, self.fit)
        wrong = project_order({side: .9*prematurely_projected[side]/.95+.1*ratings[side] for side in quality}, self.fit)
        self.assertNotEqual(result['universal_account_100'], wrong)

    def test_reference_metadata_cannot_affect_the_account_decision(self):
        quality, ratings = {'White': 2000., 'Black': 1800.}, {'White': 1700., 'Black': 1600.}
        changed = deepcopy(self.fit)
        changed['commercial_reference'] = {'White': 0, 'Black': 9000}
        for player in changed['players'].values():
            player['reference'] = -10000
        with patch.object(estimator, 'unprojected_quality', return_value=quality):
            self.assertEqual(estimator.predict({}, self.fit, ratings, []), estimator.predict({}, changed, ratings, []))


@unittest.skipUnless(ARTIFACT.is_file(), 'Saved16-game experiment artifacts are optional local fixtures.')
@unittest.skip(WITHDRAWN_TEST_REASON)
class SavedAccountArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from analysis.player_rating.bayesian_shared_curve import summarize
        from tests.analysis.test_player_rating_arithmetic_coverage import recorded_rating_cases
        recorded = json.loads(ARTIFACT.read_text(encoding='utf-8'))['players']
        cls.expected = {(row['game'], row['method'], row['side']): row['estimate']
                        for row in recorded}
        actual = {(row['game'], row['side']): row['actual'] for row in recorded}
        if any(row['actual'] != actual[row['game'], row['side']] for row in recorded):
            raise AssertionError('Recorded account inputs disagree between method variants.')
        # This regression belongs to the recorded experiment. A newly added PGN
        # is not a fixture until its evidence and benchmark have been recorded.
        names = sorted({game for game, _, _ in cls.expected}, key=lambda name: int(name[4:]))
        cls.cases = []
        for case in recorded_rating_cases(names):
            name, evidence = case['name'], case['evidence']
            cls.cases.append({'game': name, 'input': {
                'evidence': evidence, 'fit': summarize(evidence),
                'ratings': {side: float(actual[name, side]) for side in ('White', 'Black')}}})
        cls.qualities, cls.predictions = {}, {}
        for case in cls.cases:
            data = case['input']
            calibration = [other['input'] for other in cls.cases if other is not case]
            quality = estimator.unprojected_quality(data['evidence'], data['fit'], calibration)
            cls.qualities[case['game']] = quality
            # Ratings are structurally absent from unprojected_quality, so reuse
            # its measured point pair when checking all account perturbations.
            with patch.object(estimator, 'unprojected_quality', return_value=quality):
                cls.predictions[case['game']] = estimator.predict(data['evidence'], data['fit'], data['ratings'], calibration)

    def test_all16_games_match_the_direct_unprojected_benchmark(self):
        checked = 0
        for case in self.cases:
            for name, pair in self.predictions[case['game']].items():
                for side, value in pair.items():
                    with self.subTest(game=case['game'], method=name, side=side):
                        self.assertAlmostEqual(value, self.expected[case['game'], name, side], places=8)
                    checked += 1
        self.assertEqual(checked, 96)

    def test_all_actual_elo_perturbations_obey_analytic_sensitivity_bounds(self):
        for case in self.cases:
            data, quality = case['input'], self.qualities[case['game']]
            before = self.predictions[case['game']]
            for changed_side in ('White', 'Black'):
                opponent = 'Black' if changed_side == 'White' else 'White'
                for shift in (-200., 200.):
                    ratings = dict(data['ratings'])
                    ratings[changed_side] += shift
                    with patch.object(estimator, 'unprojected_quality', return_value=quality):
                        after = estimator.predict(data['evidence'], data['fit'], ratings, [])
                    for weight in estimator.ACCOUNT_WEIGHTS:
                        name = f'universal_account_{round(1000*weight):03d}'
                        with self.subTest(game=case['game'], method=name, side=changed_side, shift=shift):
                            self.assertLessEqual(abs(after[name][changed_side]-before[name][changed_side]), weight*abs(shift)+1e-9)
                            self.assertLessEqual(abs(after[name][opponent]-before[name][opponent]), .5*weight*abs(shift)+1e-9)

    def test_full_quality_components_ignore_commercial_metadata(self):
        case = self.cases[0]
        data = case['input']
        fit = deepcopy(data['fit'])
        fit['headers'] = {'WhiteEloEstimate': 9999, 'BlackEloEstimate': -9999}
        for player in fit['players'].values():
            player.update(commercial_reference=-10000, reference_elo=10000)
        evidence = {side: {**record, 'reference_elo': 9999, 'commercial_reference': -9999}
                    for side, record in data['evidence'].items()}
        calibration = [{**other['input'],
                        'references': {'White': -9999, 'Black': 9999},
                        'fit': {**other['input']['fit'], 'commercial_reference': 10000}}
                       for other in self.cases if other is not case]
        changed = estimator.unprojected_quality(evidence, fit, calibration)
        for side in ('White', 'Black'):
            self.assertAlmostEqual(changed[side], self.qualities[case['game']][side], places=10)


if __name__ == '__main__':
    unittest.main()
