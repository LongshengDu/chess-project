"""Calibration privacy, color symmetry, exclusion and numeric-contract checks."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON

import numpy as np

from analysis.player_rating.calibration import DEFAULT_PATH, evidence_fingerprint, load_calibration
from tests.analysis.build_rating_calibration import ROOT, context_moments, load_numeric_context


def context():
    return {side: {'observations': [
        {'qualities': {'position': [50., 100.]}, 'maia_probabilities': [[.4, .6]]*21,
         'position_win_probability': .4, 'played_index': index}]}
        for side, index in (('White', 0), ('Black', 1))}


class CalibrationTests(unittest.TestCase):
    def test_outcomes_and_ratings_do_not_enter_fingerprint_or_moments(self):
        evidence, changed = context(), context()
        for record in changed.values():
            record.update(actual_rating=2600, reference=3000, name='unused')
            record['observations'][0]['played_index'] = 1-record['observations'][0]['played_index']
        self.assertEqual(evidence_fingerprint(evidence), evidence_fingerprint(changed))
        for competitive in (False, True):
            self.assertEqual(context_moments(evidence, competitive=competitive),
                             context_moments(changed, competitive=competitive))
        changed['White']['observations'][0]['position_win_probability'] = .5
        self.assertNotEqual(evidence_fingerprint(evidence), evidence_fingerprint(changed))

    def test_color_swap_preserves_fingerprint(self):
        evidence = context()
        evidence['White']['observations'][0]['qualities']['position'] = [45., 98.]
        changed = deepcopy({'White': evidence['Black'], 'Black': evidence['White']})
        for record in changed.values():
            for row in record['observations']:
                row['position_win_probability'] = 1-row['position_win_probability']
        self.assertEqual(evidence_fingerprint(evidence), evidence_fingerprint(changed))

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_frozen_asset_whitelist_and_immutability(self):
        payload = json.loads(DEFAULT_PATH.read_text(encoding='utf-8'))
        corpus = load_calibration()
        self.assertEqual(len(corpus.records), 16)
        unseen = corpus.for_evidence(context())
        self.assertEqual((len(unseen.records), unseen.excluded_count), (16, 0))
        self.assertTrue(all(set(row) == {'fingerprint', 'arithmetic', 'competitive'} for row in payload['contexts']))
        self.assertTrue(all(set(row[name]) == {'knots', 'variance'} for row in payload['contexts']
                            for name in ('arithmetic', 'competitive')))
        with self.assertRaises(FrozenInstanceError):
            corpus.records = ()
        mean, variance = corpus.population(np.arange(0., 3201., 5.))
        self.assertTrue(np.all(np.isfinite(mean)) and np.all(np.diff(mean) >= -1e-10))
        self.assertTrue(np.all(variance >= 0))

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_every_saved_target_excluded_and_generated_moments_match(self):
        from analysis.settings import CONFIG
        corpus = load_calibration()
        records = {record.fingerprint: record for record in corpus.records}
        sources = sorted((ROOT/'games').glob('output/game*-full/analysis.json'))
        if len(sources) < len(records):
            self.skipTest('The original saved calibration evidence is not installed.')
        seen = set()
        for source in sources:
            evidence = load_numeric_context(source, CONFIG['ANALYSIS']['CACHE_DIR'])
            fingerprint = evidence_fingerprint(evidence)
            if fingerprint not in records:
                continue
            seen.add(fingerprint)
            filtered = corpus.for_evidence(evidence)
            self.assertEqual((len(filtered.records), filtered.excluded_count), (15, 1))
            self.assertTrue(all(row.fingerprint != fingerprint for row in filtered.records))
            self.assertEqual(len(corpus.records), 16)
            swapped = deepcopy({'White': evidence['Black'], 'Black': evidence['White']})
            for record in swapped.values():
                for row in record['observations']:
                    row['position_win_probability'] = 1-row['position_win_probability']
            self.assertEqual(evidence_fingerprint(swapped), fingerprint)
            self.assertEqual(len(corpus.for_evidence(swapped).records), 15)
            for name, competitive in (('arithmetic', False), ('competitive', True)):
                generated = context_moments(evidence, competitive=competitive)
                frozen = getattr(records[fingerprint], name)
                np.testing.assert_array_equal(generated['knots'], frozen.knots)
                self.assertEqual(generated['variance'], frozen.variance)
        self.assertEqual(seen, set(records))

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_asset_matches_selected_experiment_curves_and_population(self):
        from analysis.player_rating.bayesian_shared_curve import summarize
        from analysis.settings import CONFIG
        from tests.analysis.universal_competitiveness import weighted_fit
        from tests.analysis.edge_global_quality import _population
        corpus = load_calibration()
        records = {record.fingerprint: record for record in corpus.records}
        sources = sorted((ROOT/'games').glob('output/game*-full/analysis.json'))
        if len(sources) < len(records):
            self.skipTest('The original saved calibration evidence is not installed.')
        arithmetic, transformed = [], []
        for source in sources:
            evidence = load_numeric_context(source, CONFIG['ANALYSIS']['CACHE_DIR'])
            fingerprint = evidence_fingerprint(evidence)
            if fingerprint not in records:
                continue
            record = records[fingerprint]
            # The experimental fit interface needs an index, but calibration
            # curves never use its value: choose a synthetic index for all rows.
            for side in evidence.values():
                for row in side['observations']:
                    row['played_index'] = 0
            ordinary = summarize(evidence)
            competitive = weighted_fit(evidence, .5)
            arithmetic.append({'fit': ordinary})
            transformed.append({'fit': competitive})
            for name, fit in (('arithmetic', ordinary), ('competitive', competitive)):
                expected = fit['diagnostics']['curve']
                frozen = getattr(record, name)
                np.testing.assert_allclose(frozen.knots, expected['monotone_expected_accuracy'], atol=1e-12, rtol=0)
                self.assertLess(abs(frozen.variance-expected['likelihood']['accuracy_variance']), 1e-12)
        grid = np.arange(0., 3201., 5.)
        self.assertEqual(len(arithmetic), len(records))
        for name, calibration in (('arithmetic', arithmetic), ('competitive', transformed)):
            _, variances, mean, between = _population(calibration, grid)
            frozen_mean, frozen_variance = corpus.population(grid, name)
            np.testing.assert_allclose(frozen_mean, mean, atol=1e-12, rtol=0)
            np.testing.assert_allclose(frozen_variance, between+variances.mean(), atol=1e-12, rtol=0)

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_malformed_assets_rejected(self):
        for mutation in ('duplicate', 'nonmonotone', 'variance', 'label', 'grid', 'count'):
            with self.subTest(mutation=mutation), TemporaryDirectory() as folder:
                payload = json.loads(DEFAULT_PATH.read_text(encoding='utf-8'))
                if mutation == 'duplicate':
                    payload['contexts'][1] = payload['contexts'][0]
                elif mutation == 'nonmonotone':
                    payload['contexts'][0]['arithmetic']['knots'][0] = 100.
                elif mutation == 'variance':
                    payload['contexts'][0]['arithmetic']['variance'] = -1.
                elif mutation == 'label':
                    payload['contexts'][0]['actual_rating'] = 1600
                elif mutation == 'grid':
                    payload['rating_grid'][0] = 500
                else:
                    payload['provenance']['context_count'] = 17
                target = Path(folder)/'calibration.json'
                target.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_calibration(target)

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_asset_content_change_updates_identity(self):
        original = load_calibration()
        payload = json.loads(DEFAULT_PATH.read_text(encoding='utf-8'))
        payload['contexts'][0]['arithmetic']['variance'] += .01
        with TemporaryDirectory() as folder:
            target = Path(folder)/'calibration.json'
            target.write_text(json.dumps(payload), encoding='utf-8')
            self.assertNotEqual(load_calibration(target).content_hash, original.content_hash)


if __name__ == '__main__':
    unittest.main()
