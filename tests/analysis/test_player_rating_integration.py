"""Coach dispatch and saved-evidence refresh through the common rating interface."""
from analysis.settings import CONFIG as ANALYSIS_CONFIG
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from xml.etree import ElementTree

import analysis.player_rating as rating_package
from unittest.mock import patch

import chess.pgn

from analysis.game.pipeline import analyze_game
from analysis.game.summary import compact_summary
from analysis.player_rating.service import refresh_saved_rating
from analysis.player_rating.service import evidence_cache_path, get_estimator
from analysis.player_rating.bayesian_shared_curve import Args
from tests.coach.fixtures import FakeEngines
from tests.coach.fixtures_rating import replacement_estimator


class CoachRatingTests(unittest.TestCase):
    def setUp(self):
        selector = patch.dict(ANALYSIS_CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve')
        selector.start()
        self.addCleanup(selector.stop)

    def test_fresh_pipeline_and_unchanged_saved_refresh_replace_existing_figures(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 Nc6 *'))
        engines = FakeEngines()
        self.addCleanup(engines._temp.cleanup)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'game-full'/'player-rating'
            analysis = analyze_game(game, engines, 'white', 1600, progress=lambda _: None,
                                    rating_output_dir=output)
            self.assertEqual(json.loads((output/'fit.json').read_text(encoding='utf-8'))['rating_fit'], analysis['rating_fit'])
            self.assertEqual(ElementTree.parse(output/'analysis.svg').getroot().tag,
                             '{http://www.w3.org/2000/svg}svg')
            (output/'analysis.svg').write_bytes(b'obsolete figure')
            (output/'prior.svg').unlink()
            (output/'analysis.png').write_bytes(b'obsolete format')
            (output/'posterior.svg').write_text('old separate plot', encoding='utf-8')
            before = copy.deepcopy(analysis)
            with patch('analysis.player_rating.service.fit_evidence', side_effect=AssertionError('Unchanged fit must be reused')):
                refresh_saved_rating(analysis, engines.cache.directory, output_dir=output)
            self.assertEqual(analysis, before)
            for stem in ('analysis', 'prior'):
                self.assertEqual(ElementTree.parse(output/f'{stem}.svg').getroot().tag,
                                 '{http://www.w3.org/2000/svg}svg')
                self.assertFalse((output/f'{stem}.png').exists())
            self.assertFalse((output/'posterior.svg').exists())
            self.assertFalse(list(engines.cache.directory.rglob('*.png')))
            self.assertFalse(list(engines.cache.directory.rglob('*.svg')))

    def test_default_curve_refreshes_interval_without_more_engine_calls(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 Nc6 *'))
        engines = FakeEngines()
        analysis = analyze_game(game, engines, 'white', 1600, progress=lambda _: None)
        self.assertEqual(analysis['played_elo_method'], 'bayesian_shared_curve')
        self.assertEqual(analysis['played_elo_central_interval'], .20)
        self.assertEqual(analysis['played_elo_rating_range'], [0, 3200])
        self.assertNotIn('played_elo_confidence', analysis)
        self.assertNotIn('played_elo_base', analysis)
        self.assertIn('Bayesian shared-curve fit', compact_summary(analysis)['elo_note'])
        first = copy.deepcopy(analysis['played_elo'])
        calls = (len(engines.pair_calls), len(engines.sf_calls))
        with patch('analysis.player_rating.bayesian_shared_curve.ARGS', Args(central_interval=.50)):
            refresh_saved_rating(analysis, engines.cache.directory)
            self.assertEqual(analysis['played_elo_central_interval'], .50)
        refresh_saved_rating(analysis, engines.cache.directory)
        self.assertEqual(analysis['played_elo'], first)
        self.assertEqual((len(engines.pair_calls), len(engines.sf_calls)), calls)

    def test_replacement_estimator_runs_fresh_and_on_saved_evidence(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 Nc6 *'))
        engines = FakeEngines()
        analysis = analyze_game(game, engines, 'white', 1600, progress=lambda _: None)
        initial = copy.deepcopy(analysis['played_elo'])
        key = analysis['rating_fit']['evidence_key']
        calls = (len(engines.pair_calls), len(engines.sf_calls))
        package_paths = list(rating_package.__path__)
        with replacement_estimator() as module_path:
            estimator = get_estimator()
            self.assertEqual(estimator.__class__.__module__, 'analysis.player_rating.test_replacement')
            self.assertTrue(module_path.is_file())
            self.assertEqual(estimator.version, 1)
            refresh_saved_rating(analysis, engines.cache.directory)
            self.assertEqual(analysis['played_elo_method'], 'test_replacement')
            self.assertEqual(analysis['played_elo']['white']['estimate'], 1300)
            self.assertEqual(analysis['played_elo']['black']['estimate'], 1700)
            self.assertIn('Test replacement fit', compact_summary(analysis)['elo_note'])
            self.assertNotIn('curve', analysis['played_elo_diagnostics'])
            self.assertEqual(analysis['rating_fit']['evidence_key'], key)
            self.assertEqual((len(engines.pair_calls), len(engines.sf_calls)), calls)
            fresh = analyze_game(game, engines, 'white', 1600, progress=lambda _: None)
            self.assertEqual(fresh['played_elo'], analysis['played_elo'])
            self.assertEqual(fresh['rating_fit']['method'], 'test_replacement')
        self.assertNotIn('analysis.player_rating.test_replacement', sys.modules)
        self.assertFalse(hasattr(rating_package, 'test_replacement'))
        self.assertEqual(list(rating_package.__path__), package_paths)
        self.assertFalse(module_path.exists())
        refresh_saved_rating(analysis, engines.cache.directory)
        self.assertEqual(analysis['played_elo'], initial)

    def test_previous_shared_cache_is_read_and_metadata_migrated(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        engines = FakeEngines()
        analysis = analyze_game(game, engines, 'white', 1600, progress=lambda _: None)
        key = analysis['rating_fit']['evidence_key']
        legacy_cache = engines.cache.directory / 'elo-calibration' / f'calibration-{key}.json'
        legacy_cache.parent.mkdir()
        evidence_cache_path(engines.cache.directory, key).replace(legacy_cache)
        analysis['elo_calibration'] = analysis.pop('rating_fit')
        analysis['elo_calibration']['model_signature'] = 'previous-version'
        analysis['played_elo_curve'] = {'obsolete': True}
        analysis['played_elo_base'] = {'obsolete': True}
        analysis['played_elo_confidence'] = .68
        calls = (len(engines.pair_calls), len(engines.sf_calls))
        refresh_saved_rating(analysis, engines.cache.directory)
        self.assertEqual(analysis['rating_fit']['method'], 'bayesian_shared_curve')
        for field in ('elo_calibration', 'played_elo_curve', 'played_elo_base', 'played_elo_confidence'):
            self.assertNotIn(field, analysis)
        self.assertEqual((len(engines.pair_calls), len(engines.sf_calls)), calls)

    def test_removed_methods_reject_before_engine_work(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        for method in ('bayesian', 'minimax', 'bayesian_calibration', 'minimax_calibration'):
            engines = FakeEngines()
            with patch.dict(ANALYSIS_CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD=method):
                with self.assertRaisesRegex(ValueError, 'Unknown rating method'):
                    analyze_game(game, engines, 'white', 1600, progress=lambda _: None)
            self.assertFalse(engines.pair_calls or engines.sf_calls)

    def test_old_saved_analysis_requires_evidence_instead_of_silent_old_method(self):
        with self.assertRaisesRegex(ValueError, '--analysis-only'):
            refresh_saved_rating({}, '.')


if __name__ == '__main__':
    unittest.main()
