"""Filename-based estimators, abstract interface, validation and cache reuse."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import Mock, patch

import chess
import chess.pgn

import analysis.player_rating as rating_package
from analysis.player_rating.interface import PlayerRating
from analysis.player_rating.service import (evidence_cache_path, fit_evidence, fit_game,
                             get_estimator, rating_signature)
from analysis.player_rating.evidence import collect_evidence, validate_evidence
from analysis.settings import CONFIG
from tests.analysis.test_player_rating_bayesian_shared_curve import evaluate, game_records


FIXTURE_SOURCE = '''
from analysis.player_rating.interface import PlayerRating
from tests.coach.fixtures_rating import fixed_fit

class Rating(PlayerRating):
    def __init__(self, args=None):
        self.args = {'central_interval': .5} if args is None else args

    @property
    def parameters(self):
        return self.args

    def fit(self, evidence):
        return fixed_fit(evidence, central_interval=self.args['central_interval'])
'''


@contextmanager
def method_file(method, source=FIXTURE_SOURCE):
    """Use an actual new module file; no import or estimator registry mocks."""
    module_name = f'analysis.player_rating.{method}'
    sentinel = object()
    prior_module = sys.modules.get(module_name, sentinel)
    prior_attribute = getattr(rating_package, method, sentinel)
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(rating_package, '__path__', [directory, *rating_package.__path__]):
        Path(directory, f'{method}.py').write_text(textwrap.dedent(source), encoding='utf-8')
        importlib.invalidate_caches()
        try:
            yield
        finally:
            if prior_module is sentinel:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = prior_module
            if prior_attribute is sentinel:
                rating_package.__dict__.pop(method, None)
            else:
                setattr(rating_package, method, prior_attribute)
            importlib.invalidate_caches()


class RatingInterfaceTests(unittest.TestCase):
    def setUp(self):
        selector = patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve')
        selector.start()
        self.addCleanup(selector.stop)

    def test_old_methods_are_removed_not_aliases(self):
        for name in ('bayesian', 'minimax', 'bayesian_fit', 'bayesian_calibration', 'minimax_calibration'):
            with self.assertRaisesRegex(ValueError, 'Unknown rating method'):
                get_estimator(name)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            get_estimator('missing_rating_file')

    def test_file_and_yaml_are_the_only_changes_required_for_a_new_estimator(self):
        game, rows = game_records()
        evidence = collect_evidence(game, rows, evaluate)
        with method_file('fixture_fit'), patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='fixture_fit'):
            estimator = get_estimator()
            self.assertIsInstance(estimator, PlayerRating)
            self.assertEqual(estimator.__class__.__module__, 'analysis.player_rating.fixture_fit')
            result = fit_evidence(evidence, evidence_key='fixture')
        self.assertEqual(result['name'], 'Fixture Fit')
        self.assertEqual(result['method_id'], 'fixture_fit')
        self.assertEqual(result['players']['White']['estimate'], 1300)
        self.assertEqual(result['rating_fit']['method_version'], 1)
        self.assertEqual(result['rating_fit']['evidence_key'], 'fixture')
        self.assertEqual(result['central_interval'], .5)
        self.assertEqual(result['rating_range'], [600, 2600])

    def test_signature_changes_with_version_and_method_args_but_evidence_is_reused(self):
        game, rows = game_records()
        callback = Mock(side_effect=evaluate)
        with tempfile.TemporaryDirectory() as directory:
            initial = fit_game(game, rows, callback, directory, 'fixture-model')
            count = callback.call_count
            with method_file('fixture_fit'), patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='fixture_fit'):
                first = fit_game(game, rows, callback, directory, 'fixture-model')
                original_signature = rating_signature()
                with patch.object(type(get_estimator()), 'version', 2):
                    revised = fit_game(game, rows, callback, directory, 'fixture-model', args={'central_interval': .9})
                    self.assertNotEqual(original_signature, rating_signature())
                self.assertNotEqual(original_signature, rating_signature(args={'central_interval': .9}))
                self.assertEqual(revised['central_interval'], .9)
            self.assertEqual(callback.call_count, count)
            self.assertEqual(initial['rating_fit']['evidence_key'], first['rating_fit']['evidence_key'])
            self.assertEqual(first['rating_fit']['evidence_key'], revised['rating_fit']['evidence_key'])
            self.assertEqual(len(list((Path(directory)/'player-rating').glob('rating-*.json'))), 1)

    def test_explicit_rating_output_is_exported_again_on_evidence_cache_hits(self):
        game, rows = game_records()
        callback = Mock(side_effect=evaluate)
        with tempfile.TemporaryDirectory() as directory, \
                patch('analysis.player_rating.figures.export_figures') as export:
            cache, output = Path(directory)/'cache', Path(directory)/'output'/'player-rating'
            first = fit_game(game, rows, callback, cache, 'fixture-model', output_dir=output)
            calls = callback.call_count
            second = fit_game(game, rows, callback, cache, 'fixture-model', output_dir=output)
            self.assertEqual(first, second)
            self.assertEqual(callback.call_count, calls)
            self.assertEqual(export.call_count, 2)
            self.assertTrue(all(call.args == (first, output) for call in export.call_args_list))
            export.reset_mock()
            fit_game(game, rows, callback, cache, 'fixture-model')
            export.assert_not_called()

    def test_evidence_requires_explicit_diagonal_metadata_and_removes_account_labels(self):
        game, rows = game_records()
        evidence = collect_evidence(game, rows, evaluate)
        for record in evidence.values():
            record.pop('schema_version')  # Compatible previous shared-evidence cache.
            record.update(actual_elo=600, reference_elo=2600, side='irrelevant', name='private')
        clean = validate_evidence(evidence)
        self.assertEqual(set(clean['White']), {'schema_version', 'conditioning', 'rating_grid', 'observations'})
        evidence['White'].pop('conditioning')
        with self.assertRaisesRegex(ValueError, 'Rerun full analysis'):
            fit_evidence(evidence)

    def test_invalid_outputs_are_rejected_before_application_serialization(self):
        game, rows = game_records()
        evidence = collect_evidence(game, rows, evaluate)
        source = FIXTURE_SOURCE.replace("return fixed_fit(evidence, central_interval=self.args['central_interval'])",
            "result = fixed_fit(evidence)\n"
            "        result['players']['White']['estimate'] = 2700\n"
            "        return result")
        with method_file('broken_fit', source), patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='broken_fit'):
            with self.assertRaisesRegex(ValueError, 'supported range'):
                fit_evidence(evidence)

    def test_declared_method_range_can_extend_beyond_measured_maia_grid(self):
        game, rows = game_records()
        evidence = collect_evidence(game, rows, evaluate)
        source = FIXTURE_SOURCE.replace("return fixed_fit(evidence, central_interval=self.args['central_interval'])",
            "result = fixed_fit(evidence)\n"
            "        result['rating_range'] = [0, 3200]\n"
            "        result['players']['White'].update(estimate=2800, interval=[2700, 2900])\n"
            "        return result")
        with method_file('extended_fit', source), patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='extended_fit'):
            self.assertEqual(fit_evidence(evidence)['players']['White']['estimate'], 2800)

    def test_interval_and_range_metadata_are_validated(self):
        game, rows = game_records()
        evidence = collect_evidence(game, rows, evaluate)
        for field, value, message in (
                ('central_interval', 'True', 'central_interval'),
                ('central_interval', '0', 'central_interval'),
                ('central_interval', "float('nan')", 'central_interval'),
                ('rating_range', '[2600, 600]', 'rating_range'),
                ('rating_range', "[0, float('inf')]", 'rating_range')):
            source = FIXTURE_SOURCE.replace("return fixed_fit(evidence, central_interval=self.args['central_interval'])",
                f"result = fixed_fit(evidence)\n        result[{field!r}] = {value}\n        return result")
            with self.subTest(field=field, value=value), method_file('invalid_metadata', source), \
                 patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='invalid_metadata'):
                with self.assertRaisesRegex(ValueError, message):
                    fit_evidence(evidence)

    def test_invalid_module_names_and_cache_path_traversal_fail(self):
        for method in ('../other', 'analysis.player_rating.bayesian_shared_curve', 'bayesian_shared_curve.py',
                       'Bayesian', 'bad-name', '', 3, False):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'filename'):
                get_estimator(method)
        with self.assertRaisesRegex(ValueError, 'cache key'):
            evidence_cache_path('cache', '../other')

    def test_base_class_and_incomplete_subclasses_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            PlayerRating()
        with method_file('abstract_fit', 'from analysis.player_rating.interface import PlayerRating\nclass Rating(PlayerRating): pass'):
            with self.assertRaisesRegex(ValueError, 'abstract fit'):
                get_estimator('abstract_fit')

    def test_module_must_define_its_own_subclass_and_valid_contract(self):
        invalid_modules = [
            ('no_class', 'def fit(evidence): return {}', 'Rating\\(PlayerRating\\)'),
            ('no_inheritance', 'class Rating:\n    def fit(self, evidence): return {}', 'Rating\\(PlayerRating\\)'),
            ('aliased_class', 'from analysis.player_rating.bayesian_shared_curve import Rating', 'own Rating'),
            ('bad_fit', 'from analysis.player_rating.interface import PlayerRating\nclass Rating(PlayerRating):\n    fit = None', 'synchronous fit'),
            ('extra_argument', 'from analysis.player_rating.interface import PlayerRating\nclass Rating(PlayerRating):\n    def fit(self, evidence, required): return {}', 'other required arguments'),
            ('async_fit', 'from analysis.player_rating.interface import PlayerRating\nclass Rating(PlayerRating):\n    async def fit(self, evidence): return {}', 'synchronous fit'),
            ('needs_init', FIXTURE_SOURCE.replace('def __init__(self, args=None):', 'def __init__(self, required):'), 'without arguments'),
            ('bad_name', FIXTURE_SOURCE.replace('class Rating(PlayerRating):', "class Rating(PlayerRating):\n    name = ''"), 'display name'),
            ('bad_version', FIXTURE_SOURCE.replace('class Rating(PlayerRating):', 'class Rating(PlayerRating):\n    version = False'), 'positive integer'),
            ('bad_id', FIXTURE_SOURCE.replace('class Rating(PlayerRating):', "class Rating(PlayerRating):\n    id = 'another_file'"), 'filename'),
            ('bad_parameters', FIXTURE_SOURCE.replace('return self.args', "return {'invalid': float('nan')}"), 'finite JSON mapping'),
        ]
        for method, source, error in invalid_modules:
            with self.subTest(method=method), method_file(method, source), self.assertRaisesRegex(ValueError, error):
                get_estimator(method)

    def test_missing_dependency_is_not_reported_as_missing_method(self):
        with method_file('dependency_error', 'import unavailable_rating_fixture_dependency'):
            with self.assertRaisesRegex(ValueError, "missing dependency 'unavailable_rating_fixture_dependency'"):
                get_estimator('dependency_error')

    def test_complete_legal_evidence_checked_before_inference(self):
        game, rows = game_records()
        rows[-1]['scores'].pop(next(iter(rows[-1]['scores'])))
        callback = Mock(side_effect=AssertionError('invalid evidence'))
        with self.assertRaisesRegex(ValueError, 'every legal move'):
            collect_evidence(game, rows, callback)
        callback.assert_not_called()

    def test_malformed_numeric_cache_is_rebuilt_on_full_run(self):
        game, rows = game_records()
        callback = Mock(side_effect=evaluate)
        with tempfile.TemporaryDirectory() as directory:
            first = fit_game(game, rows, callback, directory, 'fixture')
            calls = callback.call_count
            cache = evidence_cache_path(directory, first['rating_fit']['evidence_key'])
            cache.write_text('{incomplete', encoding='utf-8')
            rebuilt = fit_game(game, rows, callback, directory, 'fixture')
            self.assertGreater(callback.call_count, calls)
            self.assertEqual(rebuilt, first)
            self.assertEqual(validate_evidence(json.loads(cache.read_text(encoding='utf-8'))), json.loads(cache.read_text(encoding='utf-8')))
            incomplete = json.loads(cache.read_text(encoding='utf-8'))
            incomplete['White']['observations'].pop()
            cache.write_text(json.dumps(incomplete), encoding='utf-8')
            calls = callback.call_count
            rebuilt = fit_game(game, rows, callback, directory, 'fixture')
            self.assertGreater(callback.call_count, calls)
            self.assertEqual(rebuilt, first)

    def test_empty_and_forced_only_games_have_no_false_rating_precision(self):
        callback = Mock(side_effect=AssertionError('no inference needed'))
        empty = chess.pgn.Game()
        forced = chess.pgn.Game()
        forced.setup(chess.Board('7k/6Q1/8/8/8/8/8/K7 b - - 0 1'))
        forced.add_main_variation(chess.Move.from_uci('h8g7'))
        rows = [{'move': 'h8g7', 'position_score': -100, 'scores': {'h8g7': 0}}]
        with tempfile.TemporaryDirectory() as directory:
            for game, records in ((empty, []), (forced, rows)):
                fit = fit_game(game, records, callback, directory, 'fixture')
                for player in fit['players'].values():
                    self.assertIsNone(player['estimate'])
                    self.assertEqual(player['interval'], [0, 3200])
            self.assertEqual(fit['players']['White']['moves_used'], 0)
            self.assertEqual(fit['players']['Black']['moves_used'], 0)
            self.assertIsNone(fit['players']['Black']['average_accuracy'])

    def test_one_ply_identifies_only_the_side_that_played(self):
        game, rows = game_records('1. f3 *')
        row = rows[0]
        row['position_score'] = 200
        row['scores'] = {move: (-200 if move == row['move'] else 200) for move in row['scores']}
        def varying(board, own, other):
            legal = sorted(move.uci() for move in board.legal_moves)
            return [{'policy': {move: ((.8-.6*(rating-600)/2000) if move == row['move'] else
                                      (1-(.8-.6*(rating-600)/2000))/(len(legal)-1)) for move in legal}}
                    for rating in own]
        with tempfile.TemporaryDirectory() as directory:
            fit = fit_game(game, rows, varying, directory, 'fixture')
        self.assertIsNotNone(fit['players']['White']['estimate'])
        self.assertIsNone(fit['players']['Black']['estimate'])
        self.assertEqual(fit['players']['Black']['interval'], [0, 3200])

    def test_game_mismatch_rejected_even_when_cached_evidence_exists(self):
        first, rows = game_records('1. e4 *')
        other, _ = game_records('1. d4 *')
        with tempfile.TemporaryDirectory() as directory:
            fit_game(first, rows, evaluate, directory, 'fixture')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                fit_game(other, rows, evaluate, directory, 'fixture')


if __name__ == '__main__':
    unittest.main()
