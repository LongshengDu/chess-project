"""Scale changes re-express ratings without changing evidence or inference."""
from copy import deepcopy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
import chess.pgn

from analysis import elo_convert
from tests.analysis import compare_scaled_rating_methods as runner
from tests.analysis import scaled_rating_figures as figures
from tests.analysis.compare_intuitive_curves import metrics


def fixture():
    methods = ['shared_curve_intersection', 'bayesian_shared_curve', 'population_affine_account_prior']
    axis = list(range(0, 3201, 100))
    native = list(range(600, 2601, 100))
    context = {'game': 'game0', 'number': 0, 'scale': {'scale': 'cr', 'name': 'Chess.com Rapid'},
               'actual': {'White': 1600., 'Black': 1500.}, 'accuracy': {'White': 98., 'Black': 85.},
               'source_axis': [runner.convert_point(r, 'lb', 'cr') for r in axis],
               'source_native_axis': [runner.convert_point(r, 'lb', 'cr') for r in native],
               'shared_accuracy': [70.+r/160. for r in axis], 'knots': [70.+r/160. for r in native], 'variance': 4.}
    data = {'games': 1, 'method_order': methods, 'methods': runner.method_metadata(methods, {}),
            'source_scales': ['cr'], 'contexts': [context], 'players': [], 'summary_by_scale': {'cr': {}},
            'scale_invariance': runner.audit_summary([]), 'conversion_model': 'fixture'}
    for method in methods:
        for side, estimate in (('White', None if method == methods[0] else 2600.), ('Black', 1500.)):
            data['players'].append({'game': 'game0', 'number': 0, 'side': side, 'method': method,
                                    'source_scale': 'cr', 'reference': 2400. if side == 'White' else 1600.,
                                    'actual': 1600. if side == 'White' else 1500., 'estimate': estimate,
                                    'accuracy': 98. if side == 'White' else 85., 'edge': side == 'White',
                                    'conversion_extrapolated': method == methods[-1] and side == 'White'})
        data['summary_by_scale']['cr'][method] = metrics([row for row in data['players'] if row['method'] == method])
    return data


class ScaledRunnerTests(unittest.TestCase):
    def test_analysis_identity_uses_every_position_and_move_not_filename_or_headers(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 *'))
        board = game.board(); rows = []
        for move in game.mainline_moves():
            rows.append({'fen': board.fen(), 'played': {'move': move.uci()}})
            board.push(move)
        self.assertTrue(runner.analysis_matches_game({'moves': rows}, game))
        renamed = deepcopy(game); renamed.headers['White'] = 'Renamed player'
        self.assertTrue(runner.analysis_matches_game({'moves': rows}, renamed))
        wrong = deepcopy(rows); wrong[-1]['played']['move'] = 'b1c3'
        self.assertFalse(runner.analysis_matches_game({'moves': wrong}, game))
        wrong = deepcopy(rows); wrong[0]['fen'] = wrong[-1]['fen']
        self.assertFalse(runner.analysis_matches_game({'moves': wrong}, game))
        self.assertFalse(runner.analysis_matches_game({'moves': rows[:-1]}, game))

    def test_canonical_context_replaces_stale_accounts_without_rounding_or_mutation(self):
        evidence = {side: {'actual_rating': 999., 'observations': [
            {'qualities': [25., 100.], 'maia_probabilities': [[.4, .6]], 'position_win_probability': .7}]}
            for side in runner.SIDES}
        original = deepcopy(evidence)
        converted, actuals = runner.canonical_context(evidence, {'White': 1401., 'Black': 1551.}, 'cr')
        self.assertEqual(evidence, original)
        self.assertEqual(converted['White']['observations'], evidence['White']['observations'])
        self.assertEqual(actuals['White'], elo_convert.convert(1401., 'cr', 'lb', extrapolate=True))
        self.assertNotEqual(actuals['White'], round(actuals['White']))
        self.assertEqual(converted['White']['actual_rating'], actuals['White'])

    def test_missing_roots_remain_missing_and_extrapolation_is_explicit(self):
        self.assertIsNone(runner.convert_point(None, 'lb', 'cr'))
        value = runner.converted_metadata(3100., 'lb', 'cr')
        self.assertTrue(value['extrapolated'])
        self.assertAlmostEqual(runner.convert_point(value['value'], 'cr', 'lb'), 3100., places=8)
        self.assertFalse(runner.converted_metadata(2600., 'lb', 'cr')['extrapolated'])
        for value in (True, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                runner.convert_point(value, 'lb', 'cr')

    def test_all_four_scales_commute_for_endpoint_and_fractional_points(self):
        for scale in runner.SCALES:
            for rating in (0., 200., 600., 1543.712345, 2600., 3000., 3200.):
                with self.subTest(scale=scale, rating=rating):
                    target = runner.convert_point(rating, 'lb', scale)
                    self.assertAlmostEqual(runner.convert_point(target, scale, 'lb'), rating, places=7)

    def test_representation_audit_refits_each_coordinate_with_normalized_accounts(self):
        evidence = {side: {'observations': []} for side in runner.SIDES}
        actuals = {'White': 1832.123456, 'Black': 1742.654321}
        seen = []

        def predict(ratings):
            seen.append(dict(ratings))
            return {'fixture': {side: .2*ratings[side]+1400. for side in runner.SIDES}}

        baseline = {'shared_curve_intersection': {'White': None, 'Black': 1700.}}
        base = {**baseline, **predict(actuals)}
        with patch.object(runner, 'baseline_predictions', side_effect=lambda evidence: (deepcopy(baseline), {}, {}, {})):
            checks = runner.check_representations(evidence, actuals, [predict], base)
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(checks), 4*2*2)
        self.assertTrue(runner.audit_summary(checks)['passed'])
        self.assertEqual(sum(row['canonical_prediction_error'] is None for row in checks), 4)
        self.assertTrue(all(abs(row['White']-actuals['White']) < 1e-8 for row in seen))

    def test_direct_predictor_gets_updated_context_as_well_as_actual_argument(self):
        evidence = {side: {'actual_rating': 999., 'observations': []} for side in runner.SIDES}
        module = SimpleNamespace(predict=lambda context, ratings: {'mock': {
            side: context[side]['actual_rating']+ratings[side] for side in runner.SIDES}})
        predictors = runner.prepare_predictors([module], evidence, {'White': 1000., 'Black': 1200.})
        self.assertEqual(predictors[0]({'White': 1500., 'Black': 1700.})['mock'], {'White': 3000., 'Black': 3400.})
        self.assertEqual(evidence['White']['actual_rating'], 999.)

    def test_inference_roundoff_allowance_does_not_relax_conversion_accuracy(self):
        row = {'actual_roundtrip_error': 1e-12, 'canonical_prediction_error': 1e-6,
               'target_commutation_error': 1e-6, 'point_roundtrip_error': 1e-12}
        audit = runner.audit_summary([row])
        self.assertTrue(audit['passed'])
        self.assertFalse(audit['strict_1e_7_passed'])
        self.assertFalse(runner.audit_summary([{**row, 'actual_roundtrip_error': 1e-6}])['passed'])
        self.assertFalse(runner.audit_summary([{**row, 'canonical_prediction_error': 1e-4}])['passed'])

    def test_scope_assignment_keeps_population_only_controls_separate(self):
        self.assertEqual(runner.scope('hierarchical_affine_translated_prior'), 'target_curve')
        self.assertEqual(runner.scope('population_affine_account_prior'), 'population_control')
        self.assertEqual(runner.scope('simple_global_arithmetic_secant'), 'target_curve')
        self.assertEqual(runner.scope('simple_global_population_inverse'), 'population_control')


class ScaledFigureTests(unittest.TestCase):
    def test_all_methods_are_partitioned_once_independently_of_scores(self):
        data = fixture()
        partition = figures.groups(data)
        self.assertEqual([method for _, _, methods in partition for method in methods], data['method_order'])
        self.assertEqual(partition[-1][0], 'population_control')

    def test_scatter_truthfully_labels_source_scale_missing_roots_and_ring(self):
        data = fixture(); before = deepcopy(data)
        figure = figures.build_scatter(data, 'cr', data['method_order'], title='Fixture')
        try:
            self.assertTrue(any('Chess.com Rapid' in text.get_text() for text in figure.texts))
            subtitle = next(text for text in figure.texts if 'Chess.com Rapid' in text.get_text())
            for axis in figure.axes:
                self.assertLess(axis.title.get_window_extent().y1, subtitle.get_window_extent().y0)
            self.assertTrue(any('No crossing' in text.get_text() for text in figure.axes[0].texts))
            for axis in figure.axes:
                self.assertEqual(axis.get_xlim(), (0., 3200.))
                self.assertEqual(axis.get_ylim(), (0., 3200.))
                self.assertEqual(axis.get_box_aspect(), 1.)
            self.assertEqual(data, before)
        finally:
            figure.clear()

    def test_curve_uses_converted_coordinates_without_transforming_accuracy(self):
        context = fixture()['contexts'][0]
        figure = figures.build_curves([context])
        try:
            np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), context['source_axis'])
            np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), context['shared_accuracy'])
            self.assertTrue(any('vertical location is not an accuracy' in text.get_text() for text in figure.texts))
        finally:
            figure.clear()

    def test_export_includes_every_method_and_svg_only(self):
        data = fixture()
        with TemporaryDirectory() as directory:
            result = figures.export(data, Path(directory))
            text = Path(result['html']).read_text(encoding='utf-8')
            for method in data['method_order']:
                self.assertIn(figures.label(method), text)
            self.assertIn('Chess.com Rapid', text)
            self.assertIn('Population-only controls', text)
            self.assertIn('matching sign', text)
            for item in result['figures']:
                xml = ElementTree.parse(Path(directory)/item['file']).getroot()
                self.assertTrue(xml.findall('.//{http://www.w3.org/2000/svg}text'))
                self.assertIsNone(xml.find('.//{http://www.w3.org/2000/svg}image'))
            self.assertFalse(list(Path(directory).glob('*.png')))

    def test_duplicate_rows_and_wrong_counts_are_rejected(self):
        data = fixture()
        data['players'].append(deepcopy(data['players'][0]))
        with self.assertRaises(ValueError):
            figures.build_scatter(data, 'cr', data['method_order'], title='Fixture')


if __name__ == '__main__':
    unittest.main()
