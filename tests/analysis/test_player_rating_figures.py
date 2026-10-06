"""Production figure export uses the fitted distribution and saved schema."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np

from analysis.player_rating.figures import _accuracy_intersection, export_figures, export_saved_figures
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.service import fit_evidence, store_elo_fit
from analysis.player_rating.scale import display_fit, from_native, native_jacobian
from analysis.settings import CONFIG


def evidence():
    policy = [[1-p, p] for p in np.linspace(.2, .9, len(RATINGS))]
    row = {'played_index': 1, 'qualities': {'root': [0., 100.], 'position': [0., 100.]},
           'weight': 1., 'maia_probabilities': policy}
    return {side: {'conditioning': 'equal_opponent', 'rating_grid': list(RATINGS),
                   'observations': [row] * 4} for side in ('White', 'Black')}


class PlayerRatingFigureTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'))


    def arithmetic(self, white=1600, black=1500):
        records = evidence()
        for side, rating in (('White', white), ('Black', black)):
            records[side]['actual_rating'] = rating
            for row in records[side]['observations']:
                row['position_win_probability'] = .5
        with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='arithmetic_coverage'):
            return fit_evidence(records)

    def capture(self, fit):
        """Inspect actual plotted artists before the exporter releases its figures."""
        pictures = []
        def record(figure, path, **kwargs):
            pictures.append({'size': figure.get_size_inches().tolist(),
                             'axes': [{'ylim': axis.get_ylim(), 'position': axis.get_position().bounds,
                                       'xlim': axis.get_xlim(), 'xticks': axis.get_xticks().tolist(),
                                       'yticks': axis.get_yticks().tolist(), 'ylabel': axis.get_ylabel(),
                                       'xlabel': axis.get_xlabel(),
                                       'title': axis.get_title(),
                                       'yticklabels': [label.get_text() for label in axis.get_yticklabels()],
                                       'texts': [text.get_text() for text in axis.texts],
                                       'lines': [{'x': np.asarray(line.get_xdata()).tolist(),
                                                  'y': np.asarray(line.get_ydata()).tolist()}
                                                 for line in axis.lines],
                                       'labels': axis.get_legend_handles_labels()[1]}
                                      for axis in figure.axes]})
        with tempfile.TemporaryDirectory() as directory, patch('matplotlib.figure.Figure.savefig', record):
            export_figures(fit, directory)
        return pictures

    def test_saved_fit_exports_svg_only_and_removes_only_owned_legacy_images(self):
        fit = fit_evidence(evidence())
        saved = {}
        store_elo_fit(saved, fit)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('curve-and-prior.png', 'curve-and-prior.svg', 'posterior.png',
                         'posterior.svg', 'analysis.svg', 'prior.svg', 'chess.svg',
                         'analysis.png', 'prior.png'):
                (output/name).write_text('old image')
            paths = export_saved_figures(saved, directory, title='A synthetic game')
            self.assertEqual(len(paths), 3)
            exported = json.loads(paths['fit'].read_text(encoding='utf-8'))
            self.assertEqual(exported['diagnostics'], fit['diagnostics'])
            self.assertEqual(exported['central_interval'], .20)
            for stem in ('analysis', 'prior'):
                svg = ElementTree.parse(paths[f'{stem}_svg']).getroot()
                self.assertEqual(svg.tag, '{http://www.w3.org/2000/svg}svg')
                self.assertIsNone(svg.find('.//{http://www.w3.org/2000/svg}image'))
            self.assertEqual({p.name for p in output.iterdir()}, {'fit.json', 'analysis.svg', 'prior.svg', 'chess.svg'})
            self.assertEqual((output/'chess.svg').read_text(), 'old image')

    def test_accuracy_and_both_posteriors_share_one_horizontal_image(self):
        fit = fit_evidence(evidence())
        fit['players']['White']['average_accuracy'] = 55.
        fit['players']['Black']['average_accuracy'] = 76.
        pictures = self.capture(fit)
        combined, prior = pictures
        self.assertEqual(len(combined['axes']), 2)
        accuracy, posterior = combined['axes']
        self.assertEqual(accuracy['ylim'], (50., 100.))
        for picture in pictures:
            for axis in picture['axes']:
                self.assertEqual(axis['xlim'], (200., 3000.))
                np.testing.assert_array_equal(axis['xticks'], np.arange(200., 3001., 200.))
        self.assertAlmostEqual(accuracy['position'][1], posterior['position'][1])
        self.assertGreater(posterior['position'][0], accuracy['position'][0])
        labels = '\n'.join(posterior['labels'])
        self.assertIn('White:', labels)
        self.assertIn('Black:', labels)
        annotations = '\n'.join(accuracy['texts'])
        self.assertIn('White intersection: 1,600', annotations)
        self.assertIn('Black intersection: 2,200', annotations)
        self.assertNotEqual(fit['players']['White']['estimate'], 1600)
        self.assertEqual(len(prior['axes']), 1)
        self.assertEqual(prior['size'], [combined['size'][0]/2, combined['size'][1]])

    def test_prior_figure_shows_raw_weight_and_requested_axes_without_rescaling_posterior(self):
        fit = fit_evidence(evidence())
        original = deepcopy(fit)
        combined, prior = self.capture(fit)
        axis = prior['axes'][0]
        self.assertEqual(axis['title'], 'Rating prior before normalization')
        self.assertEqual(axis['ylabel'], 'Prior weight')
        self.assertEqual(axis['xlim'], (200., 3000.))
        self.assertEqual(axis['ylim'], (0., 1.))
        np.testing.assert_array_equal(axis['xticks'], np.arange(200., 3001., 200.))
        np.testing.assert_allclose(axis['yticks'], [0., .2, .4, .6, .8, 1.])
        line = axis['lines'][0]
        np.testing.assert_allclose(np.interp([200., 400., 600., 1600., 2600., 2800., 3000.], line['x'], line['y']),
                                   [0., 1/17, 16/17, 1., 16/17, 1/17, 0.])
        self.assertEqual(line['y'], fit['diagnostics']['curve']['prior_weights'])
        posterior_lines = combined['axes'][1]['lines']
        for side in ('White', 'Black'):
            self.assertTrue(any(item['y'] == fit['diagnostics']['curve']['posterior_densities'][side]
                                for item in posterior_lines))
        self.assertEqual(fit, original)

    def test_older_fit_prior_uses_explicit_relative_weights_without_inventing_raw_values(self):
        fit = fit_evidence(evidence())
        curve = fit['diagnostics']['curve']
        curve.pop('prior_weights', None)
        density = np.asarray(curve['prior_density'])
        original = deepcopy(fit)
        prior = self.capture(fit)[1]['axes'][0]
        self.assertEqual(prior['title'], 'Relative rating prior (peak = 1)')
        self.assertEqual(prior['ylabel'], 'Prior weight')
        self.assertEqual(prior['ylim'], (0., 1.))
        np.testing.assert_allclose(prior['lines'][0]['y'], density/density.max())
        self.assertEqual(fit, original)

    def test_accuracy_panel_states_saved_effective_sigma_scale_and_probability(self):
        fit = fit_evidence(evidence())
        curve = fit['diagnostics']['curve']
        curve['likelihood'] = {'accuracy_sigma': 1.234567, 'sigma_scale': .5}
        curve['top_probability'] = .95
        combined = self.capture(fit)[0]
        self.assertIn('Top 95% probability', combined['axes'][0]['title'])
        self.assertIn('σ = 1.235 accuracy points (scale ×0.5)', combined['axes'][0]['title'])
        self.assertNotIn('σ =', combined['axes'][1]['title'])
        self.assertEqual(combined['axes'][0]['ylim'], (50., 100.))

    def test_default_accuracy_panel_identifies_all_legal_moves(self):
        fit = fit_evidence(evidence())
        title = self.capture(fit)[0]['axes'][0]['title']
        self.assertIn('All legal moves', title)
        self.assertIn('scale ×1', title)
        self.assertNotIn('Top 100%', title)

    def test_historical_fit_omits_unknown_sigma_and_can_use_saved_probability_arguments(self):
        fit = fit_evidence(evidence())
        curve = fit['diagnostics']['curve']
        curve.pop('likelihood', None)
        curve.pop('top_probability', None)
        curve.pop('selection', None)
        fit['parameters'] = {'top_probability': .95}
        title = self.capture(fit)[0]['axes'][0]['title']
        self.assertIn('Top 95% probability', title)
        self.assertNotIn('σ =', title)
        self.assertNotIn('scale', title)
        fit.pop('parameters')
        title = self.capture(fit)[0]['axes'][0]['title']
        self.assertEqual(title, 'Shared accuracy · gray regions are extrapolated')

    def test_interrupted_image_render_preserves_existing_image_and_legacy_artifacts(self):
        fit = fit_evidence(evidence())
        def interrupted(figure, path, **kwargs):
            Path(path).write_bytes(b'incomplete image')
            raise OSError('Rendering interrupted')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            originals = {'analysis.svg': b'existing analysis', 'prior.svg': b'existing prior',
                         'analysis.png': b'legacy analysis', 'prior.png': b'legacy prior',
                         'posterior.svg': b'legacy posterior', 'chess.svg': b'user chess diagram'}
            for name, content in originals.items():
                (output/name).write_bytes(content)
            with patch('matplotlib.figure.Figure.savefig', interrupted):
                with self.assertRaisesRegex(OSError, 'Rendering interrupted'):
                    export_figures(fit, output)
            for name, content in originals.items():
                self.assertEqual((output/name).read_bytes(), content)
            self.assertEqual({path.name for path in output.iterdir()}, {*originals, 'fit.json'})

    def test_extrapolated_crossing_is_marked_and_missing_crossing_not_invented(self):
        fit = fit_evidence(evidence())
        curve = fit['diagnostics']['curve']
        upper = np.interp(3000., curve['fine_ratings'], curve['shared_accuracy'])
        measured_upper = curve['monotone_expected_accuracy'][-1]
        fit['players']['White']['average_accuracy'] = (upper+measured_upper)/2
        fit['players']['Black']['average_accuracy'] = 100.
        texts = '\n'.join(self.capture(fit)[0]['axes'][0]['texts'])
        self.assertIn('White intersection:', texts)
        self.assertIn('Extrapolated', texts)
        self.assertIn('Black: no intersection in 200–3000', texts)
        self.assertNotIn('Black intersection:', texts)

    def test_crossing_outside_display_is_not_annotated_as_visible(self):
        fit = fit_evidence(evidence())
        curve = fit['diagnostics']['curve']
        for side, rating in (('White', 100.), ('Black', 3100.)):
            fit['players'][side]['average_accuracy'] = float(np.interp(
                rating, curve['fine_ratings'], curve['shared_accuracy']))
        original = deepcopy(fit)
        texts = '\n'.join(self.capture(fit)[0]['axes'][0]['texts'])
        for side in ('White', 'Black'):
            self.assertIn(f'{side}: no intersection in 200–3000', texts)
            self.assertNotIn(f'{side} intersection:', texts)
        self.assertEqual(fit, original)

    def test_flat_intersection_is_a_range(self):
        ratings = np.array([0., 600., 1600., 2600., 3200.])
        accuracy = np.array([50., 60., 80., 80., 95.])
        self.assertEqual(_accuracy_intersection(ratings, accuracy, 80.), (1600., 2600.))
        self.assertEqual(_accuracy_intersection(ratings, accuracy, 70.), (1100., 1100.))
        self.assertIsNone(_accuracy_intersection(ratings, accuracy, 96.))

    def test_unobserved_player_does_not_create_a_fake_posterior(self):
        records = evidence()
        records['Black']['observations'] = []
        fit = fit_evidence(records)
        self.assertIsNone(fit['diagnostics']['curve']['posterior_densities']['Black'])
        posterior = self.capture(fit)[0]['axes'][1]
        self.assertIn('Black: insufficient information', '\n'.join(posterior['texts']))
        self.assertFalse(any(label.startswith('Black:') for label in posterior['labels']))

    def test_other_estimator_without_curve_exports_numeric_result_only(self):
        fit = {'players': {}, 'diagnostics': {}, 'method_id': 'another_estimator'}
        with tempfile.TemporaryDirectory() as directory:
            paths = export_figures(fit, directory)
            self.assertEqual(paths, {'fit': Path(directory)/'fit.json'})
            self.assertEqual(json.loads(paths['fit'].read_text(encoding='utf-8')), fit)





    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_arithmetic_points_distinguish_supplied_accounts_from_common_anchor(self):
        fit = self.arithmetic()
        original = deepcopy(fit)
        combined, prior = self.capture(fit)
        accuracy, points = combined['axes']
        self.assertIn('Arithmetic reference curve', accuracy['title'])
        self.assertIn('scale ×1', accuracy['title'])
        self.assertEqual(points['lines'], [])
        self.assertIn('no uncertainty interval', points['title'])
        self.assertNotIn('ordering', points['title'])
        self.assertNotIn('Posterior', points['title'])
        self.assertEqual(points['yticklabels'], ['Final estimate', 'Local curve mean',
                                                 'Population curve mean', 'Actual Elo supplied',
                                                 'Shared account anchor'])
        self.assertEqual(points['labels'], ['White', 'Black'])
        for row in fit['diagnostics']['components'].values():
            for key in ('unrounded_estimate', 'local_mean', 'population_mean',
                        'actual_rating', 'common_account_rating'):
                if row[key] is not None:
                    self.assertIn(f'{row[key]:,.0f}', points['texts'])
        self.assertEqual(points['texts'].count('1,550'), 2)
        self.assertEqual(prior['axes'][0]['title'], 'Component rating prior before normalization')
        self.assertEqual(fit, original)

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_arithmetic_missing_accounts_do_not_invent_anchor_points(self):
        fit = self.arithmetic(None, None)
        points = self.capture(fit)[0]['axes'][1]
        self.assertEqual(len(points['texts']), 6)  # Three available decisions per player.
        self.assertTrue(all(row['common_account_rating'] is None
                            for row in fit['diagnostics']['components'].values()))

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_saved_arithmetic_exports_components_and_replaces_prior_render(self):
        fit = self.arithmetic()
        saved = {}
        store_elo_fit(saved, fit)
        with tempfile.TemporaryDirectory() as directory:
            paths = export_saved_figures(saved, directory)
            self.assertEqual(set(paths), {'fit', 'analysis_svg', 'prior_svg'})
            text = paths['analysis_svg'].read_text(encoding='utf-8')
            self.assertIn('Local curve mean', text)
            self.assertIn('Shared account anchor', text)
            self.assertNotIn('Posterior density', text)
            self.assertNotIn('shaded central', text)
            paths['analysis_svg'].write_text('stale image', encoding='utf-8')
            export_saved_figures(saved, directory)
            self.assertEqual(paths['analysis_svg'].read_text(encoding='utf-8').count('Local curve mean'), 1)
            exported = json.loads(paths['fit'].read_text(encoding='utf-8'))
            self.assertIsNone(exported['players']['White']['interval'])
            self.assertIsNone(exported['central_interval'])

    def test_display_scale_pushes_curve_and_posterior_forward_without_refitting(self):
        native = fit_evidence(evidence())
        native['players']['White']['average_accuracy'] = 55.
        converted = display_fit(native, 'cr', {'White': None, 'Black': None})
        original = deepcopy(converted)
        combined, prior = self.capture(converted)
        curve = native['diagnostics']['curve']
        grid = np.asarray(curve['fine_ratings'])
        transformed = from_native(grid, 'cr')
        jacobian = native_jacobian(grid, 'cr')
        accuracy, posterior = combined['axes']
        np.testing.assert_allclose(accuracy['xlim'], from_native([200., 3000.], 'cr'))
        np.testing.assert_allclose(accuracy['lines'][0]['x'], transformed)
        np.testing.assert_allclose(accuracy['lines'][0]['y'], curve['shared_accuracy'])
        self.assertEqual(accuracy['xlabel'], 'Rating (Chess.com Rapid)')
        self.assertIn(f'White intersection: {from_native(1600., "cr"):,.0f}', '\n'.join(accuracy['texts']))
        native_density = np.asarray(curve['posterior_densities']['White'])
        pushed = native_density/jacobian
        self.assertTrue(any(np.allclose(line['y'], pushed) for line in posterior['lines'] if len(line['y']) == len(pushed)))
        self.assertAlmostEqual(float(np.trapezoid(pushed, transformed)), 1., places=5)
        expected_prior = np.asarray(curve['prior_density'])/jacobian
        np.testing.assert_allclose(prior['axes'][0]['lines'][0]['y'], expected_prior/expected_prior.max())
        self.assertEqual(prior['axes'][0]['title'], 'Converted rating prior density (peak = 1)')
        self.assertEqual(converted, original)

    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_component_points_and_saved_metadata_use_display_scale_once(self):
        native = self.arithmetic()
        converted = display_fit(native, 'cr', {'White': from_native(1600., 'cr'), 'Black': from_native(1500., 'cr')})
        points = self.capture(converted)[0]['axes'][1]
        component = native['diagnostics']['components']['White']
        for key in ('unrounded_estimate', 'local_mean', 'population_mean', 'actual_rating', 'common_account_rating'):
            self.assertIn(f'{from_native(component[key], "cr"):,.0f}', points['texts'])
        saved = {}
        store_elo_fit(saved, converted)
        with tempfile.TemporaryDirectory() as directory:
            paths = export_saved_figures(saved, directory)
            exported = json.loads(paths['fit'].read_text(encoding='utf-8'))
            self.assertEqual(exported['rating_scale'], converted['rating_scale'])
            self.assertEqual(exported['canonical_rating_range'], converted['canonical_rating_range'])
            self.assertIn('Chess.com Rapid', paths['analysis_svg'].read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
