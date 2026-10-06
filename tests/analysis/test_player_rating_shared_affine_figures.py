"""Single-game affine figures use saved moments and no population data."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from analysis.player_rating.figures import export_figures
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.scale import display_fit, from_native, native_jacobian
from analysis.player_rating.shared_curve_affine import Rating


def evidence():
    policy = [[.2*(1-p), .8*(1-p), p] for p in np.linspace(.25, .9, len(RATINGS))]
    return {side: {'conditioning': 'equal_opponent', 'rating_grid': list(RATINGS),
                   'actual_rating': actual,
                   'observations': [{'played_index': played, 'weight': 1.,
                                     'qualities': {'root': [20., 75., 100.], 'position': [20., 75., 100.]},
                                     'maia_probabilities': deepcopy(policy)} for _ in range(12)]}
            for side, actual, played in (('White', 1500., 2), ('Black', 1600., 1))}


class SharedAffineFigureTests(unittest.TestCase):
    def setUp(self):
        self.fit = Rating().fit(evidence())

    def capture(self, fit):
        pictures = []
        def record(figure, path, **kwargs):
            pictures.append([{'title': axis.get_title(), 'xlabel': axis.get_xlabel(),
                              'texts': [text.get_text() for text in axis.texts],
                              'labels': axis.get_legend_handles_labels()[1],
                              'lines': [(np.asarray(line.get_xdata()), np.asarray(line.get_ydata())) for line in axis.lines]}
                             for axis in figure.axes])
        with TemporaryDirectory() as directory, patch('matplotlib.figure.Figure.savefig', record):
            paths = export_figures(fit, directory)
        self.assertEqual(set(paths), {'fit', 'analysis_svg', 'prior_svg', 'method-explanation_svg'})
        return pictures

    def test_single_curve_and_its_moments_replace_population_explanations(self):
        original = deepcopy(self.fit)
        combined, prior, explanation = self.capture(self.fit)
        self.assertEqual(len(explanation), 4)
        self.assertEqual(combined[0]['labels'][0], 'Game shared curve C')
        np.testing.assert_array_equal(combined[0]['lines'][0][1], self.fit['diagnostics']['curve']['shared_accuracy'])
        captions = '\n'.join(axis['title']+'\n'+'\n'.join(axis['labels']+axis['texts'])
                             for picture in (combined, prior, explanation) for axis in picture)
        self.assertNotIn('Population', captions)
        self.assertNotIn('Blended', captions)
        self.assertIn('All curve and measurement moments come from this game.', captions)
        self.assertIn('Conditional measurement ±1σ', captions)
        self.assertIn('no uncertainty interval', combined[1]['title'])
        self.assertEqual(self.fit, original)

    def test_source_scale_maps_saved_native_points_and_prior_density(self):
        displayed = display_fit(self.fit, 'cr', {side: from_native(value, 'cr')
                               for side, value in (('White', 1500.), ('Black', 1600.))})
        combined, prior, explanation = self.capture(displayed)
        accuracy, points = explanation[3]['lines'][1]
        affine = self.fit['diagnostics']['affine']
        native = np.clip(affine['prior_mean']+affine['affine_slope']*(accuracy-affine['accuracy_mean']), 0., 3200.)
        np.testing.assert_allclose(points, from_native(native, 'cr'))
        density = np.asarray(self.fit['diagnostics']['curve']['prior_density'])/native_jacobian(
            self.fit['diagnostics']['curve']['fine_ratings'], 'cr')
        np.testing.assert_allclose(prior[0]['lines'][0][1], density/density.max())
        self.assertIn('Chess.com Rapid', combined[0]['xlabel'])
        self.assertIn('then converted', combined[1]['title'])

    def test_flat_game_has_no_affine_mapping_and_removes_stale_explanation(self):
        data = evidence()
        for side in data.values():
            for row in side['observations']:
                row['qualities'] = {'root': [100.]*3, 'position': [100.]*3}
        fit = Rating().fit(data)
        with TemporaryDirectory() as directory:
            stale = Path(directory)/'method-explanation.svg'
            stale.write_text('obsolete explanation', encoding='utf-8')
            paths = export_figures(fit, directory)
            self.assertNotIn('method-explanation_svg', paths)
            self.assertFalse(stale.exists())
            self.assertIn('White: no estimate', paths['analysis_svg'].read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
