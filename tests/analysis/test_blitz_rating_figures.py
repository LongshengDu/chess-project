"""Scatter comparisons retain missing observations without inventing estimates."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

from tests.analysis.blitz_rating_figures import build_figure, export


def fixture():
    methods = ['shared_curve_intersection', 'bayesian_shared_curve', 'uncertainty_weighted_shared_curve']
    data = {'games': 1, 'method_order': methods, 'method_names': dict(zip(methods, ['Intersection', 'Bayesian', 'Weighted'])),
            'players': [], 'summary': {}}
    for method in methods:
        for side, estimate in (('White', None if method == methods[0] else 2400.), ('Black', 1500.)):
            data['players'].append({'game': 'game0', 'number': 0, 'side': side, 'method': method,
                                    'reference': 2000. if side == 'White' else 1600., 'actual': 1400.,
                                    'estimate': estimate, 'display_estimate': estimate, 'average_accuracy': 90.,
                                    'edge': side == 'White', 'intersection_status': 'above_support' if estimate is None else 'native',
                                    'intersection_bound': 3200 if estimate is None else None})
        missing = int(method == methods[0])
        data['summary'][method] = {'estimated_players': 2-missing, 'missing_players': missing, 'mae': 100.,
                                   'maximum_error': 200., 'native_mae': 100., 'edge_mae': None if missing else 100.}
    return data


class BlitzRatingFigureTests(unittest.TestCase):
    def test_equal_square_axes_and_boundary_marker_are_explicit(self):
        data = fixture()
        original = deepcopy(data)
        figure = build_figure(data)
        try:
            self.assertEqual(len(figure.axes), 3)
            for axis in figure.axes:
                self.assertEqual(axis.get_xlim(), (0., 3200.))
                self.assertEqual(axis.get_ylim(), (0., 3200.))
                self.assertEqual(axis.get_box_aspect(), 1.)
            self.assertIn('1/2 estimates', figure.axes[0].get_title())
            self.assertIn('g0W*', [text.get_text() for text in figure.axes[0].texts])
            self.assertTrue(any('not Elo estimates' in text.get_text() for text in figure.texts))
            self.assertEqual(data, original)
        finally:
            figure.clear()

    def test_export_is_text_svg_only_and_replaces_previous_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = export(fixture(), Path(directory))
            svg = ElementTree.parse(path).getroot()
            self.assertEqual(svg.tag, '{http://www.w3.org/2000/svg}svg')
            self.assertIsNone(svg.find('.//{http://www.w3.org/2000/svg}image'))
            self.assertTrue(svg.findall('.//{http://www.w3.org/2000/svg}text'))
            path.write_text('stale', encoding='utf-8')
            self.assertEqual(export(fixture(), Path(directory)), path)
            self.assertEqual([entry.name for entry in Path(directory).iterdir()], [path.name])

    def test_flat_missing_crossing_is_listed_without_boundary_point(self):
        data = fixture()
        data['players'][0].update(intersection_status='flat', intersection_bound=None)
        figure = build_figure(data)
        try:
            self.assertTrue(any('No unique crossing' in text.get_text() and 'g0W' in text.get_text() for text in figure.texts))
            self.assertEqual(list(figure.axes[0].texts), [])
        finally:
            figure.clear()

    def test_invalid_counts_and_clipped_estimates_are_rejected(self):
        data = fixture()
        data['summary']['shared_curve_intersection']['estimated_players'] = 2
        with self.assertRaises(ValueError):
            build_figure(data)
        data = fixture()
        data['players'][1]['estimate'] = 3300.
        with self.assertRaises(ValueError):
            build_figure(data)


if __name__ == '__main__':
    unittest.main()
