"""Reference-free invariants for the simple shared-curve research formulas."""
from copy import deepcopy
import unittest

import numpy as np

from tests.analysis import simple_curve_account as model


def fixture():
    records = {}
    for side, index in (('White', 0), ('Black', 1)):
        rows = [{'qualities': {'position': [99., 80., 40.], 'root': [99., 80., 40.]},
                 'played_index': index, 'weight': 1.,
                 'maia_probabilities': [[.1+.025*r, .6-.02*r, .3-.005*r] for r in range(21)]}
                for _ in range(10)]
        records[side] = {'observations': rows}
    return records


class SimpleCurveAccountTests(unittest.TestCase):
    def test_fixed_context_formulas_are_nondecreasing_over_full_accuracy_support(self):
        prepared = model.prepare(fixture(), {'White': 1600., 'Black': 1700.})
        values = [model.points_at(accuracy, prepared, 'White') for accuracy in np.linspace(0., 100., 201)]
        for method in model.METHODS:
            series = np.array([row[method] for row in values])
            self.assertTrue(np.isfinite(series).all(), method)
            self.assertTrue(np.all((series >= 0) & (series <= 3200)), method)
            self.assertGreaterEqual(np.diff(series).min(), -1e-8, method)

    def test_equal_accuracy_and_equal_actual_ratings_produce_a_tie(self):
        prepared = model.prepare(fixture(), {'White': 1600., 'Black': 1600.})
        self.assertEqual(model.points_at(90., prepared, 'White'), model.points_at(90., prepared, 'Black'))

    def test_metadata_and_input_order_do_not_change_inference(self):
        evidence = fixture()
        before = deepcopy(evidence)
        expected = model.predict(evidence, {'White': 1600., 'Black': 1700.})
        self.assertEqual(evidence, before)
        for record in evidence.values():
            record['reference'] = 9999
            record['WhiteEloEstimate'] = 0
        self.assertEqual(model.predict(evidence, {'White': 1600., 'Black': 1700.}), expected)
        swapped = {'White': evidence['Black'], 'Black': evidence['White']}
        observed = model.predict(swapped, {'White': 1700., 'Black': 1600.})
        for method in model.METHODS:
            self.assertEqual(expected[method]['White'], observed[method]['Black'])
            self.assertEqual(expected[method]['Black'], observed[method]['White'])

    def test_missing_observations_and_accounts_are_explicit(self):
        evidence = fixture()
        evidence['Black']['observations'] = []
        result = model.predict(evidence, {})
        self.assertTrue(all(pair['Black'] is None and pair['White'] is not None for pair in result.values()))
        evidence['White']['observations'] = []
        result = model.predict(evidence, {})
        self.assertTrue(all(pair == {'White': None, 'Black': None} for pair in result.values()))


if __name__ == '__main__':
    unittest.main()
