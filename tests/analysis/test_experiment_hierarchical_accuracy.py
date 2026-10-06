"""Offline measurement-runner provenance and unchanged-affine invariants."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


import numpy as np

from analysis.player_rating.calibration import evidence_fingerprint
from tests.analysis.experiment_hierarchical_accuracy import fit_measurement, metrics, population_evidence, write_tables
from tests.analysis.lichess_measurement import measure
from tests.analysis.test_player_rating_arithmetic_coverage import evidence_fixture


class AccuracyExperimentTests(unittest.TestCase):
    def test_population_context_erases_choice_and_account_but_keeps_policy_law(self):
        evidence = evidence_fixture()
        evidence['White']['actual_rating'] = 1000.
        evidence['Black']['actual_rating'] = 2000.
        original = deepcopy(evidence)
        numeric = population_evidence(evidence)
        self.assertEqual(evidence_fingerprint(numeric), evidence_fingerprint(evidence))
        for side in ('White', 'Black'):
            self.assertNotIn('actual_rating', numeric[side])
            self.assertTrue(all(row['played_index'] == 0 for row in numeric[side]['observations']))
        for method in ('arithmetic', 'lichess_plugin', 'lichess_reciprocal_plugin'):
            first, second = measure(evidence, method), measure(numeric, method)
            np.testing.assert_array_equal(first['curve']['shared_accuracy'], second['curve']['shared_accuracy'])
            self.assertEqual(first['curve']['likelihood'], second['curve']['likelihood'])
        self.assertEqual(evidence, original)


    def test_scoring_distinguishes_arithmetic_order_from_exact_reference_ties(self):
        rows = []
        for game, references, predictions in [('tie', (1500, 1500), (1480., 1520.)),
                                                ('decisive', (1525, 1500), (1490., 1530.))]:
            for side, reference, prediction, accuracy in zip(('White', 'Black'), references, predictions, (90., 89.), strict=True):
                rows.append({'game': game, 'side': side, 'reference': reference, 'estimate': prediction,
                             'arithmetic_accuracy': accuracy, 'edge': False, 'low_accuracy_quartile': True})
        result = metrics(rows)
        self.assertEqual(result['reference_order_matches'], 1)
        self.assertEqual(result['arithmetic_order_matches'], 0)
        self.assertEqual(result['low_accuracy_quartile']['players'], 4)
        self.assertEqual((result['over'], result['under']), (2, 2))

    def test_tables_mark_the_same_original_edge_side_in_every_method(self):
        rows = [{'game': 'game0', 'side': side, 'method': method, 'estimate': value,
                 'reference': 1500, 'edge': side == 'Black'}
                for method in ('arithmetic', 'volatility_weighted')
                for side, value in (('White', 1600.), ('Black', 1400.))]
        with TemporaryDirectory() as directory:
            write_tables(rows, directory)
            table = (Path(directory)/'comparison.md').read_text(encoding='utf-8')
            self.assertIn('| game0* | 1500 / 1500* | 1600 / 1400* | 1600 / 1400* |', table)
            self.assertIn('no intersection within the measured Maia 600–2600', table)


if __name__ == '__main__':
    unittest.main()
