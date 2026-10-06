"""Fixture-free invariants of the eight shared arithmetic-accuracy scales."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from analysis.player_rating.calibration import CalibrationCorpus, CalibrationCurve, CalibrationRecord
from tests.analysis import simple_accuracy_scale as scale


def synthetic_inputs():
    policies = [[.7-.02*k, .3+.02*k] for k in range(21)]
    qualities = ([30., 100.], [20., 95.], [40., 100.])
    evidence = {}
    for side, choices in (('White', (1, 1, 1)), ('Black', (1, 0, 1))):
        evidence[side] = {'observations': [
            {'played_index': played, 'qualities': {'position': list(q)},
             'maia_probabilities': deepcopy(policies), 'position_win_probability': p}
            for q, played, p in zip(qualities, choices, (.3, .55, .9), strict=True)]}
    curve1 = CalibrationCurve(tuple(72.+.7*k for k in range(21)), 9.)
    curve2 = CalibrationCurve(tuple(70.+.8*k for k in range(21)), 11.)
    corpus = CalibrationCorpus('fixture', (
        CalibrationRecord('a'*64, curve1, curve1), CalibrationRecord('b'*64, curve2, curve2)))
    return evidence, {'White': 1800., 'Black': 1500.}, corpus


class SimpleAccuracyScaleTests(unittest.TestCase):
    def test_descriptions_cover_every_method_and_return_independent_mapping(self):
        descriptions = scale.describe()
        self.assertEqual(set(descriptions), set(scale.METHODS))
        self.assertTrue(all(isinstance(text, str) and text.strip() for text in descriptions.values()))
        descriptions.clear()
        self.assertEqual(len(scale.describe()), 8)

    def test_every_raw_map_is_bounded_and_monotone_including_endpoints(self):
        accuracy = np.linspace(0., 100., 1001)
        local = 65.+.009*scale.RATING_GRID
        population = 70.+.007*scale.RATING_GRID
        results = scale.raw_scales(accuracy, local, population)
        for method, values in results.items():
            with self.subTest(method=method):
                self.assertEqual(values.shape, accuracy.shape)
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(np.all(np.diff(values) >= -1e-10))
                self.assertTrue(np.all((values >= 200.) & (values <= 3000.)))

    def test_inverse_plateau_uses_midpoint_without_favoring_an_endpoint(self):
        grid = scale.RATING_GRID
        plateau = np.where(grid < 1400., 80.+(grid-1400.)/100.,
                           np.where(grid > 1800., 80.+(grid-1800.)/100., 80.))
        self.assertEqual(float(scale._inverse(80., plateau)), 1600.)

    def test_prediction_is_pure_and_invariant_under_color_exchange(self):
        evidence, ratings, corpus = synthetic_inputs()
        original = deepcopy(evidence), dict(ratings), deepcopy(corpus)
        with patch.object(scale, 'load_calibration', side_effect=AssertionError('Unexpected filesystem read')):
            first = scale.predict(evidence, ratings, calibration=corpus)
            exchanged = deepcopy({'White': evidence['Black'], 'Black': evidence['White']})
            for record in exchanged.values():
                for row in record['observations']:
                    row['position_win_probability'] = 1-row['position_win_probability']
            second = scale.predict(exchanged, {'White': ratings['Black'], 'Black': ratings['White']}, calibration=corpus)
        self.assertEqual(evidence, original[0])
        self.assertEqual(ratings, original[1])
        self.assertEqual(corpus, original[2])
        for method in scale.METHODS:
            with self.subTest(method=method):
                self.assertAlmostEqual(first[method]['White'], second[method]['Black'], places=8)
                self.assertAlmostEqual(first[method]['Black'], second[method]['White'], places=8)
                self.assertGreaterEqual(first[method]['White'], first[method]['Black'])

    def test_account_perturbations_are_common_bounded_offsets(self):
        evidence, ratings, corpus = synthetic_inputs()
        baseline = scale.predict(evidence, ratings, calibration=corpus)
        for changed in ('White', 'Black', 'both'):
            for delta in (-200., 200.):
                shifted = {side: value+(delta if side == changed or changed == 'both' else 0.)
                           for side, value in ratings.items()}
                result = scale.predict(evidence, shifted, calibration=corpus)
                expected = delta*scale.ACCOUNT_WEIGHT/(1. if changed == 'both' else 2.)
                for method in scale.METHODS:
                    for side in scale.SIDES:
                        with self.subTest(changed=changed, delta=delta, method=method, side=side):
                            self.assertAlmostEqual(result[method][side]-baseline[method][side], expected, places=8)

    def test_unrelated_reference_metadata_has_no_effect(self):
        evidence, ratings, corpus = synthetic_inputs()
        baseline = scale.predict(evidence, ratings, calibration=corpus)
        changed = deepcopy(evidence)
        for record in changed.values():
            record.update(commercial_reference=-12345, player_name='ignored', account_rating=9999)
        self.assertEqual(scale.predict(changed, ratings, calibration=corpus), baseline)

    def test_invalid_accuracy_and_flat_population_slope_are_rejected(self):
        local = 65.+.009*scale.RATING_GRID
        population = 70.+.007*scale.RATING_GRID
        for invalid in (-1., 101., float('nan')):
            with self.subTest(accuracy=invalid), self.assertRaises(ValueError):
                scale.raw_scales(invalid, local, population)
        with self.assertRaises(ValueError):
            scale.raw_scales(80., local, np.full_like(population, 80.))


if __name__ == '__main__':
    unittest.main()
