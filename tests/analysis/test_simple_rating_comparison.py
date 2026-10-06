"""Pair symmetries and honest displayed-score checks for the isolated experiment."""
from copy import deepcopy
import unittest

from tests.analysis import simple_pair_pooling as pooling
from tests.analysis.experiment_simple_rating import classify, matches, rounded
from tests.analysis.test_simple_curve_account import fixture


class SimpleRatingComparisonTests(unittest.TestCase):
    def test_reference_ties_and_decisive_signs_are_distinct_checks(self):
        self.assertFalse(matches(25, -49))
        self.assertTrue(matches(25, 51))
        self.assertTrue(matches(25, 1))
        self.assertTrue(matches(0, -49))
        self.assertFalse(matches(0, 50))
        self.assertTrue(matches(110, 1))
        self.assertFalse(matches(110, 0))
        self.assertNotEqual(classify(110), classify(1))
        self.assertEqual(rounded(1500.1), rounded(1500.4))

    def test_pooling_is_symmetric_and_does_not_mutate_evidence(self):
        evidence = fixture()
        original = deepcopy(evidence)
        actual = {'White': 1600., 'Black': 1700.}
        base = pooling.predict(evidence, actual)
        swapped = pooling.predict({'White': evidence['Black'], 'Black': evidence['White']},
                                  {'White': actual['Black'], 'Black': actual['White']})
        self.assertEqual(original, evidence)
        for name, pair in base.items():
            self.assertAlmostEqual(pair['White'], swapped[name]['Black'])
            self.assertAlmostEqual(pair['Black'], swapped[name]['White'])
            self.assertGreater(pair['White'], pair['Black'])

    def test_equal_inputs_tie_and_a_common_account_shift_does_not_change_contrast(self):
        evidence = fixture()
        evidence['Black'] = deepcopy(evidence['White'])
        for pair in pooling.predict(evidence, {'White': 1600., 'Black': 1700.}).values():
            self.assertEqual(pair['White'], pair['Black'])
        evidence = fixture()
        base = pooling.predict(evidence, {'White': 1600., 'Black': 1700.})
        changed = pooling.predict(evidence, {'White': 1800., 'Black': 1700.})
        for name, pair in base.items():
            for side in ('White', 'Black'):
                self.assertAlmostEqual(changed[name][side]-pair[side], 10.)


if __name__ == '__main__':
    unittest.main()
