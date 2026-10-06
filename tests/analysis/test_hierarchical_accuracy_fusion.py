"""Meaningful invariants of the research center/contrast point decision."""
from copy import deepcopy
import unittest

from tests.analysis.hierarchical_accuracy_fusion import preserve_arithmetic_contrast


class HierarchicalAccuracyFusionTests(unittest.TestCase):
    def test_retains_gap_and_uses_alternative_center_despite_reversed_alternative_order(self):
        arithmetic = {'White': 1800., 'Black': 1400.}
        alternative = {'White': 1300., 'Black': 1500.}
        result = preserve_arithmetic_contrast(arithmetic, alternative)
        self.assertEqual(result['players'], {'White': 1600., 'Black': 1200.})
        self.assertEqual(result['diagnostics']['common_shift'], -200.)
        self.assertEqual(sum(result['players'].values())/2, sum(alternative.values())/2)
        self.assertEqual(result['players']['White']-result['players']['Black'], 400.)

    def test_identity_when_centers_match_even_with_different_alternative_gap(self):
        arithmetic = {'White': 1830., 'Black': 1470.}
        result = preserve_arithmetic_contrast(arithmetic, {'White': 1700., 'Black': 1600.})
        self.assertEqual(result['players'], arithmetic)

    def test_color_symmetry_and_input_immutability(self):
        arithmetic = {'White': 2134.25, 'Black': 1742.5}
        alternative = {'White': 1960.5, 'Black': 1835.25}
        original = deepcopy((arithmetic, alternative))
        result = preserve_arithmetic_contrast(arithmetic, alternative)
        reverse = lambda values: {'White': values['Black'], 'Black': values['White']}
        swapped = preserve_arithmetic_contrast(reverse(arithmetic), reverse(alternative))
        self.assertEqual(swapped['players'], reverse(result['players']))
        self.assertEqual((arithmetic, alternative), original)

    def test_support_clipping_reduces_gap_without_order_reversal(self):
        result = preserve_arithmetic_contrast({'White': 2500., 'Black': 500.},
                                              {'White': 3100., 'Black': 2900.})
        self.assertEqual(result['players'], {'White': 3200., 'Black': 2000.})
        self.assertEqual(result['diagnostics']['clipped'], {'White': True, 'Black': False})
        self.assertEqual(result['diagnostics']['unclipped'], {'White': 4000., 'Black': 2000.})

    def test_complete_tie_stays_tied(self):
        result = preserve_arithmetic_contrast({'White': 1700., 'Black': 1700.},
                                              {'White': 1000., 'Black': 2000.})
        self.assertEqual(result['players'], {'White': 1500., 'Black': 1500.})

    def test_missing_side_explicitly_falls_back_without_inventing_pair(self):
        result = preserve_arithmetic_contrast({'White': 1500., 'Black': None},
                                              {'White': 1800., 'Black': 3400.})
        self.assertEqual(result['players'], {'White': 1800., 'Black': 3200.})
        self.assertFalse(result['diagnostics']['applied'])
        self.assertEqual(result['diagnostics']['clipped'], {'White': False, 'Black': True})

    def test_input_clipping_cannot_precede_the_center_contrast_operation(self):
        result = preserve_arithmetic_contrast({'White': 4000., 'Black': 3000.},
                                              {'White': 3500., 'Black': 3100.})
        self.assertEqual(result['diagnostics']['unclipped'], {'White': 3800., 'Black': 2800.})
        self.assertEqual(result['players'], {'White': 3200., 'Black': 2800.})
        self.assertEqual(result['diagnostics']['arithmetic_contrast'], 1000.)

    def test_rejects_bad_native_inputs(self):
        valid = {'White': 1500., 'Black': 1600.}
        for bad in (True, float('nan'), float('inf'), '1500'):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                preserve_arithmetic_contrast({'White': bad, 'Black': 1600.}, valid)
        with self.assertRaises(ValueError):
            preserve_arithmetic_contrast(valid, valid, (3200., 0.))


if __name__ == '__main__':
    unittest.main()
