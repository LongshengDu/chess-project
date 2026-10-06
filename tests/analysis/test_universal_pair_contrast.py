"""Paired location/contrast decision identities."""
import unittest

from tests.analysis.universal_pair_contrast import combine_pair, transform_centers


class PairContrastTests(unittest.TestCase):
    def setUp(self):
        self.fit = {'players': {'White': {'unrounded_estimate': 2100.},
                                'Black': {'unrounded_estimate': 1500.}}}
        self.new = {'White': 2000., 'Black': 1800.}

    def test_center_is_preserved_for_both_declared_contrasts(self):
        for weight in (1., .5):
            combined = combine_pair(self.new, self.fit, weight)
            self.assertEqual((combined['White']+combined['Black'])/2, 1900.)
        self.assertEqual(combine_pair(self.new, self.fit, 1.), {'White': 2200., 'Black': 1600.})
        self.assertEqual(combine_pair(self.new, self.fit, .5), {'White': 2100., 'Black': 1700.})

    def test_exchanging_players_only_exchanges_outputs(self):
        swapped_new = {'White': self.new['Black'], 'Black': self.new['White']}
        swapped_fit = {'players': {'White': self.fit['players']['Black'], 'Black': self.fit['players']['White']}}
        first = transform_centers({'test': self.new}, self.fit)
        second = transform_centers({'test': swapped_new}, swapped_fit)
        for name in first:
            self.assertEqual(first[name]['White'], second[name]['Black'])
            self.assertEqual(first[name]['Black'], second[name]['White'])

    def test_common_shift_of_new_context_changes_both_outputs_equally(self):
        shifted = {side: value+100 for side, value in self.new.items()}
        for weight in (1., .5):
            first = combine_pair(self.new, self.fit, weight)
            second = combine_pair(shifted, self.fit, weight)
            self.assertEqual(second['White']-first['White'], 100.)
            self.assertEqual(second['Black']-first['Black'], 100.)


if __name__ == '__main__':
    unittest.main()
