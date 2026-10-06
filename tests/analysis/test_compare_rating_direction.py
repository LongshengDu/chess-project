"""Exact direction checks must not inherit the separate W/B tie tolerance."""
import unittest

from tests.analysis.compare_rating_direction import METHOD, compare


class RatingDirectionTests(unittest.TestCase):
    def test_tiny_opposite_changes_are_mismatches_and_rounding_is_separate(self):
        data = {'players': [
            {'game': 'game0', 'side': 'White', 'method': METHOD, 'actual': 1600,
             'reference': 1601, 'estimate': 1599, 'unrounded_estimate': 1599.4},
            {'game': 'game0', 'side': 'Black', 'method': METHOD, 'actual': 1600,
             'reference': 1601, 'estimate': 1600, 'unrounded_estimate': 1600.4},
            {'game': 'game1', 'side': 'White', 'method': METHOD, 'actual': 1600,
             'reference': 1600, 'estimate': 1600, 'unrounded_estimate': 1600.1}]}
        result = compare(data)
        self.assertEqual(result['matches'], 1)
        self.assertEqual(result['unrounded_matches'], 1)
        self.assertEqual(len(result['mismatches']), 2)
        self.assertEqual(len(result['rounding_direction_changes']), 2)
        self.assertEqual(result['confusion_matrix']['counts']['above']['below'], 1)
        self.assertEqual(result['confusion_matrix']['counts']['above']['equal'], 1)

    def test_missing_actual_or_duplicate_players_are_rejected(self):
        row = {'game': 'game0', 'side': 'White', 'method': METHOD, 'actual': None,
               'reference': 1700, 'estimate': 1600, 'unrounded_estimate': 1600.1}
        with self.assertRaisesRegex(ValueError, 'Actual Elo'):
            compare({'players': [row]})
        row['actual'] = 1600
        with self.assertRaisesRegex(ValueError, 'at most one'):
            compare({'players': [row, row]})

    def test_production_comparison_schema_uses_actual_headers_and_nested_raw_points(self):
        data = {'players': [{'game': 'game0', 'side': 'White', 'method_id': 'arithmetic_coverage',
                             'actual_rating': '1600', 'selected_rating': 1600, 'reference': 1601}],
                'games': [{'game': 'game0', 'players': {'White': {'unrounded_estimate': 1600.4}}}]}
        result = compare(data)
        self.assertEqual(result['method'], 'arithmetic_coverage')
        self.assertEqual(result['matches'], 0)
        self.assertEqual(result['unrounded_players'], 1)
        self.assertEqual(result['unrounded_matches'], 1)
        self.assertEqual(result['players_detail'][0]['actual'], 1600.)
        data.pop('games')
        result = compare(data)
        self.assertEqual(result['unrounded_players'], 0)
        self.assertIsNone(result['players_detail'][0]['unrounded_estimate'])


if __name__ == '__main__':
    unittest.main()
