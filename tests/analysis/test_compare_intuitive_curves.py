"""Research scoring treats order, ties, missing estimates and unmeasured checks honestly."""
import unittest

from tests.analysis.compare_intuitive_curves import metrics


class ScoringTests(unittest.TestCase):
    def pair(self, white=1500., black=1490., refs=(1700., 1500.)):
        return [dict(game='example', side=side, estimate=value, reference=reference,
                     accuracy=accuracy, edge=False, own_rating_change=None, both_ratings_change=None)
                for side, value, reference, accuracy in zip(('White', 'Black'), (white, black), refs, (90., 89.))]

    def test_small_signed_gap_is_valid_for_nontied_reference(self):
        result = metrics(self.pair())
        self.assertEqual(result['reference_order_matches'], 1)
        self.assertEqual(result['curve_order_matches'], 1)
        self.assertIsNone(result['maximum_own_rating_change'])

    def test_tie_tolerance_is_strict_and_does_not_replace_accuracy_order(self):
        self.assertEqual(metrics(self.pair(1549., 1500., (1500., 1500.)))['reference_order_matches'], 1)
        self.assertEqual(metrics(self.pair(1550., 1500., (1500., 1500.)))['reference_order_matches'], 0)
        self.assertEqual(metrics(self.pair(1500., 1501., (1500., 1500.)))['curve_order_matches'], 0)

    def test_display_uses_project_half_up_rounding(self):
        result = metrics(self.pair(1500.5, 1500.49))
        self.assertEqual(result['display_order_matches'], 1)

    def test_unavailable_crossing_is_not_a_fabricated_error(self):
        result = metrics(self.pair(None, 1490.))
        self.assertEqual(result['players'], 1)
        self.assertEqual(result['mae'], 10.)
        self.assertEqual(result['comparable_games'], 0)
        empty = metrics([])
        self.assertIsNone(empty['mae'])


if __name__ == '__main__':
    unittest.main()
