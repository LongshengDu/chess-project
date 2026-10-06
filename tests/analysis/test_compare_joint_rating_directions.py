"""Acceptance-rule tests for retrospective joint-direction scoring."""
import unittest

from tests.analysis.compare_joint_rating_directions import canonical_rows, matches_pair, saved_rows, score


class JointDirectionTests(unittest.TestCase):
    def test_reference_ties_require_small_gap_and_non_ties_only_require_order(self):
        self.assertTrue(matches_pair(0, 49.999))
        self.assertFalse(matches_pair(0, 50))
        self.assertFalse(matches_pair(0, -50))
        self.assertTrue(matches_pair(25, .1))
        self.assertTrue(matches_pair(500, 10))
        self.assertTrue(matches_pair(-500, -10))
        self.assertFalse(matches_pair(25, -.1))
        self.assertFalse(matches_pair(25, 0))

    def test_account_direction_has_no_tolerance_and_rounding_is_explicit(self):
        rows = [{'game': 'game0', 'side': 'White', 'actual': 1600, 'reference': 1650, 'unrounded_estimate': 1600.1},
                {'game': 'game0', 'side': 'Black', 'actual': 1600, 'reference': 1500, 'unrounded_estimate': 1599.1}]
        rounded, raw = score(rows, rounded=True), score(rows, rounded=False)
        self.assertEqual(rounded['white_black_matches'], 1)
        self.assertEqual(rounded['account_direction_matches'], 1)
        self.assertFalse(rounded['all_joint_requirements'])
        self.assertTrue(raw['all_joint_requirements'])

    def test_archive_labels_must_agree_with_current_pgn(self):
        labels = {('game0', 'White'): {'reference': 1600, 'actual': 1500}}
        row = {'game': 'game0', 'side': 'White', 'method': 'example', 'estimate': 1600, 'actual': 1501}
        with self.assertRaises(ValueError):
            canonical_rows([row], labels)
        row['actual'] = 1500
        self.assertEqual(canonical_rows([row], labels)[0]['reference'], 1600)
        with self.assertRaises(ValueError):
            canonical_rows([row, row], labels)

    def test_prediction_adapter_never_uses_precomputed_ranking(self):
        data = {'scores': [{'method': 'example', 'ordering': 999}], 'games': [
            {'game': 'game0', 'predictions': {'example': {'White': 1600, 'Black': 1500}},
             'reference': {'White': 1550, 'Black': 1450}}]}
        rows = saved_rows(data)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['estimate'], 1600)
        self.assertEqual(rows[1]['reference'], 1450)


if __name__ == '__main__':
    unittest.main()
