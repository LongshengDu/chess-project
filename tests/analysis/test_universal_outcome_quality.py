"""Known-formula inversion checks, independent of commercial reference scores."""
import unittest

import numpy as np

from analysis.lichess_accuracy import move_accuracy
from tests.analysis.universal_outcome_quality import (
    PERFECT_ACCURACY_MAXIMUM_LOSS, ZERO_ACCURACY_MINIMUM_LOSS, outcome_quality,
)


class OutcomeQualityTests(unittest.TestCase):
    def test_inverts_every_uncensored_test_loss(self):
        losses = np.linspace(PERFECT_ACCURACY_MAXIMUM_LOSS+.0001,
                             ZERO_ACCURACY_MINIMUM_LOSS-.0001, 500)
        accuracy = [move_accuracy(100., 100.-loss) for loss in losses]
        np.testing.assert_allclose(100-outcome_quality(accuracy), losses, atol=1e-12, rtol=0)

    def test_clamped_endpoints_use_minimum_consistent_loss(self):
        self.assertEqual(float(outcome_quality(100.)), 100.)
        self.assertAlmostEqual(100-float(outcome_quality(0.)), ZERO_ACCURACY_MINIMUM_LOSS, places=12)
        self.assertEqual(move_accuracy(100., 100.-PERFECT_ACCURACY_MAXIMUM_LOSS/2), 100.)
        self.assertAlmostEqual(move_accuracy(100., 100.-ZERO_ACCURACY_MINIMUM_LOSS), 0., places=12)

    def test_transform_is_bounded_and_monotone(self):
        values = outcome_quality(np.linspace(0., 100., 1001))
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertGreaterEqual(values.min(), 0.)
        self.assertLessEqual(values.max(), 100.)

    def test_invalid_accuracies_are_rejected(self):
        for value in (-.1, 100.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                outcome_quality(value)


if __name__ == '__main__':
    unittest.main()
