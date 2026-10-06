"""Order and influence checks for continuous context/population combinations."""
from copy import deepcopy
import unittest

import numpy as np

from tests.analysis import edge_posterior_consensus as consensus
from tests.analysis.test_edge_rating import make_case


class EdgePosteriorConsensusTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.calibration = [make_case((0., 60., 99.), 7), make_case((20., 70., 100.), 6)]

    def predictions(self, case=None, calibration=None):
        case = self.case if case is None else case
        calibration = self.calibration if calibration is None else calibration
        return consensus.predict(case['evidence'], case['fit'], case['ratings'], calibration)

    def test_more_accuracy_never_reduces_estimate_in_fixed_context(self):
        # Keep the game curve, likelihood variance and population fixed. Varying
        # them simultaneously would test a different, unsupported ordering claim.
        changed = deepcopy(self.case)
        sequences = {}
        for accuracy in np.linspace(50., 100., 201):
            for side in ('White', 'Black'):
                changed['fit']['players'][side]['average_accuracy'] = float(accuracy)
            prediction = self.predictions(changed)
            for method, pair in prediction.items():
                for side, value in pair.items():
                    sequences.setdefault((method, side), []).append(value)
        for (method, side), values in sequences.items():
            self.assertTrue(np.isfinite(values).all(), msg=f'{method}/{side}')
            self.assertTrue(np.all(np.diff(values) >= -1e-9), msg=f'{method}/{side}')
            self.assertGreater(values[-1], values[0], msg=f'{method}/{side}')

    def test_no_switch_jump_at_measured_curve_boundaries(self):
        changed = deepcopy(self.case)
        knots = changed['fit']['diagnostics']['curve']['monotone_expected_accuracy']
        for boundary in (knots[0], knots[-1]):
            values = []
            for shift in (-1e-6, 0., 1e-6):
                changed['fit']['players']['White']['average_accuracy'] = boundary + shift
                values.append(self.predictions(changed))
            for method in values[0]:
                sequence = [item[method]['White'] for item in values]
                self.assertGreaterEqual(sequence[1] + 1e-9, sequence[0])
                self.assertGreaterEqual(sequence[2] + 1e-9, sequence[1])
                self.assertLess(sequence[-1] - sequence[0], .01)

    def test_account_contribution_is_exactly_five_percent(self):
        original = self.predictions()
        for side in ('White', 'Black'):
            other = 'Black' if side == 'White' else 'White'
            for shift in (-200., 200.):
                changed = deepcopy(self.case)
                changed['ratings'][side] += shift
                actual = self.predictions(changed)
                for method in ('consensus_point_all', 'consensus_posterior_all'):
                    self.assertEqual(actual[method], original[method])
                method = 'consensus_posterior_account_all'
                self.assertAlmostEqual(actual[method][side] - original[method][side], .05 * shift, places=10)
                self.assertEqual(actual[method][other], original[method][other])

    def test_inputs_are_immutable(self):
        before = deepcopy((self.case, self.calibration))
        self.predictions()
        self.assertEqual((self.case, self.calibration), before)

    def test_calibration_labels_and_evidence_are_ignored(self):
        changed = deepcopy(self.calibration)
        for case in changed:
            case['evidence'] = {'deliberately': 'unusable'}
            case['ratings'] = {'White': -9999, 'Black': 9999}
            case['reference'] = {'White': 0, 'Black': 3200}
            case['fit']['players'] = {'not': 'consulted'}
        self.assertEqual(self.predictions(), self.predictions(calibration=changed))

    def test_calibration_order_is_invariant(self):
        original = self.predictions()
        reordered = self.predictions(calibration=self.calibration[::-1])
        for method, pair in original.items():
            for side, value in pair.items():
                self.assertAlmostEqual(value, reordered[method][side], places=10)

    def test_side_swap_swaps_predictions(self):
        original = self.predictions()
        changed = deepcopy(self.case)
        for field in ('evidence', 'ratings'):
            changed[field] = {'White': self.case[field]['Black'], 'Black': self.case[field]['White']}
        changed['fit']['players'] = {'White': self.case['fit']['players']['Black'],
                                     'Black': self.case['fit']['players']['White']}
        swapped = self.predictions(changed)
        for method, pair in original.items():
            self.assertEqual(pair['White'], swapped[method]['Black'])
            self.assertEqual(pair['Black'], swapped[method]['White'])


if __name__ == '__main__':
    unittest.main()
