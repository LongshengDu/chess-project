"""Benchmark-derived population assets cannot be loaded or used as estimators."""
from pathlib import Path
import unittest
from unittest.mock import patch

from analysis.player_rating import arithmetic_coverage
from analysis.player_rating.calibration import DEFAULT_PATH, WITHDRAWN_REASON, load_calibration
from analysis.player_rating.service import get_estimator


RETIRED = (arithmetic_coverage,)


class PopulationWithdrawalTests(unittest.TestCase):
    def test_removed_methods_have_no_module_and_no_compatibility_alias(self):
        directory = Path(arithmetic_coverage.__file__).parent
        for method in ('hierarchical_affine', 'uncertainty_ensemble'):
            with self.subTest(method=method):
                self.assertFalse((directory/(method+'.py')).exists())
                with self.assertRaisesRegex(ValueError, 'Unknown rating method'):
                    get_estimator(method)

    def test_benchmark_population_asset_is_absent(self):
        self.assertFalse(DEFAULT_PATH.exists(), 'The benchmark-derived population asset must remain deleted.')

    def test_loading_even_an_explicit_asset_path_rejects_before_file_access(self):
        with patch('builtins.open', side_effect=AssertionError('A withdrawn loader must not open files.')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('A withdrawn loader must not read bytes.')), \
             patch.object(Path, 'read_text', side_effect=AssertionError('A withdrawn loader must not read text.')):
            for value in (None, DEFAULT_PATH, Path('an-explicit-population.json'), object()):
                with self.subTest(path=value), self.assertRaises(ValueError) as raised:
                    load_calibration(value)
                self.assertEqual(str(raised.exception), WITHDRAWN_REASON)

    def test_filename_selection_and_direct_constructors_reject_withdrawn_methods(self):
        for module in RETIRED:
            with self.subTest(method=module.METHOD):
                with self.assertRaises(ValueError) as selected:
                    get_estimator(module.METHOD)
                self.assertEqual(str(selected.exception), WITHDRAWN_REASON)
                for kwargs in ({}, {'calibration': object()}):
                    with self.assertRaises(ValueError) as direct:
                        module.Rating(**kwargs)
                    self.assertEqual(str(direct.exception), WITHDRAWN_REASON)

    def test_direct_calculate_cannot_bypass_withdrawal_with_injected_records(self):
        with patch('builtins.open', side_effect=AssertionError('A withdrawn fit must not open files.')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('A withdrawn fit must not read bytes.')):
            for module in RETIRED:
                with self.subTest(method=module.METHOD), self.assertRaises(ValueError) as raised:
                    module.calculate(object(), object(), object())
                self.assertEqual(str(raised.exception), WITHDRAWN_REASON)


if __name__ == '__main__':
    unittest.main()
