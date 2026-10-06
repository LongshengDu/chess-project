"""Archived production baselines reject before loading deleted estimators."""
import unittest
from unittest.mock import patch

from tests.analysis.curve_production_baselines import prepare, predict, predict_prepared


class ProductionBaselineWithdrawalTests(unittest.TestCase):
    def test_all_entry_points_reject_without_file_access(self):
        with patch('builtins.open', side_effect=AssertionError('Unexpected asset access')):
            for function, arguments in ((prepare, ({},)), (predict, ({}, {})),
                                        (predict_prepared, ({}, {}))):
                with self.subTest(function=function.__name__):
                    with self.assertRaisesRegex(ValueError, 'Test games cannot be used'):
                        function(*arguments)


if __name__ == '__main__':
    unittest.main()
