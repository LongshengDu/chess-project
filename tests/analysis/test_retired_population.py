"""Withdrawn benchmark-population experiments fail before reading or writing."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.analysis.retired_population import REASON, reject_benchmark_population


ENTRY_POINTS = {
    'build_rating_calibration': ('build', 'main'),
    'compare_blitz_rating_methods': ('run',),
    'compare_intuitive_curves': ('run',),
    'compare_rating_methods': ('run',),
    'compare_scaled_rating_methods': ('run',),
    'experiment_edge_rating': ('run',),
    'experiment_hierarchical_accuracy': ('run',),
    'experiment_simple_rating': ('run',),
    'experiment_universal_rating': ('prepare_cases', 'run'),
    'intuitive_curve_diagnostics': ('run',),
    'joint_rating_monotone': ('run',),
    'render_hierarchical_paper': ('render',),
    'render_uncertainty_paper': ('render',),
    'simple_accuracy_scale': ('main',),
    'simple_rating_figures': ('main',),
    'universal_competitive_marginal': ('run',),
    'universal_competitive_sensitivity': ('run',),
    'universal_competitiveness': ('run',),
    'universal_context_similarity': ('main',),
    'universal_curve_uncertainty': ('run',),
    'universal_effective_sample': ('run',),
    'universal_hurdle_quality': ('run',),
    'universal_jackknife_consensus': ('run',),
    'universal_joint_accuracy': ('main',),
    'universal_joint_copula': ('main',),
    'universal_outcome_quality': ('main',),
    'universal_pair_contrast': ('run',),
    'universal_quality_alignment': ('main',),
    'universal_quantile_transport': ('main',),
    'universal_robust_consensus': ('run',),
}


class RetiredPopulationTests(unittest.TestCase):
    def test_guard_explains_withdrawal_and_does_not_access_files(self):
        with patch('builtins.open', side_effect=AssertionError('Unexpected file access')):
            with self.assertRaisesRegex(ValueError, 'Test games cannot be used as population/training inputs'):
                reject_benchmark_population()
        self.assertIn('withdrawn', REASON)

    def assert_guard_first(self, body):
        if (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        self.assertIsInstance(body[0], ast.ImportFrom)
        self.assertEqual(body[0].module, 'tests.analysis.retired_population')
        self.assertEqual(ast.unparse(body[1]), 'reject_benchmark_population()')
        # Execute only the entry actions: a protected body must be unreachable.
        prefix = ast.Module(body=body[:2], type_ignores=[])
        with patch('builtins.open', side_effect=AssertionError('Unexpected file access')):
            with self.assertRaisesRegex(ValueError, 'Test games cannot be used'):
                exec(compile(prefix, '<retired entry>', 'exec'), {})

    def test_every_population_runner_rejects_before_original_body(self):
        directory = Path(__file__).parent
        for name, functions in ENTRY_POINTS.items():
            tree = ast.parse((directory/(name+'.py')).read_text(encoding='utf-8-sig'))
            for function in functions:
                with self.subTest(module=name, function=function):
                    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
                    self.assert_guard_first(node.body)

    def test_historical_plots_only_cli_cannot_republish_withdrawn_results(self):
        path = Path(__file__).with_name('experiment_hierarchical_accuracy.py')
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and "__main__" in ast.unparse(node.test))
        self.assert_guard_first(main.body)


if __name__ == '__main__':
    unittest.main()
