"""Offline checks: production defaults and the isolated small-report runner."""
from coach.settings import CONFIG as COACH_CONFIG
import io
import json
import subprocess
import sys
import tempfile
import unittest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

import chess
import chess.pgn

from analysis.game.pipeline import analyze_game
from coach.coach import codex_model, parser
from tests.coach.fixtures import FakeAnalysisSession
from tests.coach.smoke_test import main

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[2] / 'config.yaml').read_text(encoding='utf-8-sig'))


class EntrypointTests(unittest.TestCase):
    def test_codex_configuration_reads_yaml_without_importing_engines(self):
        script = (
            'import sys, yaml\n'
            'from coach import agent_codex, agent_budget, agent_progress, settings\n'
            'raw = yaml.safe_load(settings.CONFIG_PATH.read_text(encoding="utf-8-sig"))\n'
            'assert agent_codex.CONFIG is agent_budget.CONFIG is agent_progress.CONFIG is settings.CONFIG\n'
            'assert agent_codex.CONFIG["COACH"]["CODEX"]["MODEL"] == raw["COACH"]["CODEX"]["MODEL"]\n'
            'assert agent_budget.RunBudget().max_tokens == raw["COACH"]["TOTAL_TOKEN_BUDGET"]\n'
            'assert agent_progress.CONFIG["COACH"]["PROGRESS_INTERVAL_SECONDS"] == raw["COACH"]["PROGRESS_INTERVAL_SECONDS"]\n'
            'assert not any(name == "engine" or name.startswith("engine.") for name in sys.modules)\n'
        )
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_coach_settings_are_independent_of_engine_settings(self):
        from coach import settings
        from engine import settings as engine_settings
        self.assertIsNot(settings.CONFIG, engine_settings.CONFIG)
        original = settings.CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER']
        with patch.dict(engine_settings.CONFIG['ANALYSIS'], STOCKFISH_THREADS_PER_WORKER=original + 1):
            self.assertEqual(settings.CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'], original)

    def test_production_has_no_test_mode_or_test_imports(self):
        import ast
        import inspect
        from coach.agent_runner import run_coach
        root = Path(__file__).resolve().parents[2] / 'coach'
        self.assertNotIn('smoke', inspect.signature(run_coach).parameters)
        self.assertFalse(any(root.glob('prompt-smoke.*')))
        for path in root.glob('*.py'):
            source = path.read_text(encoding='utf-8')
            self.assertNotIn('smoke', source.lower(), path.name)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn('test', (node.module or '').split('.'), path.name)
                    self.assertNotIn('tests', (node.module or '').split('.'), path.name)
        self.assertFalse((root / 'test').exists())
        self.assertTrue(Path(__file__).with_name('smoke_test.py').exists())

    def test_model_and_reasoning_defaults_and_no_production_smoke_option(self):
        from coach.agent_runner import run_coach
        from coach.agent_budget import RunBudget
        import inspect
        self.assertEqual(CONFIG['COACH']['MAX_MODEL_RESPONSES'], 24)
        self.assertEqual(RunBudget().max_model_responses, 24)
        self.assertEqual(inspect.signature(run_coach).parameters['max_model_responses'].default, 24)
        with patch.dict(COACH_CONFIG['COACH']['CODEX'], MODEL='gpt-6-luna'):
            args = parser().parse_args(['game.pgn', '--side', 'white', '--elo', '1400', '--rating-scale', 'lb'])
            self.assertEqual(codex_model(args), 'gpt-6-luna')
            self.assertEqual(args.max_model_responses, 24)
            override = parser().parse_args(['game.pgn', '--side', 'white', '--elo', '1400', '--rating-scale', 'lb', '--max-model-responses', '30'])
            self.assertEqual(override.max_model_responses, 30)
        self.assertFalse(hasattr(args, 'smoke_test'))
        self.assertNotIn('--smoke-test', parser().format_help())
        self.assertNotIn('--bootstrap-samples', parser().format_help())

    def test_smoke_copies_saved_analysis_calls_codex_once_and_closes_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'game' / 'analysis.json'
            source.parent.mkdir()
            seed_session = FakeAnalysisSession()
            self.addCleanup(seed_session._temp.cleanup)
            game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
            original = json.dumps(analyze_game(game, seed_session, actual_elo=1400, progress=lambda _: None))
            source.write_text(original, encoding='utf-8')
            session = Mock()
            session.__enter__ = Mock(return_value=session)
            session.__exit__ = Mock(return_value=False)
            def run(analysis, local, output, **kwargs):
                self.assertEqual(local, session)
                self.assertEqual(kwargs['model_id'], CONFIG['COACH']['CODEX']['MODEL'])
                self.assertEqual(kwargs['max_model_responses'], 1)
                self.assertFalse(kwargs['request'].allow_tools)
                self.assertEqual(kwargs['request'].report_name, 'coaching-smoke')
                self.assertEqual(kwargs['request'].max_attempts, 1)
                self.assertNotIn('smoke', kwargs)
                self.assertNotEqual(output, source.parent)
                self.assertTrue((output / 'analysis.json').is_file())
                analysis['agent_run'] = {'usage': {'total_tokens': 0}}
            with patch('tests.coach.smoke_test.AnalysisSession', return_value=session), \
                 patch('tests.coach.smoke_scenario.run_coach', side_effect=run) as coach:
                self.assertEqual(main([str(source), '--side', 'white', '--output-dir', str(Path(tmp)/'smoke'),
                                       '--cache-dir', str(Path(tmp)/'cache')]), 0)
            coach.assert_called_once()
            session.__exit__.assert_called_once()
            self.assertEqual(source.read_text(encoding='utf-8'), original)
            with patch('tests.coach.smoke_test.AnalysisSession') as factory:
                self.assertEqual(main([str(source), '--side', 'white', '--output-dir', str(source.parent)]), 1)
                factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
