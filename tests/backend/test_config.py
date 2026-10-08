"""Component-local YAML settings and consumers; no live engines or servers."""
import copy
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.raw = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8-sig'))

    def read_component(self, filename, **overrides):
        # Run the real component's YAML read, leaving the actual config file untouched.
        values = copy.deepcopy(self.raw)
        for path, value in overrides.items():
            keys = path.split('.')
            section = values
            for key in keys[:-1]:
                section = section[key]
            section[keys[-1]] = value
        source = yaml.safe_dump(values)
        read_text = Path.read_text
        def read(path, *args, **kwargs):
            return source if path == ROOT / 'config.yaml' else read_text(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read):
            return runpy.run_path(str(ROOT / filename))

    def test_engine_settings_read_yaml_and_resolve_model_paths(self):
        module = self.read_component('engine/settings.py', **{'MAIA.PLAYER_RATING': 1700, 'MAIA.MODEL': 'maia3-28m', 'MAIA.DEVICE': 'cpu', 'MAIA.TEMPERATURE': 0., 'MAIA.CHECKPOINT': 'models/custom.pt', 'MAIA.CACHE_EXECUTABLE': 'bin/maia-cache.exe'})
        config = module['CONFIG']
        self.assertEqual(config['MAIA']['CHECKPOINT'], ROOT / 'models' / 'custom.pt')
        self.assertEqual(config['MAIA']['CACHE_DIR'], ROOT / self.raw['MAIA']['CACHE_DIR'])
        self.assertEqual(config['MAIA']['CACHE_EXECUTABLE'], str(ROOT / 'bin' / 'maia-cache.exe'))
        for key, value in {'MODEL': 'maia3-28m', 'PLAYER_RATING': 1700, 'DEVICE': 'cpu', 'TEMPERATURE': 0.}.items():
            self.assertEqual(config['MAIA'][key], value)
        automatic = self.read_component('engine/settings.py', **{'MAIA.DEVICE': 'auto', 'MAIA.CHECKPOINT': None, 'MAIA.CACHE_EXECUTABLE': 'maia3-cache'})['CONFIG']['MAIA']
        self.assertEqual(automatic['CACHE_EXECUTABLE'], 'maia3-cache')
        self.assertEqual(automatic['DEVICE'], 'auto')
        self.assertIsNone(automatic['CHECKPOINT'])

    def test_exact_sections_and_analysis_resource_ownership(self):
        expected = {'MAIA','STOCKFISH','ANALYSIS','COACH','SERVER','FRONTEND'}
        self.assertEqual(set(self.raw),expected)
        self.assertEqual(set(self.raw['STOCKFISH']),
                         {'EXECUTABLE','CACHE_DIR','START_TIMEOUT_SECONDS','RELEASE_API','ASSETS'})
        for key in ('STOCKFISH_WORKERS','STOCKFISH_THREADS_PER_WORKER','STOCKFISH_HASH_MB_PER_WORKER'):
            self.assertIsInstance(self.raw['ANALYSIS'][key], int)
            self.assertGreater(self.raw['ANALYSIS'][key], 0)
        self.assertNotIn('PLAYER_RATING', self.raw['ANALYSIS'])
        exploration = self.raw['ANALYSIS']['STOCKFISH_EXPLORATION']
        self.assertEqual(set(exploration), {'MAX_DEPTH','DEFAULT_SEARCH_SECONDS','MAX_SEARCH_SECONDS'})
        self.assertGreater(exploration['DEFAULT_SEARCH_SECONDS'], 0)
        self.assertLessEqual(exploration['DEFAULT_SEARCH_SECONDS'], exploration['MAX_SEARCH_SECONDS'])
        self.assertEqual(set(self.raw['FRONTEND']),{'STATIC_DIR','BUILD','STOCKFISH_TIME_SCALE_BY_DEPTH'})
        self.assertEqual(self.raw['FRONTEND']['STATIC_DIR'], 'web/dist')
        self.assertEqual(self.raw['FRONTEND']['BUILD']['TOOLS_DIR'], 'web/.tools')
        self.assertFalse({'FRONTEND','STATIC_DIR','BUILD'} & self.raw['SERVER'].keys())
        self.assertEqual(self.raw['MAIA']['PLAYER_RATING'],1600)
        self.assertNotIn('ELO',self.raw['MAIA'])
        self.assertFalse({'WORKERS','THREADS','HASH_MB','STOCKFISH_TOTAL_HASH_MB','ELO','TOP_HUMAN_MOVES'} & self.raw['ANALYSIS'].keys())
        self.assertFalse(any(key.endswith(('OUTPUT_DIR','OUTPUT_PATH')) for key in self.raw['COACH']))
        self.assertNotIn('CACHE_DIR', self.raw['COACH'])
        for component in ('engine','backend','coach','analysis'):
            loaded = self.read_component(f'{component}/settings.py')['CONFIG']
            self.assertEqual(set(loaded),expected)
            expected_analysis = dict(self.raw['ANALYSIS'])
            if component != 'engine':
                expected_analysis['CACHE_DIR'] = ROOT / expected_analysis['CACHE_DIR']
            self.assertEqual(loaded['ANALYSIS'], expected_analysis)

    def test_flat_or_extra_root_settings_are_rejected(self):
        for source in ('MAIA_ELO: 1600', yaml.safe_dump({**self.raw,'OUTPUT_DIR':'unused'})):
            for filename in ('engine/settings.py','backend/settings.py','coach/settings.py','analysis/settings.py','web/settings.py'):
                with patch.object(Path,'read_text',return_value=source), self.assertRaisesRegex(ValueError,'six sections'):
                    runpy.run_path(str(ROOT/filename))

    def test_backend_and_engine_settings_are_independent_yaml_readers(self):
        from backend import settings as backend_settings
        from engine import settings as engine_settings
        self.assertIsNot(backend_settings.CONFIG, engine_settings.CONFIG)
        expected = engine_settings.CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER']
        with patch.dict(backend_settings.CONFIG['ANALYSIS'], STOCKFISH_THREADS_PER_WORKER=expected + 1):
            self.assertEqual(engine_settings.CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'], expected)
        for component in ('backend', 'engine', 'analysis'):
            with self.subTest(component=component):
                script = (
                    'import importlib, sys\n'
                    'module = importlib.import_module(sys.argv[1] + ".settings")\n'
                    'assert module.CONFIG_PATH.is_file()\n'
                    'assert not any(name.split(".")[0] in ({"backend", "engine", "coach", "analysis"} - {sys.argv[1]}) for name in sys.modules)\n'
                )
                result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script, component],
                                        cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_backend_game_outputs_resolve_outside_analysis_caches(self):
        config = self.read_component('backend/settings.py', **{
            'SERVER.STORAGE.OUTPUT_DIR': 'artifacts/saved-games'})['CONFIG']
        self.assertEqual(config['SERVER']['STORAGE']['OUTPUT_DIR'], ROOT/'artifacts'/'saved-games')
        self.assertEqual(self.raw['SERVER']['STORAGE']['OUTPUT_DIR'], 'backend/output')

    def test_build_tool_reads_yaml_without_importing_an_application_from_another_cwd(self):
        result = subprocess.run(
            [sys.executable, '-X', 'utf8', '-c',
             'import runpy, sys; from pathlib import Path; '
             'module = runpy.run_path(sys.argv[1]); '
             'from web import settings; '
             'assert settings.CONFIG_PATH == Path(sys.argv[1]).resolve().parents[1] / "config.yaml"; '
             'assert settings.CONFIG["FRONTEND"]["BUILD"]["TOOLS_DIR"].is_absolute(); '
             'assert not any(name.split(".")[0] in ("engine", "backend", "coach") for name in sys.modules)',
             str(ROOT / 'web/build.py')],
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_component_reads_new_yaml_values_and_ignores_environment(self):
        first = self.read_component('web/settings.py', **{'FRONTEND.BUILD.NODE_MIN_VERSION': 22})
        with patch.dict(os.environ, {'NODE_MIN_VERSION': '999', 'NODE': 'ignored-node'}):
            second = self.read_component('web/settings.py', **{'FRONTEND.BUILD.NODE_MIN_VERSION': 24})
        self.assertEqual(first['CONFIG']['FRONTEND']['BUILD']['NODE_MIN_VERSION'], 22)
        self.assertEqual(second['CONFIG']['FRONTEND']['BUILD']['NODE_MIN_VERSION'], 24)
        self.assertIsNot(first['CONFIG'], second['CONFIG'])

    def test_unsafe_yaml_is_rejected(self):
        with patch.object(Path, 'read_text', return_value='!!python/object:os.system {}'):
            with self.assertRaises(yaml.YAMLError):
                runpy.run_path(str(ROOT / 'web/settings.py'))

    def test_stockfish_resolution_preserves_override_cache_and_path_fallback(self):
        from engine.uci import stockfish_executable
        self.assertEqual(stockfish_executable(None, self.root), 'stockfish')
        (self.root / 'stockfish-unused.zip').write_bytes(b'fixture')
        (self.root / 'stockfish-folder').mkdir()
        self.assertEqual(stockfish_executable(None, self.root), 'stockfish')
        cached = self.root / 'stockfish-test.exe'
        cached.write_bytes(b'fixture')
        self.assertEqual(stockfish_executable(None, self.root), str(cached))
        self.assertEqual(stockfish_executable('custom-sf', self.root), 'custom-sf')

    def test_explicit_asset_overrides_do_not_download(self):
        from engine import assets, assets_maia, assets_stockfish
        checkpoint, executable = self.root / 'maia.pt', self.root / 'stockfish.exe'
        checkpoint.write_bytes(b'fixture')
        executable.write_bytes(b'fixture')
        module = self.read_component('engine/settings.py', **{'MAIA.CHECKPOINT': str(checkpoint), 'STOCKFISH.EXECUTABLE': str(executable)})
        with patch.dict(assets.CONFIG, module['CONFIG']), \
             patch.object(assets_stockfish, '_latest_stockfish_asset') as download, patch.object(assets_maia.subprocess, 'run') as run:
            self.assertEqual(assets.ensure_runtime_assets(), executable)
            download.assert_not_called()
            run.assert_not_called()

    def test_stockfish_assets_keep_yaml_structure(self):
        from engine import assets_stockfish
        with patch.object(assets_stockfish.platform, 'system', return_value='Windows'), \
             patch.object(assets_stockfish.platform, 'machine', return_value='AMD64'):
            filename, _, target = assets_stockfish._stockfish_target()
        self.assertEqual(filename, self.raw['STOCKFISH']['ASSETS']['windows']['x86-64'])
        self.assertEqual(target.parent, ROOT / self.raw['STOCKFISH']['CACHE_DIR'])

    def test_server_consumes_yaml_defaults_and_explicit_cli_overrides(self):
        from backend.app import main
        config = self.read_component('backend/settings.py', **{'SERVER.HOST': 'localhost', 'SERVER.PORT': 5143, 'ANALYSIS.PROFILER_DIR': 'runtime-timings', 'MAIA.DEVICE': 'cpu', 'FRONTEND.STATIC_DIR': str(self.root / 'web')})['CONFIG']
        self.assertEqual(config['ANALYSIS']['PROFILER_DIR'], ROOT / 'runtime-timings')
        config['FRONTEND']['STATIC_DIR'].mkdir(parents=True)
        (config['FRONTEND']['STATIC_DIR'] / 'index.html').write_text('fixture', encoding='utf-8')
        with patch.dict('backend.app.CONFIG', config), \
             patch('backend.app.ensure_runtime_assets', return_value=Path('fixture-sf')), \
             patch('backend.app.MaiaPolicy') as maia, patch('backend.app.StockfishScorer') as scorer, \
             patch('backend.app.RuntimeProfiler') as profiler, \
             patch('backend.app.PlatformAnalysis') as platform, patch('backend.app.create_app') as create:
            main([])
            create.return_value.run.assert_called_with(host='localhost', port=5143, threaded=True)
            self.assertEqual(platform.call_args.args[2], config['SERVER']['STORAGE']['DATABASE'])
            self.assertNotIn('time_scale', platform.call_args.kwargs)
            profiler.assert_called_once_with(ROOT / 'runtime-timings')
            self.assertIs(platform.call_args.kwargs['profiler'], profiler.return_value)
            self.assertEqual(maia.call_args.kwargs['device'], 'cpu')
            self.assertEqual(maia.call_args.args, (config['MAIA']['MODEL'], config['MAIA']['CACHE_DIR']))
            scorer.assert_called_with(Path('fixture-sf'),threads_per_worker=config['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'])
            create.assert_called_with(platform.return_value, config['FRONTEND']['STATIC_DIR'])
            scorer.return_value.close.assert_called_once()
            main(['--host', '127.0.0.1', '--port', '5150', '--analysis-strategy', 'staged',
                  '--stockfish-threads-per-worker','3', '--profiler-dir', str(self.root / 'runtime-timings')])
            create.return_value.run.assert_called_with(host='127.0.0.1', port=5150, threaded=True)
            self.assertEqual(platform.call_args.kwargs['strategy'], 'staged')
            profiler.assert_called_with(self.root / 'runtime-timings')
            self.assertIs(platform.call_args.kwargs['profiler'], profiler.return_value)
            scorer.assert_called_with(Path('fixture-sf'),threads_per_worker=3)

    def test_node_override_is_read_from_yaml_and_not_environment(self):
        from web import build_node
        config = self.read_component('web/settings.py', **{'FRONTEND.BUILD.NODE_EXECUTABLE': 'tools/node.exe'})['CONFIG']
        self.assertEqual(config['FRONTEND']['BUILD']['NODE_EXECUTABLE'], str(ROOT / 'tools' / 'node.exe'))
        config['FRONTEND']['BUILD']['NODE_EXECUTABLE'] = 'configured-node'
        runtime = build_node.NodeRuntime(config['FRONTEND']['BUILD'])
        with patch.dict(os.environ, {'NODE': 'ignored-node'}), patch.object(build_node.shutil, 'which', return_value=None), \
             patch.object(runtime, 'node_version', return_value=(22, 'v22.0.0')) as version:
            node, _ = runtime.find_node()
            self.assertEqual(node.name, 'configured-node')
            self.assertEqual(version.call_count, 1)


if __name__ == '__main__':
    unittest.main()
