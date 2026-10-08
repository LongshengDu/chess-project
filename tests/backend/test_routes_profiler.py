"""HTTP profiler validation without engines or a listening server."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from backend.app import create_app
from backend.analysis_positions import PlatformAnalysis
from backend.settings import CONFIG as BACKEND_CONFIG
from engine.stockfish import StockfishScorer


class ProfilerRoutesTests(unittest.TestCase):
    def setUp(self):
        presets = patch.dict(BACKEND_CONFIG['FRONTEND'], STOCKFISH_TIME_SCALE_BY_DEPTH={12: .2, 15: .5, 18: 1.})
        presets.start()
        self.addCleanup(presets.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.dict(BACKEND_CONFIG['ANALYSIS'], CACHE_DIR=self.root/'evidence'))
        self.maia, self.stockfish, self.profiler = Mock(), Mock(), Mock()
        self.stockfish._engine = None
        self.profiler.active = False
        self.profiler.start.return_value = {'ok': True}
        self.platform = PlatformAnalysis(self.maia, self.stockfish, self.root / 'games.sqlite3', profiler=self.profiler)
        self.client = create_app(self.platform, self.root).test_client()

    def test_status_uses_profiler_namespace(self):
        self.profiler.status.return_value = {'enabled': True, 'active': False}
        response = self.client.get('/api/platform/profiler')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json, {'enabled': True, 'active': False})
        self.profiler.status.assert_called_once_with()
        self.assertEqual(self.client.get('/api/platform/profile').status_code, 404)

    def test_invalid_target_depth_is_rejected_before_engine_or_cache_changes(self):
        for data in ({'total_positions': 4}, {'total_positions': 4, 'target_depth': True},
                     {'total_positions': 4, 'target_depth': 0},
                     {'total_positions': 4, 'target_depth': '18'},
                     {'total_positions': 4, 'target_depth': 19}):
            with self.subTest(data=data):
                self.assertEqual(self.client.post('/api/platform/profiler/start', json=data).status_code, 400)
        self.profiler.start.assert_not_called()
        self.assertEqual(self.stockfish.mock_calls, [])

    def test_valid_start_clears_caches_and_preserves_profiler_options(self):
        self.platform.maia_cache['previous'] = {}
        self.platform.stockfish_cache['previous'] = '{}'
        response = self.client.post('/api/platform/profiler/start', json={'total_positions': 4, 'target_depth': 18})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.platform.maia_cache, {})
        self.assertEqual(self.platform.stockfish_cache, {})
        options, maia, stockfish, strategy = self.profiler.start.call_args.args
        self.assertEqual(options['target_depth'], 18)
        self.assertEqual(options['total_positions'], 4)
        self.assertEqual(options['position_budget_seconds'], 10.)
        self.assertEqual(options['max_search_seconds'], 30.)
        self.assertIs(maia, self.maia)
        self.assertIs(stockfish, self.stockfish)
        self.assertEqual(strategy, 'bounded')

    def test_invalid_progress_never_reaches_profiler_state(self):
        for endpoint in ('position', 'finish'):
            for data in ([], {}, {'run_id': 1}):
                with self.subTest(endpoint=endpoint, data=data):
                    self.assertEqual(self.client.post(f'/api/platform/profiler/{endpoint}', json=data).status_code, 400)
        for data in ({'run_id': 'test'}, {'run_id': 'test', 'ply': -1}, {'run_id': 'test', 'ply': True}):
            self.assertEqual(self.client.post('/api/platform/profiler/position', json=data).status_code, 400)
        self.profiler.position.assert_not_called()
        self.profiler.finish.assert_not_called()

    def test_duplicate_start_preserves_current_caches_and_engine_hashes(self):
        self.profiler.active = True
        self.platform.stockfish = Mock(spec=StockfishScorer)
        self.platform.stockfish._engine = Mock()
        self.platform.maia_cache['current'] = {'policy': {'e2e4': 1.}}
        self.platform.stockfish_cache['current'] = '{"cp_vec":{"e2e4":25}}'
        response = self.client.post('/api/platform/profiler/start', json={'total_positions': 4, 'target_depth': 18})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn('already running', response.json['error'])
        self.assertEqual(self.platform.maia_cache, {'current': {'policy': {'e2e4': 1.}}})
        self.assertEqual(self.platform.stockfish_cache, {'current': '{"cp_vec":{"e2e4":25}}'})
        self.platform.stockfish.analysis_pool.clear_hash.assert_not_called()
        self.platform.stockfish._engine.configure.assert_not_called()
        self.profiler.start.assert_not_called()
