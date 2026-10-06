"""Web uses canonical local game analysis, with atomic completion and cancellation."""
from contextlib import contextmanager, nullcontext
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from xml.etree import ElementTree

import chess
import chess.pgn

from analysis.game.pipeline import analyze_game
from analysis.profiler.runtime import RuntimeProfiler
from analysis.stockfish_search import BOUNDED_POLICY_VERSION
from backend.app import create_app
from backend.analysis_positions import PlatformAnalysis
from backend.settings import CONFIG as BACKEND_CONFIG
from tests.coach.fixtures import FakeEngines


class FullGameAnalysisTests(unittest.TestCase):
    def setUp(self):
        presets = patch.dict(BACKEND_CONFIG['FRONTEND'], STOCKFISH_TIME_SCALE_BY_DEPTH={12: .2, 15: .5, 18: 1.})
        presets.start()
        self.addCleanup(presets.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.dict(BACKEND_CONFIG['SERVER']['STORAGE'], OUTPUT_DIR=self.root/'output'))
        self.platform = PlatformAnalysis(Mock(), Mock(), self.root / 'games.sqlite3')
        self.client = create_app(self.platform, self.root).test_client()
        self.pgn = '[WhiteElo "1400"]\n[BlackElo "1500"]\n\n1. e4 e5 2. Nf3 Nc6 *'
        self.game_id = self.client.post('/api/platform/games', json={'pgn': self.pgn}).json['game_id']
        self.url = f'/api/platform/games/{self.game_id}'

    def fixture(self):
        return {'schema_version': 4, 'played_elo': {'white': {'estimate': 1400}},
                'positions': [{'fen': chess.STARTING_FEN, 'maia': {}, 'stockfish': {'cp_vec': {}}}],
                'moves': [], 'performance': {'method': 'lichess'}}

    def events(self, response):
        return [json.loads(line) for line in response.get_data(as_text=True).splitlines()]

    def test_web_and_direct_pipeline_have_the_same_ratings_moves_and_performance(self):
        class CompleteEngines(FakeEngines):
            def initial_analysis(engine, *args):
                result = super().initial_analysis(*args)
                result['search'].update(target_depth=18, budget_seconds=10., max_budget_seconds=30.,
                                        policy_version=BOUNDED_POLICY_VERSION)
                return result
        direct_engines, web_engines = CompleteEngines(), CompleteEngines()
        self.addCleanup(direct_engines._temp.cleanup)
        self.addCleanup(web_engines._temp.cleanup)
        expected = analyze_game(chess.pgn.read_game(io.StringIO(self.pgn)), direct_engines,
                                'white', 1400, progress=lambda _: None)
        with patch('backend.analysis_games.Engines.borrowed', return_value=nullcontext(web_engines)) as borrowed:
            response = self.client.post(self.url + '/analyze', json={'run_id': 'same-pipeline', 'target_depth': 18})
            events = self.events(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[-1]['type'], 'complete', events[-1])
        result = events[-1]['analysis']
        for field in ('moves', 'played_elo', 'performance', 'rating_fit', 'played_elo_method'):
            self.assertEqual(result[field], expected[field], field)
        self.assertEqual(len(result['positions']), 5)
        self.assertEqual(sorted(event['index'] for event in events if event['type'] == 'position'), list(range(5)))
        borrowed.assert_called_once()
        self.assertIs(borrowed.call_args.args[2], self.platform.maia)
        self.assertIs(borrowed.call_args.args[3], self.platform.stockfish)
        limits = borrowed.call_args.kwargs['limits']
        self.assertEqual((limits.depth, limits.verify_ms, limits.max_ms), (18, 10000, 30000))
        reopened = PlatformAnalysis(Mock(), Mock(), self.root / 'games.sqlite3')
        cached = create_app(reopened, self.root).test_client().get(self.url + '/analysis').json
        self.assertEqual(cached['analysis'], result)
        self.assertEqual(cached['positions'], result['positions'])
        output = self.platform.repository.rating_output_directory(self.game_id)
        self.assertEqual(output, self.root/'output'/f'{self.game_id}-full'/'player-rating')
        for stem in ('analysis', 'prior'):
            self.assertEqual(ElementTree.parse(output/f'{stem}.svg').getroot().tag,
                             '{http://www.w3.org/2000/svg}svg')
        self.assertEqual(json.loads((output/'fit.json').read_text(encoding='utf-8'))['rating_fit'], result['rating_fit'])

    def render_fixture(self, result, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('fit.json', 'analysis.svg', 'prior.svg', 'method-explanation.svg'):
            (directory/name).write_text('new ' + name, encoding='utf-8')

    def test_repeated_web_runs_replace_figures_and_remove_old_formats(self):
        output = self.platform.repository.rating_output_directory(self.game_id)
        output.mkdir(parents=True)
        (output/'analysis.png').write_text('previous', encoding='utf-8')
        (output/'posterior.svg').write_text('obsolete layout', encoding='utf-8')
        (output/'notes.txt').write_text('keep local notes', encoding='utf-8')
        with patch('backend.analysis_games.Engines.borrowed', return_value=nullcontext(Mock())), \
             patch('backend.analysis_games.analyze_game', side_effect=lambda *_a, **_k: self.fixture()), \
             patch('backend.analysis_games.export_saved_figures', side_effect=self.render_fixture) as export:
            for run_id in ('first-fit', 'same-cached-fit'):
                events = self.events(self.client.post(self.url+'/analyze', json={'run_id': run_id}))
                self.assertEqual(events[-1]['type'], 'complete', events[-1])
                self.assertEqual((output/'analysis.svg').read_text(encoding='utf-8'), 'new analysis.svg')
                self.assertEqual((output/'method-explanation.svg').read_text(encoding='utf-8'), 'new method-explanation.svg')
                self.assertFalse((output/'analysis.png').exists())
                (output/'analysis.svg').write_text('older renderer', encoding='utf-8')
        self.assertEqual(export.call_count, 2)
        self.assertFalse((output/'posterior.svg').exists())
        self.assertEqual((output/'notes.txt').read_text(encoding='utf-8'), 'keep local notes')
        self.assertEqual(list((self.root/'output').glob('.rating-*')), [])

    def test_cancellation_during_render_keeps_previous_figures_and_analysis(self):
        original = self.fixture()
        self.platform.repository.save_full_analysis(self.game_id, original)
        output = self.platform.repository.rating_output_directory(self.game_id)
        output.mkdir(parents=True)
        (output/'analysis.svg').write_text('accepted figure', encoding='utf-8')
        def cancel_render(*args):
            self.render_fixture(*args)
            self.assertTrue(self.platform.full_games.cancel(self.game_id, 'cancel-render'))
        with patch('backend.analysis_games.Engines.borrowed', return_value=nullcontext(Mock())), \
             patch('backend.analysis_games.analyze_game', return_value=self.fixture()), \
             patch('backend.analysis_games.export_saved_figures', side_effect=cancel_render):
            events = self.events(self.client.post(self.url+'/analyze', json={'run_id': 'cancel-render'}))
        self.assertEqual(events[-1]['type'], 'cancelled')
        self.assertEqual((output/'analysis.svg').read_text(encoding='utf-8'), 'accepted figure')
        self.assertEqual(self.platform.repository.load_full_analysis(self.game_id), original)

    def test_deletion_during_render_does_not_publish_an_output_directory(self):
        def delete_during_render(*args):
            self.render_fixture(*args)
            self.platform.repository.delete(self.game_id)
        with patch('backend.analysis_games.Engines.borrowed', return_value=nullcontext(Mock())), \
             patch('backend.analysis_games.analyze_game', return_value=self.fixture()), \
             patch('backend.analysis_games.export_saved_figures', side_effect=delete_during_render):
            events = self.events(self.client.post(self.url+'/analyze', json={'run_id': 'delete-render'}))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('deleted', events[-1]['message'])
        self.assertFalse(self.platform.repository.rating_output_directory(self.game_id).parent.exists())
        self.assertEqual(list((self.root/'output').iterdir()), [])

    def test_failed_figure_publication_restores_previous_files_and_database(self):
        original = self.fixture()
        self.platform.repository.save_full_analysis(self.game_id, original)
        output = self.platform.repository.rating_output_directory(self.game_id)
        output.mkdir(parents=True)
        for name in ('fit.json', 'analysis.svg', 'prior.svg', 'method-explanation.svg'):
            (output/name).write_text('accepted ' + name, encoding='utf-8')
        staged = self.root/'staged'
        self.render_fixture(None, staged)
        replace = Path.replace
        def fail_new_svg(path, target):
            if path == staged/'method-explanation.svg':
                raise OSError('Cannot publish figure')
            return replace(path, target)
        with patch.object(Path, 'replace', fail_new_svg), self.assertRaisesRegex(OSError, 'publish figure'):
            self.platform.repository.save_full_analysis(self.game_id, {**original, 'new': True}, rating_figures=staged)
        self.assertEqual(self.platform.repository.load_full_analysis(self.game_id), original)
        for name in ('fit.json', 'analysis.svg', 'prior.svg', 'method-explanation.svg'):
            self.assertEqual((output/name).read_text(encoding='utf-8'), 'accepted ' + name)

    def test_failed_run_preserves_complete_saved_analysis(self):
        original = self.fixture()
        self.platform.repository.save_full_analysis(self.game_id, original)
        with patch('backend.analysis_games.Engines.borrowed', return_value=nullcontext(Mock())), \
             patch('backend.analysis_games.analyze_game', side_effect=RuntimeError('engine unavailable')):
            events = self.events(self.client.post(self.url + '/analyze', json={'run_id': 'failed'}))
        self.assertEqual(events[-1], {'type': 'error', 'message': 'engine unavailable'})
        self.assertEqual(self.platform.repository.load_full_analysis(self.game_id), original)
        self.assertFalse(self.platform.full_games.runs)

    def test_cancelled_run_drains_worker_without_publishing(self):
        stopped = threading.Event()
        def blocked(game, engines, *, cancel, **kwargs):
            try:
                self.assertTrue(cancel.wait(timeout=5))
                raise InterruptedError('cancelled')
            finally:
                stopped.set()
        with patch('backend.analysis_games.Engines.borrowed', return_value=nullcontext(Mock())), \
             patch('backend.analysis_games.analyze_game', side_effect=blocked):
            response = self.client.post(self.url + '/analyze', json={'run_id': 'cancel-me'}, buffered=False)
            self.assertEqual(self.client.post(self.url + '/analyze/cancel', json={'run_id': 'cancel-me'}).json,
                             {'cancelled': True})
            events = self.events(response)
        self.assertTrue(stopped.is_set())
        self.assertEqual(events[-1]['type'], 'cancelled')
        self.assertIsNone(self.platform.repository.load_full_analysis(self.game_id))
        self.assertFalse(self.platform.full_games.runs)

    def test_disconnect_closes_borrowed_session_and_releases_job(self):
        closed = threading.Event()
        @contextmanager
        def borrowed(*args, **kwargs):
            try:
                yield Mock()
            finally:
                closed.set()
        def blocked(game, engines, *, cancel, **kwargs):
            self.assertTrue(cancel.wait(timeout=5))
            raise InterruptedError('disconnected')
        with patch('backend.analysis_games.Engines.borrowed', borrowed), \
             patch('backend.analysis_games.analyze_game', side_effect=blocked):
            response = self.client.post(self.url + '/analyze', json={'run_id': 'disconnect'}, buffered=False)
            response.close()
        self.assertTrue(closed.is_set())
        self.assertFalse(self.platform.full_games.runs)
        self.assertIsNone(self.platform.repository.load_full_analysis(self.game_id))

    def test_deletion_during_analysis_cannot_publish_an_orphan(self):
        def deleted(*args, **kwargs):
            self.platform.repository.delete(self.game_id)
            return self.fixture()
        with patch('backend.analysis_games.Engines.borrowed', return_value=nullcontext(Mock())), \
             patch('backend.analysis_games.analyze_game', side_effect=deleted):
            events = self.events(self.client.post(self.url + '/analyze', json={'run_id': 'deleted'}))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIsNone(self.platform.repository.load_full_analysis(self.game_id))
        self.assertEqual(self.platform.repository.load_analysis(self.game_id), [])

    def test_invalid_duplicate_and_cross_game_cancellation_requests(self):
        for data in ([], {}, {'run_id': 'bad', 'target_depth': 19}, {'run_id': 'bad', 'target_depth': True}):
            self.assertEqual(self.client.post(self.url + '/analyze', json=data).status_code, 400)
        self.assertEqual(self.client.post('/api/platform/games/missing/analyze', json={'run_id': 'missing'}).status_code, 404)
        run = self.platform.full_games.prepare(self.game_id, {'run_id': 'active'})
        try:
            self.assertEqual(self.client.post(self.url + '/analyze', json={'run_id': 'duplicate'}).status_code, 400)
            self.assertFalse(self.platform.full_games.cancel('other-game', 'active'))
            self.assertFalse(run.cancelled.is_set())
            self.assertTrue(self.platform.full_games.save_positions(self.game_id, [{'maia': {}}]))
            self.assertEqual(self.platform.repository.load_analysis(self.game_id), [])
        finally:
            self.platform.full_games.release(run)

    def test_full_result_and_visual_cache_publish_atomically_and_cascade(self):
        original = self.fixture()
        self.platform.repository.save_full_analysis(self.game_id, original)
        with self.assertRaises(KeyError):
            self.platform.repository.save_full_analysis(self.game_id, {'played_elo': {}})
        self.assertEqual(self.platform.repository.load_full_analysis(self.game_id), original)
        self.assertEqual(self.platform.repository.load_analysis(self.game_id), original['positions'])
        self.platform.repository.delete(self.game_id)
        self.assertIsNone(self.platform.repository.load_full_analysis(self.game_id))
        self.assertFalse(self.platform.repository.save_full_analysis(self.game_id, original))

    def test_terminal_zero_depth_survives_interactive_autosave(self):
        terminal = {'ply': 4, 'fen': chess.STARTING_FEN, 'maia': {}, 'stockfish': {
            'depth': 0, 'target_depth': 18, 'cp_vec': {}, 'mate_vec': {},
            'terminal_cp': 0, 'complete': True}}
        # A repetition draw can have legal moves; its full history established
        # the outcome, so validating terminality from FEN alone is insufficient.
        response = self.client.post(self.url + '/analysis', json=[terminal])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get(self.url + '/analysis').json['positions'], [terminal])
        del terminal['stockfish']['terminal_cp']
        self.assertEqual(self.client.post(self.url + '/analysis', json=[terminal]).status_code, 400)

    def test_unconsumed_stream_releases_its_own_profiler(self):
        profiler = self.platform.profiler = RuntimeProfiler(self.root / 'profiler')
        maia = SimpleNamespace(_engine=SimpleNamespace(cfg=SimpleNamespace(model='test', device='cpu')))
        stockfish = SimpleNamespace(hash_mb=128, analysis_pool=SimpleNamespace(
            workers=1, threads_per_worker=1, total_threads=1, hash_mb=128))
        with patch('torch.cuda.is_available', return_value=False):
            profiler_id = profiler.start({'game_id': self.game_id, 'total_positions': 5, 'target_depth': 18},
                                        maia, stockfish, 'bounded')['run_id']
        run = self.platform.full_games.prepare(self.game_id, {'run_id': 'not-consumed', 'profiler_id': profiler_id})
        self.platform.full_games.release(run)
        self.assertFalse(profiler.active)
        self.assertFalse(profiler.data['complete'])
        self.assertTrue(profiler.data['interrupted']['cancelled'])
        self.assertFalse(self.platform.full_games.runs)


if __name__ == '__main__':
    unittest.main()
