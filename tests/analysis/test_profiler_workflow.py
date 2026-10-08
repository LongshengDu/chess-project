import json
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import chess

from analysis.profiler.runtime import RuntimeProfiler
from analysis.profiler.comparison import compare
from analysis.profiler.report import load_timings, main as report_main, shared_components, summarize


class RuntimeProfilerTests(unittest.TestCase):
    def test_completion_requires_every_position_and_checkpoints_survive(self):
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False):
            profiler = RuntimeProfiler(Path(directory))
            maia = SimpleNamespace(_engine=SimpleNamespace(cfg=SimpleNamespace(model='test',device='cpu')))
            sf = SimpleNamespace(hash_mb=128, analysis_pool=SimpleNamespace(
                workers=4, threads_per_worker=2, total_threads=8, hash_mb=128))
            run = profiler.start({'total_positions':2}, maia, sf, 'staged')
            with self.assertRaisesRegex(ValueError, 'already running'):
                profiler.start({'total_positions':2}, maia, sf, 'staged')
            profiler.record('stockfish_search', status='started', phase='engine_top')
            self.assertEqual(profiler.status()['current_search']['phase'], 'engine_top')
            profiler.position({**run, 'ply':0, 'position_ms':100})
            with self.assertRaisesRegex(ValueError, 'Not all positions'):
                profiler.finish(run)
            with self.assertRaisesRegex(ValueError, 'already recorded'):
                profiler.position({**run, 'ply':0})
            partial = json.loads((Path(directory) / 'runtime-profiler.json').read_text(encoding='utf-8'))
            self.assertFalse(partial['complete'])
            self.assertEqual(len(partial['positions']), 1)
            profiler.position({**run, 'ply':1, 'position_ms':200})
            profiler.finish({**run, 'browser_run_ms':300})
            saved = json.loads((Path(directory) / 'runtime-profiler.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['configuration']['stockfish_workers'],4)
            self.assertEqual(saved['configuration']['stockfish_threads_per_worker'],2)
            self.assertEqual(saved['configuration']['stockfish_total_threads'],8)
            self.assertEqual(saved['configuration']['stockfish_hash_mb_per_worker'],128)
            self.assertEqual(saved['configuration']['stockfish_total_hash_mb'],512)
            self.assertTrue(saved['complete'])
            self.assertEqual(saved['browser']['browser_run_ms'], 300)
            self.assertFalse(profiler.status()['active'])
            self.assertFalse((Path(directory) / 'runtime-profiler.tmp.json').exists())


class ProfilerLoaderTests(unittest.TestCase):
    def test_loads_runtime_timings(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            current = directory / 'runtime-profiler.json'
            current.write_text('{"source": "current"}', encoding='utf-8')
            self.assertEqual(load_timings(directory), ({'source': 'current'}, current))

    def test_missing_or_invalid_current_file_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with self.assertRaises(FileNotFoundError) as error:
                load_timings(directory)
            self.assertEqual(Path(error.exception.filename), directory / 'runtime-profiler.json')
            (directory / 'runtime-profiler.json').write_text('invalid', encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):
                load_timings(directory)


class SharedProfilerReportTests(unittest.TestCase):
    def timings(self):
        # Identical FENs deliberately exercise ply attribution, independent of
        # PGN validation performed by the report command.
        fen = chess.STARTING_FEN
        terminal = '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'
        positions, events = [], []
        for ply, wall in ((0, 700), (1, 900)):
            positions.append({'ply': ply, 'fen': fen, 'move': 'e2e4', 'label': f'{ply+1}. e4',
                              'complete': True, 'elapsed_ms': 1000+ply*100})
            search = {'kind': 'stockfish_search', 'ply': ply, 'fen': fen, 'phase': 'engine_top',
                      'target_depth': 18, 'multipv': 4, 'root_move': None}
            events += [{**search, 'status': 'started'},
                       {**search, 'status': 'finished', 'achieved_depth': 18, 'wall_ms': wall,
                        'cancelled': False, 'nodes': 1000},
                       {'kind': 'stockfish', 'ply': ply, 'fen': fen, 'cache_hit': False,
                        'terminal': False, 'complete': True, 'depth': 18, 'target_depth': 18,
                        'wall_ms': wall+100, 'acquire_ms': 10,
                        'options': {'maiaCandidateMoves': ['a2a3'], 'forcedCandidateMoves': ['e2e4']},
                        'root_move_depth_vec': {'e2e4': 18, 'a2a3': 18}}]
        positions.append({'ply': 2, 'fen': terminal, 'move': None, 'label': 'Final position',
                          'complete': True, 'elapsed_ms': 1101})
        events += [{'kind': 'stockfish', 'ply': 2, 'fen': terminal, 'cache_hit': False, 'terminal': True,
                    'complete': True, 'depth': 0, 'wall_ms': 1, 'acquire_ms': 0},
                   {'kind': 'maia_batch', 'positions': [{'ply': ply, 'fen': fen, 'cache_hit': False,
                                                        'rating_pairs': 21} for ply in (0, 1)],
                    'batch_size': 42, 'wall_ms': 200, 'inference_ms': 100, 'forward_ms': 50},
                   {'kind': 'accuracy_curve', 'wall_ms': 50},
                   {'kind': 'game_analysis_completed', 'wall_ms': 1150}]
        return {'complete': True, 'configuration': {'pipeline': 'shared', 'total_positions': 3,
                    'target_depth': 18, 'strategy': 'bounded', 'model': 'test', 'device': 'cpu',
                    'stockfish_workers': 2, 'stockfish_threads_per_worker': 1},
                'browser': {'browser_run_ms': 1200, 'save_ms': 5}, 'positions': positions, 'events': events}

    def test_parallel_worker_totals_and_maia_batch_timings_are_not_a_wall_partition(self):
        report = summarize(self.timings())
        self.assertFalse(report['components_additive'])
        self.assertAlmostEqual(report['components_seconds']['engine_top'], 1.6)
        self.assertGreater(report['components_seconds']['engine_top'], report['total_seconds'])
        self.assertAlmostEqual(report['components_seconds']['maia'], .2)
        self.assertAlmostEqual(report['maia_inference_seconds'], .1)
        self.assertAlmostEqual(report['maia_details_seconds']['forward'], .05)
        self.assertAlmostEqual(report['components_seconds']['accuracy_curve'], .05)
        self.assertIsNone(report['components_seconds']['other'])
        self.assertIsNone(report['between_positions_seconds'])
        self.assertAlmostEqual(report['positions'][0]['engine_top_seconds'], .7)
        self.assertAlmostEqual(report['positions'][1]['engine_top_seconds'], .9)
        for row in report['positions']:
            self.assertIsNone(row['maia_server_seconds'])
            self.assertIsNone(row['stockfish_queue_seconds'])
        self.assertEqual(report['positions'][2]['achieved_depth'], 0)
        self.assertTrue(report['positions'][2]['terminal'])
        self.assertIn('Accuracy curve', dict(shared_components(report)))

    def test_rejects_incomplete_run_missing_positions_and_shallow_candidates(self):
        data = self.timings()
        data['complete'] = False
        with self.assertRaisesRegex(ValueError, 'not complete'):
            summarize(data)
        data = self.timings()
        data['positions'].pop()
        with self.assertRaisesRegex(ValueError, 'every position'):
            summarize(data)
        data = self.timings()
        data['configuration']['strategy'] = 'staged'
        data['events'][2]['root_move_depth_vec']['a2a3'] = 10
        with self.assertRaisesRegex(ValueError, 'selected candidate'):
            summarize(data)

    def test_bounded_time_completion_retains_real_depth_and_rejects_cancellation(self):
        data = self.timings()
        data['events'][1].update(achieved_depth=14, stop_reason='time', time_limit_seconds=.7)
        data['events'][2].update(depth=14, target_reached=False, stop_reason='time',
                                root_move_depth_vec={'e2e4': 14, 'a2a3': 12})
        report = summarize(data)
        self.assertEqual(report['positions'][0]['achieved_depth'], 14)
        self.assertFalse(report['positions'][0]['target_reached'])
        self.assertEqual(report['searches'][0]['stop_reason'], 'time')
        data['events'][1]['cancelled'] = True
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            summarize(data)

    def test_missing_search_completion_and_wrong_ply_are_rejected(self):
        data = self.timings()
        data['events'].pop(1)
        with self.assertRaisesRegex(ValueError, 'completion event'):
            summarize(data)
        data = self.timings()
        data['events'][0]['ply'] = 2
        with self.assertRaisesRegex(ValueError, 'recorded game position'):
            summarize(data)
        data = self.timings()
        data['events'][-4]['terminal'] = False
        with self.assertRaisesRegex(ValueError, 'terminal marker'):
            summarize(data)

    def test_cached_result_does_not_replay_old_search_time(self):
        data = self.timings()
        data['events'] = [event for event in data['events']
                          if not (event['kind'] == 'stockfish_search' and event['ply'] == 0)]
        result = next(event for event in data['events'] if event['kind'] == 'stockfish' and event['ply'] == 0)
        result.update(cache_hit=True, wall_ms=2, phases=[{'wall_ms': 700}])
        report = summarize(data)
        self.assertTrue(report['positions'][0]['cache_hit'])
        self.assertEqual(report['positions'][0]['engine_top_seconds'], 0)
        self.assertAlmostEqual(report['positions'][0]['position_seconds'], .002)

    def test_cli_exports_terminal_only_timings_without_native_searches(self):
        data = self.timings()
        position = {**data['positions'][-1], 'ply': 0}
        completion = {**data['events'][-4], 'ply': 0}
        data.update(positions=[position], events=[completion, data['events'][-2], data['events'][-1]])
        data['configuration']['total_positions'] = 1
        raw_filename = 'runtime-profiler.json'
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            pgn = directory/'terminal.pgn'
            pgn.write_text(f'[SetUp "1"]\n[FEN "{position["fen"]}"]\n\n1/2-1/2\n', encoding='utf-8')
            (directory/raw_filename).write_text(json.dumps(data), encoding='utf-8')
            with patch('sys.argv', ['analysis.profiler.report', str(directory), '--pgn', str(pgn)]), patch('sys.stdout', io.StringIO()):
                report_main()
            for filename in ('REPORT.md', 'report.html', 'positions.csv', 'searches.csv', 'summary.json', 'timings.svg'):
                self.assertTrue((directory/filename).is_file(), filename)
            markdown = (directory/'REPORT.md').read_text(encoding='utf-8')
            self.assertIn('not additive elapsed time', markdown)
            self.assertNotIn('52 positions', markdown)
            self.assertIn('not a position duration', markdown)
            self.assertIn(f']({raw_filename})', markdown)
            self.assertIn(f'href="{raw_filename}"', (directory/'report.html').read_text(encoding='utf-8'))
            summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['searches'], [])
            self.assertIn('root_move', (directory/'searches.csv').read_text(encoding='utf-8-sig'))

    def test_comparison_reads_shared_stockfish_contract(self):
        data = self.timings()
        board = chess.Board()
        board.push_uci('e2e4')
        positions = [data['positions'][0], {**data['positions'][1], 'fen': board.fen(),
                                           'move': None, 'label': 'Final position'}]
        data.update(positions=positions, events=[])
        data['configuration']['total_positions'] = 2
        for position, best in zip(positions, ('e2e4', 'e7e5')):
            sf = {'complete': True, 'coverage_complete': True, 'depth': 18, 'target_depth': 18,
                  'best_move': best, 'cp_vec': {best: 15}, 'mate_vec': {},
                  'root_move_depth_vec': {best: 18}}
            data['events'].append({'kind': 'stockfish', 'ply': position['ply'], 'fen': position['fen'],
                                   'terminal': False, 'cache_hit': True, 'wall_ms': 2, **sf})
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ('baseline', 'optimized'):
                destination = directory / name
                destination.mkdir()
                (destination/'runtime-profiler.json').write_text(json.dumps(data), encoding='utf-8')
            result = compare(directory/'baseline', directory/'optimized')
        self.assertEqual(result['best_move_agreement'], 1)
        self.assertEqual(result['max_played_cp_delta'], 0)
        self.assertEqual(result['speedup'], 1)


if __name__ == '__main__':
    unittest.main()
