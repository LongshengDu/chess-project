"""Search failures remain distinguishable from watchdog-triggered process stops."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chess
import chess.engine

from analysis.session import AnalysisSession, Limits


class RuntimeTests(unittest.TestCase):
    def test_fully_cached_owned_session_does_not_load_or_start_engines(self):
        from analysis.cache.requests import maia_request, stockfish_initial_request
        from engine.assets_identity import asset_identity
        from tests.analysis.test_stockfish_search import evaluation_frame

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'stockfish-fixture'
            executable.write_bytes(b'fixture executable')
            session = AnalysisSession(chess.STARTING_FEN, Path(directory) / 'cache', stockfish_path=executable)
            model = Mock(model_signature={'maia': 'fixture-model', 'history_window': 8, 'device': 'cpu'})
            signature = {**model.model_signature, 'stockfish': asset_identity(executable),
                         'threads': session.engines.threads_per_worker, 'hash_mb': session.engines.hash_mb}
            board = chess.Board()
            value = {'policy': {move.uci(): 1 / board.legal_moves.count() for move in board.legal_moves}, 'value': .5}
            session.cache.put(board, 'maia', maia_request(signature, 1600, 1600), value)
            request = stockfish_initial_request(signature, session.limits.depth,
                session.limits.verify_ms / 1000, session.limits.max_ms / 1000,
                session.limits.analysis_strategy, {'forcedCandidateMoves': ['e2e4']})
            session.cache.put(board, 'stockfish', request, evaluation_frame(board, depth=4))
            with patch('engine.runtime.start_stockfish') as start, patch('engine.maia.MaiaPolicy', return_value=model):
                with session:
                    self.assertEqual(session.human_pairs([], [1600], [1600]), [value])
                    self.assertTrue(session.initial_analysis([], 'e2e4', [])['search']['complete'])
                start.assert_not_called()
                model.load.assert_not_called()
                model.batch_evaluate.assert_not_called()

    def test_analysis_cache_is_owned_by_session_and_survives_native_resource_cleanup(self):
        from analysis.cache.positions import PositionCache
        from engine.runtime import EngineRuntime

        with tempfile.TemporaryDirectory() as directory:
            session = AnalysisSession(chess.STARTING_FEN, directory, stockfish_path='unused')
            self.assertIsInstance(session.cache, PositionCache)
            self.assertEqual(session.cache.directory, Path(directory))
            self.assertIsInstance(session.engines, EngineRuntime)
            self.assertFalse(hasattr(session.engines, 'cache'))
            self.assertFalse(hasattr(session.engines, 'analysis_cache'))
            self.assertFalse(hasattr(session, 'maia'))
            self.assertFalse(hasattr(session, 'stockfish'))
            session.cache.put(chess.Board(), 'fixture', {}, {'complete': True})
            process = session.engines.stockfish = Mock()
            session.close()
            process.quit.assert_called_once()
            self.assertEqual(session.cache.get(chess.Board(), 'fixture', {}), {'complete': True})

    def test_session_entry_does_not_start_engines_and_signature_failure_preserves_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'stockfish-fixture'
            executable.touch()
            session = AnalysisSession(chess.STARTING_FEN, Path(directory)/'analysis', stockfish_path=executable)
            session.cache.put(chess.Board(), 'fixture', {}, {'complete': True})
            process, model = Mock(), Mock()
            from unittest.mock import PropertyMock
            with patch('engine.runtime.start_stockfish', return_value=process), \
                 patch('engine.maia.MaiaPolicy', return_value=model):
                with patch.object(type(model), 'model_signature', new_callable=PropertyMock,
                                  create=True, side_effect=RuntimeError('fixture identity failed')):
                    with self.assertRaisesRegex(RuntimeError, 'fixture identity failed'):
                        session.__enter__()
            process.quit.assert_not_called()
            model.load.assert_not_called()
            self.assertIsNone(session.engines.stockfish)
            self.assertIsNone(session.engines.maia)
            self.assertEqual(session.cache.get(chess.Board(), 'fixture', {}), {'complete': True})

    def test_initial_search_watchdog_uses_requested_budget_or_depth_policy_maximum(self):
        from tests.analysis.test_stockfish_search import SearchTests

        for strategy, seconds in (('bounded', 3.), ('staged', 8.), ('exhaustive', 8.)):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                session = AnalysisSession(chess.STARTING_FEN, directory, stockfish_path='unused',
                                  limits=Limits(verify_ms=3000, max_ms=8000, depth=18, analysis_strategy=strategy))
                self.addCleanup(session.close)
                session.engines.stockfish = SearchTests().engine()
                with patch('analysis.session.SearchDeadline') as deadline:
                    result = session.initial_analysis([], 'e2e4', [])
                self.assertEqual(deadline.call_args.args[1], seconds)
                self.assertEqual(result['search']['budget_seconds'], seconds)
                self.assertEqual(result['search']['max_budget_seconds'], seconds)

    def test_runtime_profiler_records_real_batch_and_search_events_with_mainline_indices(self):
        from tests.analysis.test_stockfish_search import SearchTests

        events = []
        with tempfile.TemporaryDirectory() as directory:
            session = AnalysisSession(chess.STARTING_FEN, directory, stockfish_path='unused',
                              record=lambda kind, **data: events.append({'kind': kind, **data}))
            self.addCleanup(session.close)
            session.engines.signature = {'fixture': True}
            session.engines.stockfish = SearchTests().engine(incomplete=True)
            session.engines.maia = Mock()
            def predict(fens, own, other, *, boards, timings):
                timings['forward_ms'] = .25
                return [{'policy': {move.uci(): 1/board.legal_moves.count() for move in board.legal_moves},
                         'value': .5} for board in boards]
            session.engines.maia.batch_evaluate.side_effect = predict
            a = chess.Board()
            b = a.copy()
            b.push_uci('e2e4')
            requests = [(board, [1500], [1500]) for board in (a, b)]
            first = session.human_pair_batches(requests)
            self.assertEqual(session.human_pair_batches(requests), first)
            scans = [session.initial_analysis([], 'e2e4', []),
                     session.initial_analysis(['e2e4'], 'e7e5', [])]
            batches = [event for event in events if event['kind'] == 'maia_batch']
            self.assertEqual([p['ply'] for p in batches[0]['positions']], [0, 1])
            self.assertEqual(batches[0]['batch_size'], 2)
            self.assertEqual(batches[0]['forward_ms'], .25)
            self.assertTrue(all(p['cache_hit'] for p in batches[1]['positions']))
            self.assertEqual(batches[1]['batch_size'], 0)
            finished = [event for event in events if event['kind'] == 'stockfish']
            self.assertEqual([event['ply'] for event in finished], [0, 1])
            self.assertTrue(all(event['complete'] and event['wall_ms'] >= event['acquire_ms'] >= 0 for event in finished))
            self.assertEqual(finished[0]['root_move_depth_vec'],
                             {line['uci']: line['depth'] for line in scans[0]['lines']})
            for event, scan, board in zip(finished, scans, (a, b), strict=True):
                self.assertEqual(event['best_move'], scan['best_move'])
                self.assertEqual(event['cp_vec'], {line['uci']: line['cp'] for line in scan['lines']})
                self.assertEqual(set(event['cp_vec']), {move.uci() for move in board.legal_moves})
                self.assertEqual(event['mate_vec'], {})
                self.assertNotIn('lines', event)
            phases = [event for event in events if event['kind'] == 'stockfish_search']
            self.assertEqual({event['ply'] for event in phases}, {0, 1})
            self.assertEqual(sum(e['status']=='started' for e in phases), sum(e['status']=='finished' for e in phases))
            session.initial_analysis([], 'e2e4', [])
            self.assertTrue(events[-1]['cache_hit'])
            self.assertEqual(events[-1]['cp_vec'], finished[0]['cp_vec'])
            session.initial_analysis(['f2f3', 'e7e5', 'g2g4', 'd8h4'], None, [])
            self.assertTrue(events[-1]['terminal'])
            self.assertIsNone(events[-1]['best_move'])
            self.assertEqual(events[-1]['cp_vec'], {})
            self.assertEqual(events[-1]['mate_vec'], {'': 0})

    def test_watchdog_stop_is_reported_as_timeout_for_both_search_paths(self):
        for initial in (False, True):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as directory:
                session = AnalysisSession(chess.STARTING_FEN, directory, stockfish_path='unused')
                self.addCleanup(session.close)
                process = session.engines.stockfish = Mock()
                process.analysis.side_effect = chess.engine.EngineTerminatedError('closed by deadline')
                timer = Mock()

                def deadline(seconds, callback):
                    timer.start.side_effect = callback
                    return timer

                with patch('analysis.stockfish_search.threading.Timer', side_effect=deadline):
                    with self.assertRaises(TimeoutError) as raised:
                        if initial:
                            session.initial_analysis([], 'e2e4', [])
                        else:
                            session.sf([], 50)
                self.assertIsInstance(raised.exception.__cause__, chess.engine.EngineTerminatedError)
                process.close.assert_called_once()
                timer.cancel.assert_called_once()
                timer.join.assert_called_once()

    def test_unrelated_engine_failure_is_not_misreported_as_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            session = AnalysisSession(chess.STARTING_FEN, directory, stockfish_path='unused')
            self.addCleanup(session.close)
            session.engines.stockfish = Mock()
            session.engines.stockfish.analysis.side_effect = chess.engine.EngineError('bad engine output')
            with self.assertRaisesRegex(chess.engine.EngineError, 'bad engine output'):
                session.sf([], 50)


if __name__ == '__main__':
    unittest.main()
