"""Search failures remain distinguishable from watchdog-triggered process stops."""
import tempfile
import unittest
from unittest.mock import Mock, patch

import chess
import chess.engine

from analysis.engine_session import Engines, Limits


class RuntimeTests(unittest.TestCase):
    def test_initial_search_watchdog_uses_requested_budget_or_depth_policy_maximum(self):
        from tests.analysis.test_stockfish_search import SearchTests

        for strategy, seconds in (('bounded', 3.), ('staged', 8.), ('exhaustive', 8.)):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                engines = Engines(chess.STARTING_FEN, directory, stockfish_path='unused',
                                  limits=Limits(verify_ms=3000, max_ms=8000, depth=18, analysis_strategy=strategy))
                self.addCleanup(engines.close)
                engines.stockfish = SearchTests().engine()
                with patch('analysis.engine_session.SearchDeadline') as deadline:
                    result = engines.initial_analysis([], 'e2e4', [])
                self.assertEqual(deadline.call_args.args[1], seconds)
                self.assertEqual(result['search']['budget_seconds'], seconds)
                self.assertEqual(result['search']['max_budget_seconds'], 8.)

    def test_runtime_profiler_records_real_batch_and_search_events_with_mainline_indices(self):
        from tests.analysis.test_stockfish_search import SearchTests

        events = []
        with tempfile.TemporaryDirectory() as directory:
            engines = Engines(chess.STARTING_FEN, directory, stockfish_path='unused',
                              record=lambda kind, **data: events.append({'kind': kind, **data}))
            self.addCleanup(engines.close)
            engines.signature = {'fixture': True}
            engines.stockfish = SearchTests().engine(incomplete=True)
            engines.maia = Mock()
            def predict(fens, own, other, *, boards, timings):
                timings['forward_ms'] = .25
                return [{'policy': {move.uci(): 1/board.legal_moves.count() for move in board.legal_moves},
                         'value': .5} for board in boards]
            engines.maia.batch_evaluate.side_effect = predict
            a = chess.Board()
            b = a.copy()
            b.push_uci('e2e4')
            requests = [(board, [1500], [1500]) for board in (a, b)]
            first = engines.human_pair_batches(requests)
            self.assertEqual(engines.human_pair_batches(requests), first)
            scans = [engines.initial_analysis([], 'e2e4', []),
                     engines.initial_analysis(['e2e4'], 'e7e5', [])]
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
            phases = [event for event in events if event['kind'] == 'stockfish_search']
            self.assertEqual({event['ply'] for event in phases}, {0, 1})
            self.assertEqual(sum(e['status']=='started' for e in phases), sum(e['status']=='finished' for e in phases))

    def test_watchdog_stop_is_reported_as_timeout_for_both_search_paths(self):
        for initial in (False, True):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as directory:
                engines = Engines(chess.STARTING_FEN, directory, stockfish_path='unused')
                self.addCleanup(engines.close)
                process = engines.stockfish = Mock()
                process.analysis.side_effect = chess.engine.EngineTerminatedError('closed by deadline')
                timer = Mock()

                def deadline(seconds, callback):
                    timer.start.side_effect = callback
                    return timer

                with patch('analysis.stockfish_search.threading.Timer', side_effect=deadline):
                    with self.assertRaises(TimeoutError) as raised:
                        if initial:
                            engines.initial_analysis([], 'e2e4', [])
                        else:
                            engines.sf([], 50)
                self.assertIsInstance(raised.exception.__cause__, chess.engine.EngineTerminatedError)
                process.close.assert_called_once()
                timer.cancel.assert_called_once()
                timer.join.assert_called_once()

    def test_unrelated_engine_failure_is_not_misreported_as_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            engines = Engines(chess.STARTING_FEN, directory, stockfish_path='unused')
            self.addCleanup(engines.close)
            engines.stockfish = Mock()
            engines.stockfish.analysis.side_effect = chess.engine.EngineError('bad engine output')
            with self.assertRaisesRegex(chess.engine.EngineError, 'bad engine output'):
                engines.sf([], 50)


if __name__ == '__main__':
    unittest.main()
