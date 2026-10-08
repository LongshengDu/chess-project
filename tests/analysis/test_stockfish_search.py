import unittest
from unittest.mock import Mock, patch

import chess
import chess.engine

from analysis.stockfish_search import SearchControl, resolve_search_seconds, stream_evaluations
from analysis.settings import CONFIG


def evaluation_frame(board=None, *, depth=18, strategy='bounded', cp=20):
    """Complete legal-move evidence for cache/transport tests using fake engines."""
    board = chess.Board() if board is None else board
    legal = [move.uci() for move in board.legal_moves]
    return {'complete': True, 'coverage_complete': True, 'depth': depth,
            'strategy': strategy, 'is_checkmate': board.is_checkmate(),
            'best_move': legal[0], 'engine_moves': legal[:4],
            'cp_vec': dict.fromkeys(legal, cp), 'mate_vec': {},
            'root_move_depth_vec': dict.fromkeys(legal, depth)}


class FakeSearch:
    def __init__(self, infos):
        self.infos = infos
        self.stopped = False
        self.closed = False

    def __enter__(self): return self
    def __iter__(self): return iter(self.infos)
    def __exit__(self, *args): self.closed = True
    def stop(self): self.stopped = True


class SearchTests(unittest.TestCase):
    def engine(self, incomplete=False):
        engine = Mock()
        def analyse(board, limit, multipv, root_moves=None):
            moves = root_moves or list(board.legal_moves)[:multipv]
            return FakeSearch([{"depth": limit.depth - int(incomplete), "pv": [move], "multipv": index + 1,
                                "score": chess.engine.PovScore(chess.engine.Cp(index), board.turn)}
                               for index, move in enumerate(moves)])
        engine.analysis.side_effect = analyse
        return engine

    def test_staged_covers_all_moves_and_finishes_played_and_human_moves(self):
        engine = self.engine()
        board = chess.Board()
        options = {"maiaCandidateMoves": ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3"],
                   "forcedCandidateMoves": ["a2a3"]}
        frames = list(stream_evaluations(engine, board, 18, options, strategy='staged'))
        self.assertTrue(all(not frame["complete"] and frame["depth"] < 18 for frame in frames[:-1]))
        final = frames[-1]
        self.assertTrue(final["complete"])
        self.assertEqual(final["depth"], 18)
        self.assertEqual(set(final["cp_vec"]), {move.uci() for move in board.legal_moves})
        for move in options["maiaCandidateMoves"][:4] + ["a2a3"]:
            self.assertEqual(final["root_move_depth_vec"][move], 18)
        self.assertEqual(final["root_move_depth_vec"]["b2b4"], 10)
        calls = engine.analysis.call_args_list
        self.assertEqual(calls[0].kwargs["multipv"], 20)
        self.assertEqual(calls[0].args[1].depth, 10)
        self.assertTrue(any(call.kwargs["multipv"] == 4 and call.args[1].depth == 18 for call in calls))

    def test_exhaustive_keeps_every_root_at_requested_depth(self):
        engine = self.engine()
        result = list(stream_evaluations(engine, chess.Board(), 18, strategy="exhaustive"))[-1]
        self.assertEqual(set(result["root_move_depth_vec"].values()), {18})
        self.assertTrue(result["complete"])
        self.assertEqual(engine.analysis.call_count, 1)

    def test_bounded_time_finish_keeps_real_depths_and_prioritizes_played_move(self):
        engine = self.engine(incomplete=True)
        options = {'maiaCandidateMoves':['e2e4','d2d4'], 'forcedCandidateMoves':['a2a3']}
        events = []
        result = list(stream_evaluations(engine, chess.Board(), 18, options, 'bounded', on_search=events.append, seconds=6.))[-1]
        self.assertTrue(result['complete'])
        self.assertFalse(result['target_reached'])
        self.assertEqual(result['stop_reason'], 'time')
        self.assertEqual(result['depth'], 17)
        self.assertEqual(result['target_depth'], 18)
        self.assertEqual(result['root_move_depth_vec']['b2b4'], 5)
        self.assertEqual(result['root_move_depth_vec']['a2a3'], 17)
        self.assertEqual(result['budget_seconds'], 6)
        calls = engine.analysis.call_args_list
        self.assertEqual(calls[0].args[1].depth, 6)
        self.assertLessEqual(calls[0].args[1].time, .3)
        self.assertEqual(calls[1].kwargs['multipv'], 1)
        self.assertEqual(calls[2].kwargs['root_moves'], [chess.Move.from_uci('a2a3')])
        self.assertTrue(all(call.args[1].time > 0 for call in calls))
        self.assertTrue(all(event['stop_reason']=='time' for event in events if event['status']=='finished'))

    def test_bounded_searches_share_one_deadline_and_cancellation_is_not_completion(self):
        engine = self.engine()
        original = engine.analysis.side_effect
        now = [100.]
        def analyse(*args, **kwargs):
            result = original(*args, **kwargs)
            now[0] += args[1].time
            return result
        engine.analysis.side_effect = analyse
        with patch('analysis.stockfish_search.time.perf_counter', side_effect=lambda:now[0]):
            result = list(stream_evaluations(engine, chess.Board(), 18,
                          {'forcedCandidateMoves':['a2a3']}, 'bounded', seconds=6.))[-1]
        self.assertLessEqual(now[0]-100, 6.00001)
        self.assertTrue(result['complete'])
        control = SearchControl()
        control.cancel()
        self.assertEqual(list(stream_evaluations(engine, chess.Board(), 18, strategy='bounded',control=control)), [])

    def test_shallow_screening_outlier_does_not_replace_best_engine_move(self):
        engine = self.engine()
        original = engine.analysis.side_effect
        def analyse(board, limit, **kwargs):
            result = original(board, limit, **kwargs)
            if limit.depth == 6:
                for info in result.infos:
                    if info['pv'][0].uci() == 'a2a3':
                        info['score'] = chess.engine.PovScore(chess.engine.Cp(9000), chess.WHITE)
            return result
        engine.analysis.side_effect = analyse
        result = list(stream_evaluations(engine, chess.Board(), 18, strategy='bounded'))[-1]
        self.assertNotEqual(result['best_move'], 'a2a3')
        self.assertEqual(result['engine_moves'][0], result['best_move'])
        self.assertEqual(result['root_move_depth_vec']['a2a3'], 6)

    def test_profiling_records_every_native_search_without_changing_scores(self):
        events = []
        options = {'maiaCandidateMoves':['e2e4'], 'forcedCandidateMoves':['a2a3']}
        profiled = list(stream_evaluations(self.engine(), chess.Board(), 18, options, strategy='staged', on_search=events.append))
        ordinary = list(stream_evaluations(self.engine(), chess.Board(), 18, options, strategy='staged'))
        self.assertEqual(profiled, ordinary)
        started = [event for event in events if event['status'] == 'started']
        finished = [event for event in events if event['status'] == 'finished']
        self.assertEqual(len(started), len(finished))
        self.assertEqual({event['phase'] for event in finished}, {'screening','human_mid','engine_top','human_final'})
        self.assertTrue(all(event['wall_ms'] >= 0 and event['achieved_depth'] == event['target_depth'] for event in finished))

    def test_interrupted_search_never_reports_completion(self):
        with self.assertRaisesRegex(RuntimeError, "stopped at depth"):
            list(stream_evaluations(self.engine(incomplete=True), chess.Board(), 18, strategy='staged'))

    def test_cancel_stops_active_search_and_skips_queued_search(self):
        control = SearchControl()
        search = FakeSearch([])
        control.attach(search)
        control.cancel()
        self.assertTrue(search.stopped)
        control.attach(None)
        late_search = FakeSearch([])
        control.attach(late_search)
        self.assertTrue(late_search.stopped)
        engine = self.engine()
        self.assertEqual(list(stream_evaluations(engine, chess.Board(), 18, control=control)), [])
        engine.analysis.assert_not_called()

    def test_same_depth_buckets_do_not_mix_iterations(self):
        engine = Mock()
        board = chess.Board('7k/8/8/8/8/8/8/R5K1 b - - 0 1')
        moves = list(board.legal_moves)
        def info(depth, move):
            return {"depth": depth, "pv": [move], "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn)}
        infos = [info(1, moves[0]), *[info(2, move) for move in moves[1:]], info(2, moves[0])]
        engine.analysis.return_value = FakeSearch(infos)
        frames = list(stream_evaluations(engine, board, 2, strategy="exhaustive"))
        self.assertEqual(len(frames), 2)  # one complete depth plus the final marker
        self.assertEqual(set(frames[-1]["root_move_depth_vec"].values()), {2})

    def test_configured_strategy_and_evaluation_seconds_apply_without_frontend_scaling(self):
        with patch.dict(CONFIG['ANALYSIS'], STOCKFISH_SEARCH_STRATEGY='bounded'), \
             patch.dict(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], DEFAULT_SEARCH_SECONDS=3.125, MAX_SEARCH_SECONDS=5.):
            for depth in (12, 15, 18, 28):
                result = list(stream_evaluations(self.engine(), chess.Board(), depth))[-1]
                self.assertEqual(result['strategy'], 'bounded')
                self.assertEqual(result['budget_seconds'], 3.125)
            self.assertEqual(resolve_search_seconds(4.5), 4.5)
            for invalid in (0, -1, True, float('nan'), float('inf'), 5.001):
                with self.subTest(seconds=invalid), self.assertRaises(ValueError):
                    resolve_search_seconds(invalid)
            with self.assertRaises(ValueError):
                resolve_search_seconds(3., maximum=2.)
            for invalid in (0, -1, True, float('nan'), float('inf'), '5'):
                with self.subTest(maximum=invalid), self.assertRaises(ValueError):
                    resolve_search_seconds(1., maximum=invalid)
        with patch.dict(CONFIG['ANALYSIS'], STOCKFISH_SEARCH_STRATEGY='exhaustive'):
            result = list(stream_evaluations(self.engine(), chess.Board(), 18))[-1]
            self.assertEqual(result['strategy'], 'exhaustive')


if __name__ == "__main__":
    unittest.main()
