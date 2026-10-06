"""Interactive scoring must honor shared depth and time limits together."""
from engine.settings import CONFIG as ENGINE_CONFIG
from analysis.settings import CONFIG as ANALYSIS_CONFIG
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import chess
import chess.engine

from engine.stockfish import StockfishScorer
from analysis.stockfish_exploration import explore
from engine.uci import seconds_to_milliseconds, start_stockfish
from tests.analysis.test_stockfish_search import FakeSearch


class StockfishScorerTests(unittest.TestCase):
    def test_direct_searches_share_one_serial_worker_and_cannot_reopen_after_close(self):
        scorer = StockfishScorer(Path('fixture'))
        active, peak = 0, 0
        guard = threading.Lock()

        def analyse(*args, **kwargs):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(.005)
            with guard:
                active -= 1
            return {'score': chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE)}

        engine = Mock(analyse=Mock(side_effect=analyse))
        with patch.object(Path, 'exists', return_value=True), \
             patch('engine.stockfish.start_stockfish', return_value=engine) as start:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(scorer.evaluate, [chess.Board() for _ in range(8)]))
            scorer.close()
            scorer.close()
            self.assertEqual(peak, 1)
            start.assert_called_once()
            engine.quit.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                scorer.evaluate(chess.Board())
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                _ = scorer.analysis_pool

    def test_direct_exploration_uses_configured_default_and_maximum(self):
        engine = Mock()
        engine.analysis.return_value = FakeSearch([{
            'pv':[chess.Move.from_uci('e2e4')], 'depth':16,
            'score':chess.engine.PovScore(chess.engine.Cp(20),chess.WHITE)}])
        with patch.dict(ANALYSIS_CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION'],
                        DEFAULT_SEARCH_SECONDS=.625, MAX_SEARCH_SECONDS=3., MAX_DEPTH=16):
            result = explore(engine,chess.Board())
            limit = engine.analysis.call_args.args[1]
            self.assertEqual((limit.time,limit.depth),(.625,16))
            self.assertEqual(result['time_limit_seconds'],.625)
            explore(engine,chess.Board(),seconds=3.)
            self.assertEqual(engine.analysis.call_args.args[1].time,3.)
            for seconds in (3.001,0,True,float('inf'),float('nan')):
                with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                    explore(engine,chess.Board(),seconds=seconds)
            self.assertEqual(engine.analysis.call_count,2)
        for maximum in (0,True,float('nan'),float('inf'),'2'):
            with self.subTest(maximum=maximum), \
                 patch.dict(ANALYSIS_CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION'], MAX_SEARCH_SECONDS=maximum), \
                 self.assertRaises(ValueError):
                explore(engine,chess.Board())

    def test_yaml_seconds_convert_to_api_milliseconds_without_changing_cli_units(self):
        self.assertEqual(seconds_to_milliseconds(3.125),3125)
        self.assertEqual(seconds_to_milliseconds(13.875),13875)
        self.assertEqual(seconds_to_milliseconds(.001),1)
        for invalid in (True,0,-1,.0001,'3',float('nan'),float('inf')):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                seconds_to_milliseconds(invalid)

    def test_startup_reads_per_worker_hash_without_dividing_by_workers(self):
        engine = Mock()
        with patch.dict(ENGINE_CONFIG['ANALYSIS'], STOCKFISH_WORKERS=8,
                        STOCKFISH_THREADS_PER_WORKER=3, STOCKFISH_HASH_MB_PER_WORKER=192), \
             patch('engine.uci.chess.engine.SimpleEngine.popen_uci', return_value=engine):
            self.assertIs(start_stockfish('fixture'), engine)
        engine.configure.assert_called_once_with({'Threads': 3, 'Hash': 192})

    def test_scorer_uses_same_per_worker_threads_and_hash_for_pool_and_single_engine(self):
        scorer = StockfishScorer(Path('fixture'), threads_per_worker=3, hash_mb=512)
        with patch.dict(ENGINE_CONFIG['ANALYSIS'], STOCKFISH_WORKERS=4), \
             patch('engine.stockfish.start_stockfish',return_value=Mock()) as single, \
             patch('engine.stockfish_pool.start_stockfish',return_value=Mock()) as pooled, \
             patch.object(Path,'exists',return_value=True):
            try:
                scorer._get_engine()
                single.assert_called_once_with(Path('fixture'),threads=3,hash_mb=512)
                pool = scorer.analysis_pool
                self.assertEqual((pool.workers,pool.total_threads),(4,12))
                with pool.acquire():
                    pooled.assert_called_once_with(Path('fixture'),threads=3,hash_mb=512)
            finally:
                scorer.close()
        single.return_value.quit.assert_called_once()
        pooled.return_value.quit.assert_called_once()

    def test_position_search_keeps_deep_ceiling_with_default_time_limit(self):
        engine = Mock()
        engine.analyse.return_value = {'score': chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE), 'depth': 21}
        scorer = StockfishScorer(Path('unused'))
        scorer._engine = engine
        with patch.dict(ENGINE_CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], MAX_DEPTH=28, DEFAULT_SEARCH_SECONDS=3.0, MAX_SEARCH_SECONDS=30.0):
            score, depth = scorer.evaluate(chess.Board())
        limit = engine.analyse.call_args.args[1]
        self.assertEqual((limit.depth, limit.time), (28, 3.0))
        self.assertEqual((score['cp'], depth), (30, 21))

    def test_candidate_search_respects_lower_shared_maximum_and_real_depths(self):
        board = chess.Board()
        moves = [chess.Move.from_uci('e2e4'), chess.Move.from_uci('d2d4')]
        engine = Mock()
        engine.analyse.return_value = [
            {'pv': [move], 'score': chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE), 'depth': 16}
            for move in moves]
        scorer = StockfishScorer(Path('unused'))
        scorer._engine = engine
        with patch.dict(ENGINE_CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], MAX_DEPTH=28, DEFAULT_SEARCH_SECONDS=3.0, MAX_SEARCH_SECONDS=1.5):
            scores, depth = scorer.score(board, moves)
        limit = engine.analyse.call_args.args[1]
        self.assertEqual((limit.depth, limit.time), (28, 1.5))
        self.assertEqual(engine.analyse.call_args.kwargs, {'multipv': 2, 'root_moves': moves})
        self.assertEqual(set(scores), {'e2e4', 'd2d4'})
        self.assertEqual(depth, 16)


if __name__ == '__main__':
    unittest.main()
