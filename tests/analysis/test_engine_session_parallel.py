import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chess

from analysis.engine_session import Engines


class GameParallelTests(unittest.TestCase):
    def test_short_game_and_full_pool_keep_configured_threads_and_hash_per_worker(self):
        with tempfile.TemporaryDirectory() as root:
            engines=Engines(chess.STARTING_FEN,root,stockfish_path='unused',
                            analysis_workers=4,threads_per_worker=3,hash_mb=512)
            engines.initial_analysis=Mock(return_value={'fixture':True})
            for positions,workers in ((1,1),(2,2),(8,4)):
                with self.subTest(positions=positions):
                    results=list(engines.analyze_positions([(i,) for i in range(positions)]))
                    self.assertEqual(len(results),positions)
                    self.assertEqual(engines.last_analysis_execution,
                                     {'workers':workers,'threads_per_worker':3,'hash_mb_per_worker':512})
                    self.assertIsNone(engines.analysis_pool)
            engines.close()

    def test_out_of_order_results_keep_original_indices_and_close_workers(self):
        with tempfile.TemporaryDirectory() as root:
            engines=Engines(chess.STARTING_FEN,root,stockfish_path='unused')
            pool=Mock(workers=2,threads_per_worker=4,hash_mb=256)
            def scan(index):
                time.sleep(.02 if index==0 else .001)
                return index*10
            engines.initial_analysis=scan
            with patch('analysis.engine_session.StockfishPool',return_value=pool):
                results=list(engines.analyze_positions([(i,) for i in range(5)]))
            self.assertNotEqual(results[0][0],0)
            self.assertEqual(sorted(results),[(i,i*10) for i in range(5)])
            pool.close.assert_called_once()
            self.assertIsNone(engines.analysis_pool)

    def test_worker_error_closes_pool_and_restores_interactive_path(self):
        with tempfile.TemporaryDirectory() as root:
            engines=Engines(chess.STARTING_FEN,root,stockfish_path='unused')
            engines.initial_analysis=Mock(side_effect=RuntimeError('fixture'))
            pool=Mock(workers=2,threads_per_worker=4,hash_mb=256)
            with patch('analysis.engine_session.StockfishPool',return_value=pool),self.assertRaises(RuntimeError):
                list(engines.analyze_positions([(0,),(1,)]))
            pool.close.assert_called_once()
            self.assertIsNone(engines.analysis_pool)

    def test_cross_position_inference_reuses_legacy_pair_caches(self):
        with tempfile.TemporaryDirectory() as root:
            engines=Engines(chess.STARTING_FEN,Path(root),stockfish_path='unused')
            engines.signature={'fixture':True}
            a=chess.Board(); b=a.copy(); b.push_uci('e2e4')
            def inference(fens,own,other,*,boards):
                self.assertEqual([len(b.move_stack) for b in boards],[0,0,1,1])
                return [{'policy':{m.uci():1/b.legal_moves.count() for m in b.legal_moves},'value':.5} for b in boards]
            engines.maia=Mock(); engines.maia.batch_evaluate.side_effect=inference
            jobs=[(board,[1500,1600],[1500,1600]) for board in (a,b)]
            result=engines.human_pair_batches(jobs)
            self.assertEqual(engines.maia.batch_evaluate.call_count,1)
            self.assertEqual(engines.human_pair_batches(jobs),result)
            self.assertEqual(engines.human_pairs(['e2e4'],[1500,1600],[1500,1600]),result[1])
            self.assertEqual(engines.maia.batch_evaluate.call_count,1)


if __name__=='__main__':unittest.main()
