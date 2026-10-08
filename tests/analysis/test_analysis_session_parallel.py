import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chess

from analysis.session import AnalysisSession


class GameParallelTests(unittest.TestCase):
    def test_short_game_and_full_pool_keep_configured_threads_and_hash_per_worker(self):
        with tempfile.TemporaryDirectory() as root:
            session=AnalysisSession(chess.STARTING_FEN,root,stockfish_path='unused',
                            analysis_workers=4,threads_per_worker=3,hash_mb=512)
            session.initial_analysis=Mock(return_value={'fixture':True})
            for positions,workers in ((1,1),(2,2),(8,4)):
                with self.subTest(positions=positions):
                    results=list(session.analyze_positions([(i,) for i in range(positions)]))
                    self.assertEqual(len(results),positions)
                    self.assertEqual(session.last_analysis_execution,
                                     {'workers':workers,'threads_per_worker':3,'hash_mb_per_worker':512})
                    self.assertIsNone(session.engines.analysis_pool)
            session.close()

    def test_out_of_order_results_keep_original_indices_and_close_workers(self):
        with tempfile.TemporaryDirectory() as root:
            session=AnalysisSession(chess.STARTING_FEN,root,stockfish_path='unused')
            pool=Mock(workers=2,threads_per_worker=4,hash_mb=256)
            release_first=threading.Event()
            def scan(index):
                if index == 0 and not release_first.wait(10):
                    raise RuntimeError('The test did not release the first position.')
                return index*10
            session.initial_analysis=scan
            with patch('engine.runtime.StockfishPool',return_value=pool):
                scans=session.analyze_positions([(i,) for i in range(5)])
                try:
                    first=next(scans)
                finally:
                    release_first.set()
                results=[first,*scans]
            self.assertNotEqual(results[0][0],0)
            self.assertEqual(sorted(results),[(i,i*10) for i in range(5)])
            pool.close.assert_called_once()
            self.assertIsNone(session.engines.analysis_pool)

    def test_worker_error_closes_pool_and_restores_interactive_path(self):
        with tempfile.TemporaryDirectory() as root:
            session=AnalysisSession(chess.STARTING_FEN,root,stockfish_path='unused')
            session.initial_analysis=Mock(side_effect=RuntimeError('fixture'))
            pool=Mock(workers=2,threads_per_worker=4,hash_mb=256)
            with patch('engine.runtime.StockfishPool',return_value=pool),self.assertRaises(RuntimeError):
                list(session.analyze_positions([(0,),(1,)]))
            pool.close.assert_called_once()
            self.assertIsNone(session.engines.analysis_pool)

    def test_cross_position_inference_reuses_individual_pair_cache(self):
        with tempfile.TemporaryDirectory() as root:
            session=AnalysisSession(chess.STARTING_FEN,Path(root),stockfish_path='unused')
            session.engines.signature={'fixture':True}
            a=chess.Board(); b=a.copy(); b.push_uci('e2e4')
            def inference(fens,own,other,*,boards):
                self.assertEqual([len(b.move_stack) for b in boards],[0,0,1,1])
                return [{'policy':{m.uci():1/b.legal_moves.count() for m in b.legal_moves},'value':.5} for b in boards]
            session.engines.maia=Mock(); session.engines.maia.batch_evaluate.side_effect=inference
            jobs=[(board,[1500,1600],[1500,1600]) for board in (a,b)]
            result=session.human_pair_batches(jobs)
            self.assertEqual(session.engines.maia.batch_evaluate.call_count,1)
            self.assertEqual(session.human_pair_batches(jobs),result)
            self.assertEqual(session.human_pairs(['e2e4'],[1500,1600],[1500,1600]),result[1])
            self.assertEqual(session.engines.maia.batch_evaluate.call_count,1)


if __name__=='__main__':unittest.main()
