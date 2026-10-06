import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from engine.stockfish_pool import StockfishPool
from analysis.stockfish_search import SearchControl


class PoolTests(unittest.TestCase):
    def test_independent_processes_keep_threads_and_hash_per_worker_and_close(self):
        created, active = [], set()
        peak = 0
        lock = threading.Lock()
        def start(*args, **kwargs):
            self.assertEqual(kwargs, {'threads':4,'hash_mb':512})
            engine=Mock(); created.append(engine); return engine
        with patch('engine.stockfish_pool.start_stockfish',side_effect=start):
            pool=StockfishPool('unused',workers=2,threads_per_worker=4,hash_mb=512)
            def work(index):
                nonlocal peak
                with pool.acquire() as engine:
                    with lock:
                        self.assertNotIn(id(engine),active)
                        active.add(id(engine)); peak=max(peak,len(active))
                    time.sleep(.02)
                    with lock: active.remove(id(engine))
                    return index
            with ThreadPoolExecutor(max_workers=4) as executor:
                self.assertEqual(list(executor.map(work,range(8))),list(range(8)))
            pool.close()
        self.assertEqual((len(created),peak),(2,2))
        for engine in created: engine.quit.assert_called_once()

    def test_cancelled_queued_search_does_not_borrow_or_start_engine(self):
        with patch('engine.stockfish_pool.start_stockfish',return_value=Mock()) as start:
            pool=StockfishPool('unused',workers=1,threads_per_worker=8,hash_mb=512)
            with pool.acquire():
                control=SearchControl(); control.cancel()
                with pool.acquire(control) as engine:self.assertIsNone(engine)
            self.assertEqual(start.call_count,1)
            pool.close()

    def test_failed_engine_is_replaced_and_closes_once(self):
        first,second=Mock(),Mock()
        with patch('engine.stockfish_pool.start_stockfish',side_effect=[first,second]):
            pool=StockfishPool('unused',workers=1,threads_per_worker=8,hash_mb=512)
            with self.assertRaises(TimeoutError):
                with pool.acquire(): raise TimeoutError('fixture')
            with pool.acquire() as engine:self.assertIs(engine,second)
            pool.close()
        first.quit.assert_called_once(); second.quit.assert_called_once()
        with self.assertRaises(RuntimeError):
            with pool.acquire(): pass

    def test_per_worker_hash_does_not_cap_worker_count(self):
        pool=StockfishPool('unused',workers=16,threads_per_worker=8,hash_mb=64)
        self.assertEqual((pool.workers,pool.threads_per_worker,pool.hash_mb,pool.total_threads),(16,8,64,128))
        pool.close()

    def test_worker_count_never_divides_configured_threads_or_hash(self):
        for workers in (1, 4, 8):
            for threads in (1, 2, 3):
                with self.subTest(workers=workers, threads=threads), \
                     patch('engine.stockfish_pool.start_stockfish',return_value=Mock()) as start:
                    pool=StockfishPool('unused',workers=workers,threads_per_worker=threads,hash_mb=512)
                    try:
                        self.assertEqual((pool.workers,pool.total_threads),(workers,workers*threads))
                        with pool.acquire():
                            start.assert_called_once_with('unused',threads=threads,hash_mb=512)
                    finally:
                        pool.close()

    def test_invalid_resource_allocations_are_rejected(self):
        for field, value in [('workers',0),('workers',True),('threads_per_worker',0),
                             ('threads_per_worker',1.5),('threads_per_worker',True),
                             ('hash_mb',15),('hash_mb',True),('hash_mb',16.5)]:
            values={'workers':4,'threads_per_worker':2,'hash_mb':512,field:value}
            with self.subTest(values=values), self.assertRaises(ValueError):
                StockfishPool('unused',**values)

    def test_close_while_a_caller_dequeues_never_hands_out_the_closed_engine(self):
        engine = Mock()
        with patch('engine.stockfish_pool.start_stockfish', return_value=engine):
            pool = StockfishPool('unused', workers=1, threads_per_worker=1, hash_mb=16)
            with pool.acquire():
                pass
            original_get = pool._slots.get

            def dequeue_then_close(*args, **kwargs):
                result = original_get(*args, **kwargs)
                pool.close()
                return result

            with patch.object(pool._slots, 'get', side_effect=dequeue_then_close):
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    with pool.acquire():
                        self.fail('The closed engine must not be leased')
            engine.quit.assert_called_once()

    def test_engine_failure_during_shutdown_does_not_close_it_twice(self):
        engine = Mock()
        with patch('engine.stockfish_pool.start_stockfish', return_value=engine):
            pool = StockfishPool('unused', workers=1, threads_per_worker=1, hash_mb=16)
            with self.assertRaises(TimeoutError):
                with pool.acquire():
                    pool.close()
                    raise TimeoutError('search interrupted by shutdown')
            pool.close()
        engine.quit.assert_called_once()

    def test_clear_hash_after_close_raises_instead_of_waiting_for_held_lease(self):
        with patch('engine.stockfish_pool.start_stockfish', return_value=Mock()):
            pool = StockfishPool('unused', workers=1, threads_per_worker=1, hash_mb=16)
            with pool.acquire():
                pool.close()
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    pool.clear_hash()

    def test_concurrent_hash_clears_collect_slots_one_caller_at_a_time(self):
        pool = StockfishPool('unused', workers=2, threads_per_worker=1, hash_mb=32)
        original_get = pool._slots.get
        first_dequeued = threading.Event()
        continue_first = threading.Event()
        second_started = threading.Event()
        calls = []

        def controlled_get(*args, **kwargs):
            calls.append(threading.current_thread().name)
            item = original_get(*args, **kwargs)
            if len(calls) == 1:
                first_dequeued.set()
                self.assertTrue(continue_first.wait(2))
            return item

        def second_clear():
            second_started.set()
            pool.clear_hash()

        with patch.object(pool._slots, 'get', side_effect=controlled_get):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(pool.clear_hash)
                self.assertTrue(first_dequeued.wait(2))
                second = executor.submit(second_clear)
                self.assertTrue(second_started.wait(2))
                # The second caller cannot consume a lease held by the first.
                self.assertEqual(len(calls), 1)
                continue_first.set()
                first.result(timeout=2)
                second.result(timeout=2)
        pool.close()
        self.assertEqual(len(calls), 4)


if __name__=='__main__':unittest.main()
