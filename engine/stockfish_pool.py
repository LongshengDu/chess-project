"""Independent Stockfish searches with fixed threads and hash memory per worker."""
from contextlib import contextmanager
from queue import Empty, Queue
import threading

import chess.engine

from .uci import start_stockfish, close_engine


class StockfishPool:
    def __init__(self, executable, *, workers, threads_per_worker, hash_mb):
        if any(type(v) is not int or v < 1 for v in (workers,threads_per_worker,hash_mb)) or hash_mb < 16:
            raise ValueError('Stockfish needs positive workers/threads per worker and at least 16 MB hash per worker.')
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.total_threads = self.workers * self.threads_per_worker
        self.hash_mb = hash_mb
        self.executable = executable
        self._slots = Queue()
        self._engines = []
        self._lock = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self._closed = threading.Event()
        for _ in range(self.workers):
            self._slots.put(None)

    @contextmanager
    def acquire(self, control=None):
        while True:
            if self._closed.is_set():
                raise RuntimeError('Stockfish pool is closed.')
            if control is not None and control.cancelled.is_set():
                yield None
                return
            try:
                engine = self._slots.get(timeout=.05)
                break
            except Empty:
                continue
        try:
            # close/cancel may have happened while this caller was queued.
            if self._closed.is_set():
                raise RuntimeError('Stockfish pool is closed.')
            if control is not None and control.cancelled.is_set():
                yield None
                return
            if engine is None:
                engine = start_stockfish(self.executable, threads=self.threads_per_worker, hash_mb=self.hash_mb)
                with self._lock:
                    if self._closed.is_set():
                        close_engine(engine)
                        raise RuntimeError('Stockfish pool is closed.')
                    self._engines.append(engine)
            yield engine
        except (chess.engine.EngineTerminatedError, TimeoutError):
            if engine is not None:
                with self._lock:
                    if engine in self._engines:
                        self._engines.remove(engine)
                        close_engine(engine)
                engine = None
            raise
        finally:
            self._slots.put(engine)

    def clear_hash(self):
        """Wait for current searches, then clear every existing worker once."""
        # Only one maintenance operation may collect all leases. Otherwise two
        # callers can each hold half the workers and wait for each other forever.
        with self._maintenance_lock:
            slots = []
            try:
                while len(slots) < self.workers:
                    if self._closed.is_set():
                        raise RuntimeError('Stockfish pool is closed.')
                    try:
                        slots.append(self._slots.get(timeout=.05))
                    except Empty:
                        continue
                with self._lock:
                    if self._closed.is_set():
                        raise RuntimeError('Stockfish pool is closed.')
                    for engine in slots:
                        if engine is not None:
                            engine.configure({'Clear Hash':None})
            finally:
                for engine in slots:
                    self._slots.put(engine)

    def close(self):
        self._closed.set()
        with self._lock:
            engines, self._engines = self._engines, []
        for engine in engines:
            close_engine(engine)
