"""Owned or borrowed Maia/Stockfish resources, without analysis or result caches."""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

from engine.stockfish_pool import StockfishPool
from engine.uci import close_engine, start_stockfish, stockfish_executable
from engine.assets_identity import asset_identity


def configured_signature():
    """Resolve the configured engine identity using local files, without engines."""
    from engine.settings import CONFIG
    from engine.maia import MaiaPolicy
    maia = MaiaPolicy.asset_signature(CONFIG['MAIA']['MODEL'], CONFIG['MAIA']['CACHE_DIR'],
                                      CONFIG['MAIA']['DEVICE'], checkpoint=CONFIG['MAIA']['CHECKPOINT'])
    executable = stockfish_executable(CONFIG['STOCKFISH']['EXECUTABLE'], CONFIG['STOCKFISH']['CACHE_DIR'])
    path = Path(shutil.which(str(executable)) or executable).resolve()
    return {**maia, 'stockfish': asset_identity(path),
            'threads': CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'],
            'hash_mb': CONFIG['ANALYSIS']['STOCKFISH_HASH_MB_PER_WORKER']}


class EngineRuntime:
    """Own native resource lifetimes; callers supply configuration explicitly."""

    def __init__(self, *, stockfish_path, stockfish_cache_directory,
                 maia_model, maia_cache_directory, checkpoint, device,
                 threads_per_worker, hash_mb, start_timeout_seconds):
        self.stockfish_path = stockfish_executable(stockfish_path, stockfish_cache_directory)
        self.maia_model = maia_model
        self.maia_cache_directory = maia_cache_directory
        self.checkpoint = checkpoint
        self.device = device
        self.threads_per_worker = threads_per_worker
        self.hash_mb = hash_mb
        self.start_timeout_seconds = start_timeout_seconds
        self.maia = self.stockfish = self.analysis_pool = None
        self.signature = None
        self._borrowed = False
        self._start_lock = threading.Lock()
        self._closed = False

    def borrow(self, maia, stockfish_scorer):
        """Attach an application's resources without taking ownership of them."""
        if self.maia is not None or self.stockfish is not None or self.analysis_pool is not None:
            raise RuntimeError('Cannot borrow engines into an active runtime.')
        self._borrowed = True
        self.maia = maia
        self.stockfish_path = stockfish_scorer.executable
        self.threads_per_worker = stockfish_scorer.threads_per_worker
        self.hash_mb = stockfish_scorer.hash_mb
        self.analysis_pool = stockfish_scorer.analysis_pool
        self._set_signature()

    def _set_signature(self):
        path = Path(shutil.which(str(self.stockfish_path)) or self.stockfish_path).resolve()
        self.signature = {'stockfish': asset_identity(path),
                          **self.maia.model_signature, 'threads': self.threads_per_worker,
                          'hash_mb': self.hash_mb}

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Engine runtime is closed.')
        if self._borrowed:
            return self
        try:
            from engine.maia import MaiaPolicy
            path = Path(shutil.which(str(self.stockfish_path)) or self.stockfish_path).resolve()
            if not path.is_file():
                raise ValueError('Stockfish not found; provide --stockfish-path or run python -m engine.assets.')
            self.maia = MaiaPolicy(self.maia_model, self.maia_cache_directory,
                                   self.device, checkpoint=self.checkpoint)
            self._set_signature()
            return self
        except BaseException:
            self.close()
            raise

    def get_stockfish(self):
        """Start the direct worker only after a cache miss needs a search."""
        with self._start_lock:
            if self._closed:
                raise RuntimeError('Engine runtime is closed.')
            if self._borrowed:
                raise RuntimeError('Borrowed searches must use the server worker pool.')
            if self.stockfish is None:
                self.stockfish = start_stockfish(self.stockfish_path, threads=self.threads_per_worker,
                    hash_mb=self.hash_mb, timeout=self.start_timeout_seconds)
            return self.stockfish

    def __exit__(self, *args):
        self.close()

    def create_pool(self, workers):
        """Allocate a worker pool at the concurrency chosen by the caller."""
        if self.analysis_pool is not None or self._borrowed:
            raise RuntimeError('This runtime already owns or borrows a worker pool.')
        self.analysis_pool = StockfishPool(self.stockfish_path, workers=workers,
                                           threads_per_worker=self.threads_per_worker,
                                           hash_mb=self.hash_mb)
        return self.analysis_pool

    def close_pool(self):
        if not self._borrowed and self.analysis_pool is not None:
            self.analysis_pool.close()
            self.analysis_pool = None

    def close(self):
        if self._borrowed:
            return
        with self._start_lock:
            self._closed = True
            self.close_pool()
            close_engine(self.stockfish)
            self.stockfish = self.maia = None
