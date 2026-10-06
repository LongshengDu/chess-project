"""Persistent Stockfish adapter and worker-pool lifecycle."""
from __future__ import annotations

from pathlib import Path
import threading
import chess
import chess.engine

from engine.settings import CONFIG
from .uci import start_stockfish, close_engine
from .stockfish_pool import StockfishPool


class StockfishScorer:
    def __init__(self, executable: Path, threads_per_worker: int = CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'],
                 hash_mb: int = CONFIG['ANALYSIS']['STOCKFISH_HASH_MB_PER_WORKER']) -> None:
        self._executable = executable
        self.threads_per_worker, self.hash_mb = threads_per_worker, hash_mb
        self._engine: chess.engine.SimpleEngine | None = None
        self._analysis_pool = None
        self._pool_lock = threading.Lock()
        self._engine_lock = threading.RLock()
        self._closed = False

    @property
    def executable(self):
        return self._executable

    @property
    def analysis_pool(self):
        with self._pool_lock:
            if self._closed:
                raise RuntimeError('Stockfish scorer is closed.')
            if self._analysis_pool is None:
                self._analysis_pool = StockfishPool(self._executable, workers=CONFIG['ANALYSIS']['STOCKFISH_WORKERS'],
                                                    threads_per_worker=self.threads_per_worker, hash_mb=self.hash_mb)
            return self._analysis_pool

    def _get_engine(self) -> chess.engine.SimpleEngine:
        with self._engine_lock:
            if self._closed:
                raise RuntimeError('Stockfish scorer is closed.')
            if self._engine is None:
                if not self._executable.exists():
                    raise RuntimeError(
                        f"Stockfish executable not found: {self._executable}"
                    )
                self._engine = start_stockfish(self._executable, threads=self.threads_per_worker, hash_mb=self.hash_mb)
            return self._engine

    def _analyse(self, board: chess.Board, **options):
        settings = CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']
        limit = chess.engine.Limit(
            depth=settings['MAX_DEPTH'],
            time=min(settings['DEFAULT_SEARCH_SECONDS'], settings['MAX_SEARCH_SECONDS']),
        )
        # SimpleEngine cancels its current search when another command arrives.
        # Serialize this single worker; independent searches use analysis_pool.
        with self._engine_lock:
            return self._get_engine().analyse(board, limit, **options)

    @staticmethod
    def _result(info: dict) -> dict | None:
        engine_score = info.get("score")
        if engine_score is None:
            return None

        score = engine_score.pov(chess.WHITE)
        mate = score.mate()
        cp = score.score()
        label = (
            f"{'+' if mate >= 0 else '-'}M{abs(mate)}"
            if mate is not None
            else f"{(cp or 0) / 100:+.2f}"
        )
        return {"cp": cp, "mate": mate, "label": label}

    def evaluate(self, board: chess.Board) -> tuple[dict | None, int | None]:
        """Evaluate the board as shown, before either side's next move."""
        info = self._analyse(board)
        if not isinstance(info, dict):
            return None, None
        depth = int(info["depth"]) if "depth" in info else None
        return self._result(info), depth

    def score(
        self,
        board: chess.Board,
        moves: list[chess.Move],
    ) -> tuple[dict[str, dict], int | None]:
        if not moves:
            return {}, None

        infos = self._analyse(
            board,
            multipv=len(moves),
            root_moves=moves,
        )
        if isinstance(infos, dict):
            infos = [infos]

        scores: dict[str, dict] = {}
        depths = []
        for info in infos:
            if not info.get("pv"):
                continue
            move = info["pv"][0]
            result = self._result(info)
            if result is None:
                continue
            scores[move.uci()] = result
            if "depth" in info:
                depths.append(int(info["depth"]))
        return scores, min(depths) if depths else None

    def close(self) -> None:
        with self._engine_lock, self._pool_lock:
            self._closed = True
            if self._analysis_pool is not None:
                self._analysis_pool.close()
                self._analysis_pool = None
            if self._engine is not None:
                close_engine(self._engine)
                self._engine = None
