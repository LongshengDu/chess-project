"""Shared Maia inference and streamed Stockfish analysis for the local web app."""
from __future__ import annotations

import json
import math
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import chess
from backend.settings import CONFIG
from engine.stockfish import StockfishScorer
from analysis.stockfish_search import SearchControl, SearchDeadline, stream_evaluations, BOUNDED_POLICY_VERSION
from analysis.engine_session import Limits
from backend.repository import GameRepository
from backend.play import PlatformPlay
from backend.analysis_games import FullGameAnalysis


class PlatformAnalysis:
    def __init__(self, maia, stockfish, database: Path, strategy=CONFIG['ANALYSIS']['STOCKFISH_SEARCH_STRATEGY'], profiler=None):
        self.maia, self.stockfish = maia, stockfish
        self.repository = GameRepository(database)
        self.maia_lock = threading.Lock()
        self.stockfish_lock = threading.Lock()
        self.stockfish_cache_lock = threading.Lock()
        self.maia_cache = OrderedDict()
        self.stockfish_cache = OrderedDict()
        self.strategy = strategy
        self.profiler = profiler
        self.search_controls = {}
        self.controls_lock = threading.Lock()
        self.play = PlatformPlay(self)
        self.full_games = FullGameAnalysis(self)

    def analysis_presets(self):
        """Frontend presets scale the shared evaluation limits, never coach searches."""
        evaluation = CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']
        scales = CONFIG['FRONTEND']['STOCKFISH_TIME_SCALE_BY_DEPTH']
        if not isinstance(scales, dict) or not scales or any(
                type(depth) is not int or depth < 1 or type(scale) not in (int, float)
                or not math.isfinite(scale) or not 0 < scale <= 1
                for depth, scale in scales.items()):
            raise ValueError('Frontend search presets require positive integer depths and scales from 0 to 1')
        presets = {depth: scale for depth, scale in sorted(scales.items())
                   if depth <= evaluation['MAX_DEPTH']}
        if not presets:
            raise ValueError('No frontend analysis preset is within STOCKFISH_EVALUATION.MAX_DEPTH')
        return presets

    def analysis_limits(self, depth=None, seconds=None):
        """Resolve one frontend preset to the common analysis session interface."""
        presets = self.analysis_presets()
        depth = max(presets) if depth is None else depth
        if type(depth) is not int or depth not in presets:
            raise ValueError(f'Analysis depth must be a configured frontend preset: {", ".join(map(str, presets))}')
        evaluation = CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']
        maximum = evaluation['MAX_SEARCH_SECONDS'] * presets[depth]
        seconds = evaluation['DEFAULT_SEARCH_SECONDS'] * presets[depth] if seconds is None else seconds
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or not 0 < seconds <= maximum):
            raise ValueError(f'Analysis search time must be positive and no greater than {maximum:g} seconds')
        return Limits(depth=depth, verify_ms=round(seconds * 1000), max_ms=round(maximum * 1000),
                      analysis_strategy=self.strategy)

    @staticmethod
    def board(fen):
        if not isinstance(fen, str) or len(fen.split()) != 6:
            raise ValueError("Provide a complete six-field FEN")
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("Invalid chess position")
        return board

    def evaluate_maia(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        fens, ratings, opponents = (data.get(key) for key in ("fens", "ratings", "opponents"))
        if not all(isinstance(items, list) for items in (fens, ratings, opponents)):
            raise ValueError("Expected arrays of positions and ratings")
        if not 1 <= len(fens) <= CONFIG['MAIA']['BATCH_SIZE'] or len(fens) != len(ratings) or len(fens) != len(opponents):
            raise ValueError(f"Supply 1–{CONFIG['MAIA']['BATCH_SIZE']} positions with matching rating arrays")
        for fen in fens:
            self.board(fen)
        if any(type(rating) is not int or not 600 <= rating <= 2600 for rating in ratings + opponents):
            raise ValueError("Ratings must be integers from 600 to 2600")
        histories, boards = data.get('histories'), None
        if histories is not None:
            if not isinstance(histories,dict):
                raise ValueError('Expected position histories')
            start_board = self.board(data.get('start_fen'))
            boards = {}
            for fen in dict.fromkeys(fens):
                history = histories.get(fen)
                if not isinstance(history,list) or len(history) > 2000 or any(not isinstance(m,str) for m in history):
                    raise ValueError('Expected complete legal move histories')
                board = start_board.copy()
                for move in history:
                    board.push_uci(move)
                if board.fen() != self.board(fen).fen():
                    raise ValueError('Maia history does not match the requested position')
                boards[fen] = board
        key = (tuple(zip(fens, ratings, opponents)),
               None if boards is None else tuple((b.root().fen(),tuple(b.move_stack)) for b in boards.values()))
        start = time.perf_counter()
        with self.maia_lock:
            if key not in self.maia_cache:
                self.maia_cache[key] = self.maia.batch_evaluate(
                    fens, ratings, opponents,
                    **({'boards':boards} if boards is not None else {}))
                if len(self.maia_cache) > CONFIG['MAIA']['REQUEST_CACHE_ENTRIES']:
                    self.maia_cache.popitem(last=False)
            self.maia_cache.move_to_end(key)
            result = self.maia_cache[key]
        return {"result": result, "time": (time.perf_counter() - start) * 1000}

    def stream_stockfish(self, board, depth, options=None, control=None, *, seconds=None):
        limits = self.analysis_limits(depth, seconds)
        seconds = limits.verify_ms / 1000
        control = control or SearchControl()
        options = options or {}
        if control.cancelled.is_set():
            return
        if board.is_game_over():
            outcome = board.outcome()
            cp = 0 if outcome.winner is None else (10000 if outcome.winner else -10000)
            yield json.dumps({"terminal_cp": cp, "is_checkmate": board.is_checkmate(),
                              "depth": depth, "complete": True, "cp_vec": {},
                              "mate_vec": {"": 0} if board.is_checkmate() else {}}) + "\n"
            return
        key = (board.root().fen(), tuple(board.move_stack), board.fen(), depth, self.strategy, BOUNDED_POLICY_VERSION,
               seconds, limits.max_ms,
               tuple(options.get("maiaCandidateMoves", [])[:4]),
               tuple(options.get("forcedCandidateMoves", [])))
        with self.stockfish_cache_lock:
            cached = self.stockfish_cache.get(key)
            if cached is not None:
                self.stockfish_cache.move_to_end(key)
        if cached is not None:
            yield cached
            return
        with self.search_engine(control) as engine:
            if control.cancelled.is_set():
                return
            watchdog = seconds if self.strategy == 'bounded' else limits.max_ms / 1000
            with SearchDeadline(engine, watchdog, 'Stockfish frontend analysis exceeded its search allowance'):
                for result in stream_evaluations(engine, board, depth, options, self.strategy, control, seconds=seconds):
                    result['max_budget_seconds'] = limits.max_ms / 1000
                    encoded = json.dumps(result) + "\n"
                    if result["complete"]:
                        with self.stockfish_cache_lock:
                            self.stockfish_cache[key] = encoded
                            if len(self.stockfish_cache) > CONFIG['ANALYSIS']['STOCKFISH_CACHE_ENTRIES']:
                                self.stockfish_cache.popitem(last=False)
                    yield encoded

    @contextmanager
    def search_engine(self, control=None):
        if isinstance(self.stockfish, StockfishScorer):
            with self.stockfish.analysis_pool.acquire(control) as engine:
                yield engine
        else:
            # Small injected engine adapters retain the serial interface.
            with self.stockfish_lock:
                yield self.stockfish._get_engine()
