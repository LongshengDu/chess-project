"""Shared Maia inference and streamed Stockfish analysis for the local web app."""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import chess
from backend.settings import CONFIG
from engine.stockfish import StockfishScorer
from engine.assets_identity import asset_identity
from analysis.cache.positions import PositionCache
from analysis.cache.requests import (maia_request, stockfish_initial_request,
                                     stockfish_request_metadata, validate_maia_prediction)
from analysis.cache.validation import validate_measurement
from analysis.cache.policy import promote_request
from analysis.stockfish_search import SearchControl, SearchDeadline, stream_evaluations
from analysis.session import Limits
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
        self.cache = PositionCache(CONFIG['ANALYSIS']['CACHE_DIR'])
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

    def history_board(self, fen, start_fen=None, moves=None):
        """Validate the actual path, including variations and repetition history."""
        position = self.board(fen)
        if start_fen is None and moves is None:
            return position
        if not isinstance(moves, list) or len(moves) > 2000 or any(not isinstance(move, str) for move in moves):
            raise ValueError('Supply the complete legal UCI move history')
        board = self.board(start_fen)
        for uci in moves:
            move = chess.Move.from_uci(uci)
            if board.is_game_over(claim_draw=False) or move not in board.legal_moves:
                raise ValueError('Move history must contain legal moves before the game ends')
            board.push(move)
        if board.fen() != position.fen():
            raise ValueError('Move history does not match the requested position')
        return board

    @property
    def stockfish_signature(self):
        path = Path(shutil.which(str(self.stockfish.executable)) or self.stockfish.executable).resolve()
        return {'stockfish': asset_identity(path),
                'threads': self.stockfish.threads_per_worker, 'hash_mb': self.stockfish.hash_mb}

    @property
    def persistent_cache_enabled(self):
        # Cold profiler requests may reuse their own memory entries, not prior runs.
        return not (self.profiler is not None and self.profiler.active)

    @staticmethod
    def _memory_key(board, request):
        return (board.root().fen(), tuple(move.uci() for move in board.move_stack),
                board.fen(), json.dumps(request, sort_keys=True))

    def maia_predictions(self, boards, ratings, opponents):
        """Batch only missing rating pairs, sharing evidence with game analysis and play."""
        signature = self.maia.model_signature
        requests = [maia_request(signature, own, opponent)
                    for own, opponent in zip(ratings, opponents)]
        keys = [self._memory_key(board, request) for board, request in zip(boards, requests)]
        results, missing = {}, {}
        with self.maia_lock:
            persistent = self.persistent_cache_enabled
            lookups = {}
            for index, (board, request, key) in enumerate(zip(boards, requests, keys)):
                # Another session can refresh the same request while this server
                # remains alive. Persistent measurements are always authoritative.
                cached = None if persistent else self.maia_cache.get(key)
                if cached is None:
                    lookups.setdefault(key[:3], [board, {}])[1].setdefault(key, index)
                else:
                    validate_maia_prediction(board, cached)
                    results[key] = cached
            for board, indices_by_key in lookups.values():
                indices = list(indices_by_key.values())
                cached_results = (self.cache.get_many(board, 'maia', [requests[i] for i in indices])
                                  if persistent else [None] * len(indices))
                for index, cached in zip(indices, cached_results):
                    if cached is None:
                        missing[keys[index]] = index
                    else:
                        validate_maia_prediction(board, cached)
                        results[keys[index]] = cached
            indices = list(missing.values())
            if indices:
                predictions = self.maia.batch_evaluate(
                    [boards[i].fen() for i in indices], [ratings[i] for i in indices],
                    [opponents[i] for i in indices], boards=[boards[i] for i in indices])
                if not isinstance(predictions, list) or len(predictions) != len(indices):
                    raise ValueError('Maia returned an invalid prediction batch')
                updates = {}
                for index, prediction in zip(indices, predictions):
                    validate_maia_prediction(boards[index], prediction)
                    results[keys[index]] = prediction
                    if persistent:
                        updates.setdefault(keys[index][:3], [boards[index], []])[1].append(
                            (requests[index], prediction))
                for board, entries in updates.values():
                    self.cache.put_many(board, 'maia', entries)
            for key, result in results.items():
                self.maia_cache[key] = result
                self.maia_cache.move_to_end(key)
                if len(self.maia_cache) > CONFIG['MAIA']['REQUEST_CACHE_ENTRIES']:
                    self.maia_cache.popitem(last=False)
        return [results[key] for key in keys]

    def evaluate_maia(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        fens, ratings, opponents = (data.get(key) for key in ("fens", "ratings", "opponents"))
        if not all(isinstance(items, list) for items in (fens, ratings, opponents)):
            raise ValueError("Expected arrays of positions and ratings")
        if not 1 <= len(fens) <= CONFIG['MAIA']['BATCH_SIZE'] or len(fens) != len(ratings) or len(fens) != len(opponents):
            raise ValueError(f"Supply 1–{CONFIG['MAIA']['BATCH_SIZE']} positions with matching rating arrays")
        if any(type(rating) is not int or not 600 <= rating <= 2600 for rating in ratings + opponents):
            raise ValueError("Ratings must be integers from 600 to 2600")
        histories = data.get('histories')
        if histories is not None and not isinstance(histories, (dict, list)):
            raise ValueError('Expected position histories')
        if isinstance(histories, list) and len(histories) != len(fens):
            raise ValueError('Supply one history per position')
        boards = [self.history_board(fen, data.get('start_fen'),
                    histories[i] if isinstance(histories, list) else histories.get(fen)
                    if histories is not None else data.get('moves')) for i, fen in enumerate(fens)]
        start = time.perf_counter()
        result = self.maia_predictions(boards, ratings, opponents)
        return {"result": result, "time": (time.perf_counter() - start) * 1000}

    def stream_stockfish(self, board, depth, options=None, control=None, *, seconds=None):
        limits = self.analysis_limits(depth, seconds)
        seconds = (limits.verify_ms if self.strategy == 'bounded' else limits.max_ms) / 1000
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
        request = stockfish_initial_request(self.stockfish_signature, depth, seconds,
                                           limits.max_ms / 1000, self.strategy, options)
        key = self._memory_key(board, request)
        persistent = self.persistent_cache_enabled
        if persistent:
            record = self.cache.select_record(board, 'stockfish', request)
            cached = record['result'] if record else None
            if record is not None:
                cached = {**cached, **stockfish_request_metadata(record['request'])}
        else:
            with self.stockfish_cache_lock:
                cached = self.stockfish_cache.get(key)
                if cached is not None:
                    self.stockfish_cache.move_to_end(key)
        if cached is not None:
            yield json.dumps(cached) + '\n'
            return
        if persistent:
            request = promote_request(self.cache.records(board, 'stockfish'), request)
            depth, seconds = request['depth'], request['budget_seconds']
            options = {'forcedCandidateMoves': request['candidate_moves'], 'maiaCandidateMoves': []}
        with self.search_engine(control) as engine:
            if control.cancelled.is_set():
                return
            watchdog = seconds
            phases = []
            def on_search(phase):
                if phase['status'] == 'finished':
                    phases.append(phase)
            with SearchDeadline(engine, watchdog, 'Stockfish frontend analysis exceeded its search allowance'):
                for result in stream_evaluations(engine, board, depth, options, self.strategy, control,
                                                 seconds=seconds, on_search=on_search):
                    result.update(stockfish_request_metadata(request))
                    if result["complete"]:
                        result['budget_seconds'] = seconds
                        result['phases'] = phases
                        result['is_checkmate'] = board.is_checkmate()
                        try:
                            validate_measurement(board, 'stockfish', request, result)
                        except ValueError:
                            cacheable = False
                        else:
                            cacheable = True
                        if cacheable and not control.cancelled.is_set():
                            if persistent:
                                self.cache.put(board, 'stockfish', request, result)
                            with self.stockfish_cache_lock:
                                self.stockfish_cache[key] = result
                                if len(self.stockfish_cache) > CONFIG['ANALYSIS']['STOCKFISH_CACHE_ENTRIES']:
                                    self.stockfish_cache.popitem(last=False)
                    yield json.dumps(result) + '\n'

    @contextmanager
    def search_engine(self, control=None):
        if isinstance(self.stockfish, StockfishScorer):
            with self.stockfish.analysis_pool.acquire(control) as engine:
                yield engine
        else:
            # Small injected engine adapters retain the serial interface.
            with self.stockfish_lock:
                yield self.stockfish._get_engine()
