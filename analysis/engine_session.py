"""Local persistent engines, bounded searches, and history-aware JSON caches."""
from __future__ import annotations

import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

from analysis.settings import CONFIG
from analysis.cache import JsonCache
from analysis.game.cancellation import AnalysisCancellation
from analysis.game.history import replay
from engine.uci import close_engine, start_stockfish, stockfish_executable, seconds_to_milliseconds
from analysis.stockfish_search import BOUNDED_POLICY_VERSION, SearchDeadline, stream_evaluations
from engine.stockfish_pool import StockfishPool


@dataclass(frozen=True)
class Limits:
    verify_ms: int = field(default_factory=lambda: seconds_to_milliseconds(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['DEFAULT_SEARCH_SECONDS']))
    max_ms: int = field(default_factory=lambda: seconds_to_milliseconds(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['MAX_SEARCH_SECONDS']))
    depth: int = field(default_factory=lambda: CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['MAX_DEPTH'])
    analysis_strategy: str = field(default_factory=lambda: CONFIG['ANALYSIS']['STOCKFISH_SEARCH_STRATEGY'])

    def __post_init__(self):
        max_search_ms = seconds_to_milliseconds(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['MAX_SEARCH_SECONDS'])
        if type(self.max_ms) is not int or not 1 <= self.max_ms <= max_search_ms:
            raise ValueError(f"The Stockfish per-search limit must be 1–{max_search_ms} ms (configured in seconds by ANALYSIS.STOCKFISH_EVALUATION.MAX_SEARCH_SECONDS).")
        if type(self.verify_ms) is not int or not 1 <= self.verify_ms <= self.max_ms:
            raise ValueError('Every search time must be positive and within --max-ms.')
        max_depth = CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['MAX_DEPTH']
        if type(self.depth) is not int or not 1 <= self.depth <= max_depth:
            raise ValueError(f'Depth ceiling must be 1–{max_depth} (ANALYSIS.STOCKFISH_EVALUATION.MAX_DEPTH).')
        if self.analysis_strategy not in ('bounded', 'staged', 'exhaustive'):
            raise ValueError('Analysis strategy must be bounded, staged or exhaustive.')


class Engines:
    def __init__(self, start_fen, cache_dir, *, limits=None, stockfish_path=None,
                 checkpoint=None, device=None,
                 analysis_workers=None, cancel=None, record=None,
                 threads_per_worker=CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'], hash_mb=CONFIG['ANALYSIS']['STOCKFISH_HASH_MB_PER_WORKER']):
        self.start_fen = start_fen
        self.limits = limits or Limits()
        self.cache = JsonCache(cache_dir)
        self.stockfish_path = stockfish_executable(stockfish_path or CONFIG['STOCKFISH']['EXECUTABLE'], CONFIG['STOCKFISH']['CACHE_DIR'])
        self.checkpoint = checkpoint if checkpoint is not None else CONFIG['MAIA']['CHECKPOINT']
        self.device = CONFIG['MAIA']['DEVICE'] if device is None else device
        self.threads_per_worker, self.hash_mb = threads_per_worker, hash_mb
        self.analysis_workers = CONFIG['ANALYSIS']['STOCKFISH_WORKERS'] if analysis_workers is None else analysis_workers
        if type(self.analysis_workers) is not int or not 1 <= self.analysis_workers <= 32:
            raise ValueError('ANALYSIS.STOCKFISH_WORKERS must be an integer from 1 to 32.')
        self.analysis_pool = None
        self._borrowed = False
        self.cancellation = AnalysisCancellation(cancel)
        self.record = record
        self.stockfish = self.maia = None
        self.lock = threading.RLock()
        self.signature = None
        self.stats = {'stockfish_calls': 0, 'stockfish_cache_hits': 0, 'maia_batches': 0, 'maia_cache_hits': 0}

    @classmethod
    def borrowed(cls, start_fen, cache_dir, maia, stockfish_scorer, *, limits=None, cancel=None, record=None):
        """Lease the server's engines; leaving this session never closes them."""
        session = cls(start_fen, cache_dir, limits=limits, cancel=cancel, record=record,
                      stockfish_path=stockfish_scorer.executable,
                      threads_per_worker=stockfish_scorer.threads_per_worker,
                      hash_mb=stockfish_scorer.hash_mb)
        session._borrowed = True
        session.maia = maia
        session.analysis_pool = stockfish_scorer.analysis_pool
        session.analysis_workers = session.analysis_pool.workers
        session._set_signature()
        return session

    def _set_signature(self):
        path = Path(shutil.which(str(self.stockfish_path)) or self.stockfish_path).resolve()
        self.signature = {'cache_version': 1, 'stockfish': [str(path), path.stat().st_mtime_ns],
                          **self.maia.model_signature, 'threads': self.threads_per_worker,
                          'hash_mb': self.hash_mb}

    def __enter__(self):
        if self._borrowed:
            self.cancellation.check()
            return self
        try:
            from engine.maia import MaiaPolicy
            path = Path(shutil.which(str(self.stockfish_path)) or self.stockfish_path).resolve()
            if not path.is_file():
                raise ValueError('Stockfish not found; provide --stockfish-path or run python -m engine.assets.')
            self.stockfish = start_stockfish(path, threads=self.threads_per_worker, hash_mb=self.hash_mb, timeout=CONFIG['STOCKFISH']['START_TIMEOUT_SECONDS'])
            self.maia = MaiaPolicy(CONFIG['MAIA']['MODEL'], CONFIG['MAIA']['CACHE_DIR'], self.device, checkpoint=self.checkpoint)
            self.maia.load()
            self._set_signature()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.cancellation.abort()
        if self._borrowed:
            return
        if self.analysis_pool is not None:
            self.analysis_pool.close()
            self.analysis_pool = None
        close_engine(self.stockfish)
        self.stockfish = None
        self.maia = None

    def board(self, history):
        return replay(self.start_fen, history)

    def human(self, history, ratings, opponent_elo):
        self.cancellation.check()
        board = self.board(history)
        ratings = list(dict.fromkeys(ratings))
        if not ratings or len(ratings) > 30 or any(type(r) is not int or not 600 <= r <= 3000 for r in ratings):
            raise ValueError('Supply 1–30 integer Maia ratings between 600 and 3000.')
        if type(opponent_elo) is not int or not 600 <= opponent_elo <= 3000:
            raise ValueError('Opponent Elo must be an integer between 600 and 3000.')
        result, missing, keys = {}, [], {}
        with self.lock:
            for rating in ratings:
                key = ['maia', self.signature, self.start_fen, history, rating, opponent_elo]
                keys[rating] = key
                cached = self.cache.get(key)
                if cached is None:
                    missing.append(rating)
                else:
                    result[str(rating)] = cached
                    self.stats['maia_cache_hits'] += 1
            if missing:
                if board.is_game_over(claim_draw=False):
                    winner = board.outcome().winner
                    predictions = [{'policy': {}, 'value': .5 if winner is None else float(winner)}] * len(missing)
                else:
                    predictions = self.maia.batch_evaluate(
                        [board.fen()] * len(missing), missing, [opponent_elo] * len(missing),
                        boards={board.fen(): board})
                    self.stats['maia_batches'] += 1
                for rating, prediction in zip(missing, predictions, strict=True):
                    policy = prediction['policy']
                    if not board.is_game_over() and (set(policy) != {m.uci() for m in board.legal_moves}
                            or any(not 0 <= p <= 1 for p in policy.values()) or not .99 <= sum(policy.values()) <= 1.01):
                        raise RuntimeError('Maia returned an incomplete or invalid legal-move distribution.')
                    result[str(rating)] = prediction
                    self.cache.put(keys[rating], prediction)
        return {str(r): result[str(r)] for r in ratings}

    def human_pairs(self, history, ratings, opponents):
        """Batch requested own/opponent Maia ratings while preserving game history."""
        self.cancellation.check()
        if not 1 <= len(ratings) <= 42 or len(ratings) != len(opponents):
            raise ValueError('Supply 1–42 matching own/opponent rating pairs.')
        if any(type(r) is not int or not 600 <= r <= 3000 for r in [*ratings, *opponents]):
            raise ValueError('Maia ratings must be integers between 600 and 3000.')
        board = self.board(history)
        key = ['maia-pairs-v1', self.signature, self.start_fen, history, list(ratings), list(opponents)]
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.stats['maia_cache_hits'] += len(ratings)
                return cached
            predictions = self.maia.batch_evaluate([board.fen()] * len(ratings), list(ratings), list(opponents),
                                                   boards={board.fen(): board})
            if len(predictions) != len(ratings):
                raise RuntimeError('Maia returned an incomplete rating batch.')
            legal = {move.uci() for move in board.legal_moves}
            for prediction in predictions:
                policy = prediction['policy']
                if set(policy) != legal or any(not 0 <= p <= 1 for p in policy.values()) or not .99 <= sum(policy.values()) <= 1.01:
                    raise RuntimeError('Maia returned an invalid legal-move distribution.')
            self.stats['maia_batches'] += 1
            self.cache.put(key, predictions)
            return predictions

    def human_pair_batches(self, requests):
        """Combine independent positions while reusing history-aware pair caches."""
        started = time.perf_counter()
        self.cancellation.check()
        results, missing, keys = [None]*len(requests), [], {}
        timings, inference_ms = {}, 0.
        with self.lock:
            for i,(board,own,other) in enumerate(requests):
                if (not 1 <= len(own) <= 42 or len(own) != len(other) or
                        any(type(r) is not int or not 600 <= r <= 3000 for r in [*own,*other])):
                    raise ValueError('Supply matching legal Maia rating pairs.')
                if board.root().fen() != chess.Board(self.start_fen).fen():
                    raise ValueError('Maia batch history starts from a different game.')
                history = [move.uci() for move in board.move_stack]
                key = ['maia-pairs-v1', self.signature, self.start_fen, history, list(own), list(other)]
                keys[i] = key
                cached = self.cache.get(key)
                if cached is not None:
                    results[i] = cached
                    self.stats['maia_cache_hits'] += len(own)
                else:
                    missing.append(i)
            if missing:
                boards, own, other = [], [], []
                for i in missing:
                    board, ratings, opponents = requests[i]
                    boards.extend([board]*len(ratings)); own.extend(ratings); other.extend(opponents)
                inferred = time.perf_counter()
                values = self.maia.batch_evaluate([board.fen() for board in boards], own, other, boards=boards,
                                                  **({'timings': timings} if self.record else {}))
                inference_ms = (time.perf_counter() - inferred) * 1000
                self.cancellation.check()
                if len(values) != len(boards):
                    raise RuntimeError('Maia returned an incomplete game batch.')
                self.stats['maia_batches'] += math.ceil(len(boards)/CONFIG['MAIA']['BATCH_SIZE'])
                offset = 0
                for i in missing:
                    count = len(requests[i][1])
                    results[i] = values[offset:offset+count]
                    legal = {move.uci() for move in requests[i][0].legal_moves}
                    for entry in results[i]:
                        policy = entry['policy']
                        if set(policy) != legal or any(not 0 <= p <= 1 for p in policy.values()) or not .99 <= sum(policy.values()) <= 1.01:
                            raise RuntimeError('Maia returned an invalid legal-move distribution.')
                    self.cache.put(keys[i],results[i])
                    offset += count
        if self.record:
            self.record('maia_batch', positions=[{'ply': len(board.move_stack), 'fen': board.fen(),
                'cache_hit': i not in missing, 'rating_pairs': len(own)}
                for i, (board, own, _) in enumerate(requests)],
                batch_size=sum(len(requests[i][1]) for i in missing),
                wall_ms=(time.perf_counter()-started)*1000, inference_ms=inference_ms, **timings)
        return results

    @contextmanager
    def _initial_engine(self, control=None):
        if self.analysis_pool is not None:
            with self.analysis_pool.acquire(control) as engine:
                self.cancellation.check()
                yield engine
        else:
            with self.lock:
                yield self.stockfish

    def analyze_positions(self, jobs, *, cancel=None):
        """Yield (original index, result) as independent game positions finish."""
        self.cancellation = AnalysisCancellation(cancel if cancel is not None else self.cancellation.event)
        pool = self.analysis_pool
        owned = pool is None and self.analysis_workers > 1 and len(jobs) > 1
        if owned:
            pool = StockfishPool(self.stockfish_path, workers=min(self.analysis_workers,len(jobs)),
                                 threads_per_worker=self.threads_per_worker, hash_mb=self.hash_mb)
            self.analysis_pool = pool
        self.last_analysis_execution = {'workers':min(pool.workers, len(jobs)) if pool else 1,
            'threads_per_worker':pool.threads_per_worker if pool else self.threads_per_worker,
            'hash_mb_per_worker':pool.hash_mb if pool else self.hash_mb}
        try:
            with self.cancellation.watch():
                if pool is None or len(jobs) < 2:
                    for i, job in enumerate(jobs):
                        self.cancellation.check()
                        yield i, self.initial_analysis(*job)
                else:
                    with ThreadPoolExecutor(max_workers=pool.workers) as executor:
                        futures = {executor.submit(self.initial_analysis,*job):i for i,job in enumerate(jobs)}
                        try:
                            for future in as_completed(futures):
                                self.cancellation.check()
                                yield futures[future],future.result()
                        except BaseException:
                            self.cancellation.abort()
                            raise
                        finally:
                            for future in futures:
                                future.cancel()
        finally:
            if owned:
                pool.close()
                self.analysis_pool = None

    def initial_analysis(self, history, played, human_candidates):
        """Run the configured root search, retaining complete legal-move coverage."""
        started = time.perf_counter()
        board = self.board(history)
        self.cancellation.check()
        if played is not None and played not in {move.uci() for move in board.legal_moves}:
            raise ValueError('The played move must be legal.')
        options = {'forcedCandidateMoves': [played] if played is not None else [], 'maiaCandidateMoves': list(human_candidates)[:4]}

        def completed(payload, *, cache_hit=False, acquire_ms=0.):
            timing = {'wall_ms': (time.perf_counter()-started)*1000,
                      'acquire_ms': acquire_ms, 'cache_hit': cache_hit}
            if self.record:
                self.record('stockfish', fen=board.fen(), ply=len(history),
                    terminal=board.is_game_over(claim_draw=False), **timing, options=options,
                    root_move_depth_vec={line['uci']:line['depth'] for line in payload['lines']},
                    **payload['search'])
            return payload

        if board.is_game_over(claim_draw=False):
            return completed({'lines': [], 'best_move': None, 'engine_moves': [], 'search': {
                'complete': True, 'coverage_complete': True, 'depth': 0,
                'target_depth': self.limits.depth,
                'terminal_cp': (-10000 if board.turn else 10000) if board.is_checkmate() else 0,
                'terminal_mate': 0 if board.is_checkmate() else None,
                'budget_seconds': self.limits.verify_ms / 1000,
                'max_budget_seconds': self.limits.max_ms / 1000,
                'result': board.result(), 'strategy': self.limits.analysis_strategy}})
        depth = self.limits.depth
        strategy = self.limits.analysis_strategy
        budget = (self.limits.verify_ms if strategy == 'bounded' else self.limits.max_ms) / 1000
        max_budget = self.limits.max_ms / 1000
        key = ['initial-analysis', BOUNDED_POLICY_VERSION, self.signature, self.start_fen,
               history, depth, budget, max_budget, strategy, options]
        if self.analysis_pool is not None:
            key += ['parallel', self.analysis_pool.threads_per_worker, self.analysis_pool.hash_mb]
        cached = self.cache.get(key)
        if (cached is not None and cached.get('search', {}).get('strategy') == strategy
                and cached['search'].get('budget_seconds') == budget
                and cached['search'].get('max_budget_seconds') == max_budget):
            with self.lock:
                self.stats['stockfish_cache_hits'] += 1
            return completed(cached, cache_hit=True)
        acquiring = time.perf_counter()
        with self.cancellation.search() as control, self._initial_engine(control) as engine:
            acquire_ms = (time.perf_counter()-acquiring)*1000
            if engine is None:
                raise RuntimeError('Stockfish is not running.')
            phases = []
            result = None

            def on_search(phase):
                if phase['status'] == 'finished':
                    phases.append(phase)
                if self.record:
                    self.record('stockfish_search', fen=board.fen(), ply=len(history), **phase)

            with SearchDeadline(engine, budget,
                    'Stockfish exceeded its initial position budget and was stopped.'):
                for result in stream_evaluations(engine, board, depth, options, strategy=strategy,
                        control=control, seconds=budget,
                        on_search=on_search):
                    pass
            self.cancellation.check()
            if not result or not result.get('complete') or not result.get('coverage_complete'):
                raise RuntimeError('Stockfish did not cover every legal move; increase the search time or select bounded ANALYSIS.STOCKFISH_SEARCH_STRATEGY.')
            lines = []
            for uci, cp in result['cp_vec'].items():
                move = chess.Move.from_uci(uci)
                child = board.copy(stack=True)
                san = child.san(move)
                child.push(move)
                mate = result['mate_vec'].get(uci)
                # Shared stream: White CP, but side-to-move mate distance.
                mate = mate * (1 if board.turn else -1) if mate is not None else None
                lines.append({'uci': uci, 'san': san, 'cp': cp if mate is None else None, 'mate': mate,
                    'depth': result['root_move_depth_vec'][uci], 'pv_uci': [uci], 'pv_san': [san],
                    'fen_after_pv': child.fen()})
            search = {k: v for k, v in result.items() if k not in ('cp_vec', 'mate_vec', 'root_move_depth_vec')}
            search['budget_seconds'] = budget
            search['max_budget_seconds'] = max_budget
            search['phases'] = phases
            payload = {'lines': lines, 'best_move': result['best_move'], 'engine_moves': result['engine_moves'], 'search': search}
            with self.lock:
                self.stats['stockfish_calls'] += len(phases)
            self.cache.put(key, payload)
            return completed(payload, acquire_ms=acquire_ms)

    def sf(self, history, movetime_ms, multipv=5, root_moves=None, pv_plies=10):
        self.cancellation.check()
        board = self.board(history)
        if type(movetime_ms) is not int or not 1 <= movetime_ms <= self.limits.max_ms:
            raise ValueError(f'Stockfish time must be 1–{self.limits.max_ms} ms.')
        if type(multipv) is not int or not 1 <= multipv <= 10 or type(pv_plies) is not int or not 1 <= pv_plies <= 24:
            raise ValueError('MultiPV must be 1–10 and PV length 1–24 plies.')
        roots = None
        if root_moves is not None:
            if not isinstance(root_moves, list) or not root_moves:
                raise ValueError('Root moves must be a nonempty list of legal UCI moves, or null.')
            roots = [chess.Move.from_uci(uci) for uci in sorted(set(root_moves))]
            if any(move not in board.legal_moves for move in roots):
                raise ValueError('Every restricted root move must be legal in this position.')
        key = ['stockfish', self.signature, self.start_fen, history, board.fen(),
               [m.uci() for m in roots] if roots else None, movetime_ms, self.limits.depth, multipv, pv_plies]
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.stats['stockfish_cache_hits'] += 1
                return cached
            if board.is_game_over(claim_draw=False):
                result = {'fen': board.fen(), 'terminal': True, 'result': board.result(), 'lines': [],
                          'evaluation': {'cp': None if board.is_checkmate() else 0,
                                         'mate': 0 if board.is_checkmate() else None,
                                         'winner': 'white' if board.outcome().winner else 'black' if board.is_checkmate() else None}}
            else:
                # UCI movetime AND depth ceiling, plus a watchdog for a hung process.
                # The extra two seconds are protocol/cleanup grace, not search time.
                infos = {}
                with self.cancellation.watch(), self.cancellation.search() as control, self._initial_engine(control) as engine:
                    if engine is None:
                        raise RuntimeError('Stockfish is not running.')
                    with SearchDeadline(engine, movetime_ms / 1000,
                            'Stockfish exceeded its time budget and was stopped.'):
                        with engine.analysis(board, chess.engine.Limit(time=movetime_ms / 1000, depth=self.limits.depth),
                                multipv=min(multipv, len(roots) if roots else board.legal_moves.count()), root_moves=roots) as search:
                            control.attach(search)
                            try:
                                for info in search:
                                    if info.get('pv') and 'score' in info and not info.get('upperbound') and not info.get('lowerbound'):
                                        infos[info.get('multipv', 1)] = info.copy()
                            finally:
                                control.attach(None)
                if not infos:
                    raise RuntimeError('Stockfish returned no exact scored line; increase the search time.')
                lines = []
                for rank, info in sorted(infos.items()):
                    walk, sans, ucis = board.copy(stack=True), [], []
                    for move in info['pv'][:pv_plies]:
                        sans.append(walk.san(move))
                        walk.push(move)
                        ucis.append(move.uci())
                    score = info['score'].white()
                    lines.append({'uci': ucis[0], 'san': sans[0], 'cp': score.score(), 'mate': score.mate(),
                                  'depth': info.get('depth', 0), 'pv_uci': ucis, 'pv_san': sans,
                                  'fen_after_pv': walk.fen()})
                result = {'fen': board.fen(), 'terminal': False, 'lines': lines,
                          'evaluation': {k: lines[0][k] for k in ('cp', 'mate')}}
                self.stats['stockfish_calls'] += 1
            result.update(score_perspective='white', movetime_ms=movetime_ms, depth_ceiling=self.limits.depth)
            self.cache.put(key, result)
            return result
