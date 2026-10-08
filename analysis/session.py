"""Analysis sessions own evidence caches and coordinate cached engine requests."""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field

import chess

from analysis.settings import CONFIG
from analysis.cache.storage import identity
from analysis.cache.positions import PositionCache
from analysis.cache.requests import (maia_request, stockfish_initial_request,
                                     stockfish_exploration_request, stockfish_request_metadata,
                                     validate_maia_prediction)
from analysis.cache.policy import promote_request
from analysis.position_results import stockfish_result, stockfish_scan
from analysis.game.cancellation import AnalysisCancellation
from analysis.game.history import replay
from engine.runtime import EngineRuntime
from engine.uci import seconds_to_milliseconds
from analysis.stockfish_search import SearchDeadline, stream_evaluations
from analysis.stockfish_exploration import analysis_result, search_lines


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


class AnalysisSession:
    def __init__(self, start_fen, cache_directory, *, limits=None, stockfish_path=None,
                 checkpoint=None, device=None,
                 analysis_workers=None, cancel=None, record=None, refresh_cache=False,
                 threads_per_worker=CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'], hash_mb=CONFIG['ANALYSIS']['STOCKFISH_HASH_MB_PER_WORKER']):
        self.start_fen = start_fen
        self.limits = limits or Limits()
        self.cache = PositionCache(cache_directory, reuse_existing=not refresh_cache)
        self.analysis_workers = CONFIG['ANALYSIS']['STOCKFISH_WORKERS'] if analysis_workers is None else analysis_workers
        if type(self.analysis_workers) is not int or not 1 <= self.analysis_workers <= 32:
            raise ValueError('ANALYSIS.STOCKFISH_WORKERS must be an integer from 1 to 32.')
        self.engines = EngineRuntime(
            stockfish_path=stockfish_path or CONFIG['STOCKFISH']['EXECUTABLE'],
            stockfish_cache_directory=CONFIG['STOCKFISH']['CACHE_DIR'],
            maia_model=CONFIG['MAIA']['MODEL'], maia_cache_directory=CONFIG['MAIA']['CACHE_DIR'],
            checkpoint=checkpoint if checkpoint is not None else CONFIG['MAIA']['CHECKPOINT'],
            device=CONFIG['MAIA']['DEVICE'] if device is None else device,
            threads_per_worker=threads_per_worker, hash_mb=hash_mb,
            start_timeout_seconds=CONFIG['STOCKFISH']['START_TIMEOUT_SECONDS'])
        self.cancellation = AnalysisCancellation(cancel)
        self.record = record
        self.lock = threading.RLock()
        self._evidence = {}
        self.stats = {'stockfish_calls': 0, 'stockfish_cache_hits': 0, 'maia_batches': 0, 'maia_cache_hits': 0}

    @classmethod
    def borrowed(cls, start_fen, cache_directory, maia, stockfish_scorer, *, limits=None, cancel=None,
                 record=None, refresh_cache=False):
        """Lease the server's engines; leaving this session never closes them."""
        session = cls(start_fen, cache_directory, limits=limits, cancel=cancel, record=record,
                      refresh_cache=refresh_cache,
                      stockfish_path=stockfish_scorer.executable,
                      threads_per_worker=stockfish_scorer.threads_per_worker,
                      hash_mb=stockfish_scorer.hash_mb)
        session.engines.borrow(maia, stockfish_scorer)
        session.analysis_workers = session.engines.analysis_pool.workers
        return session

    def __enter__(self):
        self.cancellation.check()
        try:
            self.engines.__enter__()
            return self
        except BaseException:
            self.cancellation.abort()
            raise

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.cancellation.abort()
        self.engines.close()

    def board(self, history):
        return replay(self.start_fen, history)

    def _remember(self, board, namespace, request, result, reference=None):
        """Pin the exact observation returned, independently of later refreshes."""
        if reference is None:
            reference = self.cache.measurement_reference(board, namespace, request, result)
        context = identity({'start_fen': board.root().fen(), 'moves': [m.uci() for m in board.move_stack]})
        with self.lock:
            evidence = self._evidence.setdefault(context, {'maia': {}})
            if namespace == 'maia':
                evidence['maia'][(request['own_rating'], request['opponent_rating'])] = reference
            elif request['kind'] == 'evaluation':
                evidence['stockfish'] = reference

    def position_references(self, boards, ratings):
        references = []
        for index, board in enumerate(boards):
            context = identity({'start_fen': board.root().fen(), 'moves': [m.uci() for m in board.move_stack]})
            evidence = self._evidence[context]
            references.append({'fields': {'fen': board.fen(en_passant='fen'), 'ply': index},
                'maia': {f'maia_kdd_{r}': evidence['maia'][(r, r)] for r in ratings},
                'stockfish': evidence['stockfish']})
        return references

    def human(self, history, ratings, opponent_elo):
        ratings = list(dict.fromkeys(ratings))
        if not ratings or len(ratings) > 30 or any(type(r) is not int or not 600 <= r <= 3000 for r in ratings):
            raise ValueError('Supply 1–30 integer Maia ratings between 600 and 3000.')
        if type(opponent_elo) is not int or not 600 <= opponent_elo <= 3000:
            raise ValueError('Opponent Elo must be an integer between 600 and 3000.')
        values = self.human_pairs(history, ratings, [opponent_elo] * len(ratings))
        return dict(zip(map(str, ratings), values, strict=True))

    def human_pairs(self, history, ratings, opponents):
        """Single-position requests share each rating pair with all batch requests."""
        return self._human_requests([(self.board(history), list(ratings), list(opponents))],
                                    single_position=True)[0]

    def human_pair_batches(self, requests):
        """Infer all unique missing position/rating pairs together."""
        return self._human_requests(requests)

    def _human_requests(self, requests, *, single_position=False):
        started = time.perf_counter()
        self.cancellation.check()
        requests = list(requests)
        results, missing, locations, references = [], {}, {}, {}
        timings, inference_ms = {}, 0.
        positions_missing = set()
        with self.lock:
            for i, (board, own, other) in enumerate(requests):
                if (not 1 <= len(own) <= 42 or len(own) != len(other) or
                        any(type(r) is not int or not 600 <= r <= 3000 for r in [*own, *other])):
                    raise ValueError('Supply 1–42 matching legal Maia rating pairs.')
                if board.root().fen() != chess.Board(self.start_fen).fen():
                    raise ValueError('Maia batch history starts from a different game.')
                results.append([None] * len(own))
                pair_requests = [maia_request(self.engines.signature, rating, opponent)
                                 for rating, opponent in zip(own, other, strict=True)]
                cached_pairs = self.cache.select_many(board, 'maia', pair_requests)
                for j, (rating, opponent) in enumerate(zip(own, other, strict=True)):
                    request = pair_requests[j]
                    key = identity(self.cache.reference(board, 'maia', request))
                    locations.setdefault(key, []).append((i, j))
                    record = cached_pairs[j]
                    if record is not None:
                        cached = record['result']
                        validate_maia_prediction(board, cached)
                        results[i][j] = cached
                        references[key] = record['reference']
                        missing.pop(key, None)
                        self.stats['maia_cache_hits'] += 1
                    else:
                        missing.setdefault(key, (board, rating, opponent, request))
            # Another process can populate a repeated request during these reads.
            for key, indexes in locations.items():
                existing = next((results[i][j] for i, j in indexes if results[i][j] is not None), None)
                if existing is not None:
                    for i, j in indexes:
                        results[i][j] = existing
            inferred_items = []
            for key, item in missing.items():
                board = item[0]
                if board.is_game_over(claim_draw=False):
                    winner = board.outcome().winner
                    value = {'policy': {}, 'value': .5 if winner is None else float(winner)}
                    references[key] = self.cache.put(board, 'maia', item[3], value)
                    for i, j in locations[key]:
                        results[i][j] = value
                else:
                    inferred_items.append((key, item))
                    positions_missing.update(i for i, _ in locations[key])
            if inferred_items:
                boards = [item[0] for _, item in inferred_items]
                own = [item[1] for _, item in inferred_items]
                other = [item[2] for _, item in inferred_items]
                # Aligned boards retain distinct histories even for equal FENs.
                history_boards = {boards[0].fen(): boards[0]} if single_position else boards
                inferred = time.perf_counter()
                values = self.engines.maia.batch_evaluate([board.fen() for board in boards], own, other,
                    boards=history_boards, **({'timings': timings} if self.record else {}))
                inference_ms = (time.perf_counter() - inferred) * 1000
                self.cancellation.check()
                if len(values) != len(boards):
                    raise RuntimeError('Maia returned an incomplete game batch.')
                for board, value in zip(boards, values, strict=True):
                    validate_maia_prediction(board, value)
                self.stats['maia_batches'] += math.ceil(len(boards) / CONFIG['MAIA']['BATCH_SIZE'])
                grouped = {}
                for (key, (board, _, _, request)), value in zip(inferred_items, values, strict=True):
                    board_key = identity([board.root().fen(), [move.uci() for move in board.move_stack]])
                    grouped.setdefault(board_key, (board, []))[1].append((request, value))
                    for i, j in locations[key]:
                        results[i][j] = value
                for board, entries in grouped.values():
                    written = self.cache.put_many(board, 'maia', entries)
                    for (request, _), reference in zip(entries, written, strict=True):
                        references[identity(self.cache.reference(board, 'maia', request))] = reference
            for (board, own, other), values in zip(requests, results, strict=True):
                for rating, opponent, value in zip(own, other, values, strict=True):
                    request = maia_request(self.engines.signature, rating, opponent)
                    key = identity(self.cache.reference(board, 'maia', request))
                    self._remember(board, 'maia', request, value, references[key])
        if self.record:
            self.record('maia_batch', positions=[{'ply': len(board.move_stack), 'fen': board.fen(),
                'cache_hit': i not in positions_missing, 'rating_pairs': len(own)}
                for i, (board, own, _) in enumerate(requests)], batch_size=len(inferred_items),
                wall_ms=(time.perf_counter()-started)*1000, inference_ms=inference_ms, **timings)
        return results

    def _stockfish_signature(self):
        signature = dict(self.engines.signature or {})
        if self.engines.analysis_pool is not None:
            signature.update(threads=self.engines.analysis_pool.threads_per_worker,
                             hash_mb=self.engines.analysis_pool.hash_mb)
        return signature

    @contextmanager
    def _initial_engine(self, control=None):
        if self.engines.analysis_pool is not None:
            with self.engines.analysis_pool.acquire(control) as engine:
                self.cancellation.check()
                yield engine
        else:
            with self.lock:
                yield self.engines.get_stockfish()

    def analyze_positions(self, jobs, *, cancel=None):
        """Yield (original index, result) as independent game positions finish."""
        self.cancellation = AnalysisCancellation(cancel if cancel is not None else self.cancellation.event)
        pool = self.engines.analysis_pool
        owned = pool is None and self.analysis_workers > 1 and len(jobs) > 1
        if owned:
            pool = self.engines.create_pool(min(self.analysis_workers, len(jobs)))
        self.last_analysis_execution = {'workers':min(pool.workers, len(jobs)) if pool else 1,
            'threads_per_worker':pool.threads_per_worker if pool else self.engines.threads_per_worker,
            'hash_mb_per_worker':pool.hash_mb if pool else self.engines.hash_mb}
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
                self.engines.close_pool()

    def initial_analysis(self, history, played, human_candidates):
        """Run the configured root search, retaining complete legal-move coverage."""
        started = time.perf_counter()
        board = self.board(history)
        self.cancellation.check()
        if played is not None and played not in {move.uci() for move in board.legal_moves}:
            raise ValueError('The played move must be legal.')
        options = {'forcedCandidateMoves': [played] if played is not None else [], 'maiaCandidateMoves': list(human_candidates)[:4]}
        depth = self.limits.depth
        strategy = self.limits.analysis_strategy
        budget = (self.limits.verify_ms if strategy == 'bounded' else self.limits.max_ms) / 1000
        max_budget = self.limits.max_ms / 1000
        request = stockfish_initial_request(self._stockfish_signature(), depth, budget, max_budget, strategy, options)

        def completed(payload, *, cache_hit=False, acquire_ms=0.):
            timing = {'wall_ms': (time.perf_counter()-started)*1000,
                      'acquire_ms': acquire_ms, 'cache_hit': cache_hit}
            if self.record:
                scores = stockfish_result(board, payload)
                self.record('stockfish', fen=board.fen(), ply=len(history),
                    terminal=board.is_game_over(claim_draw=False), **timing, options=options,
                    root_move_depth_vec=scores['root_move_depth_vec'],
                    **{**payload['search'], **{key: scores[key] for key in ('best_move', 'cp_vec', 'mate_vec')}})
            return payload

        if board.is_game_over(claim_draw=False):
            payload = {'lines': [], 'best_move': None, 'engine_moves': [], 'search': {
                'complete': True, 'coverage_complete': True, 'depth': 0,
                'target_depth': self.limits.depth,
                'terminal_cp': (-10000 if board.turn else 10000) if board.is_checkmate() else 0,
                'terminal_mate': 0 if board.is_checkmate() else None,
                'budget_seconds': budget, 'max_budget_seconds': max_budget,
                'result': board.result(), 'strategy': strategy}}
            result = stockfish_result(board, payload)
            self.cache.put(board, 'stockfish', request, result)
            self._remember(board, 'stockfish', request, result)
            return completed(payload)
        record = self.cache.select_record(board, 'stockfish', request)
        frame = record['result'] if record else None
        # A completed interactive search may still lack some screened roots.
        # Full-game accuracy requires scores for every legal move.
        if frame is not None and (not frame.get('complete') or not frame.get('coverage_complete')
                or set(frame.get('cp_vec', {})) != {move.uci() for move in board.legal_moves}):
            frame = None
        cached = (stockfish_scan(board, {**frame, **stockfish_request_metadata(record['request'])})
                  if frame is not None else None)
        if cached is not None:
            self._remember(board, 'stockfish', record['request'], frame, record['reference'])
            with self.lock:
                self.stats['stockfish_cache_hits'] += 1
            return completed(cached, cache_hit=True)
        request = promote_request(self.cache.records(board, 'stockfish'), request)
        depth = request['depth']
        budget = request['budget_seconds']
        max_budget = request['max_budget_seconds']
        # The promoted candidate union must reach execution unchanged. Passing
        # it as forced roots avoids the four-human-candidate input truncation.
        options = {'forcedCandidateMoves': request['candidate_moves'], 'maiaCandidateMoves': []}
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
            result = {**result, **stockfish_request_metadata(request), 'phases': phases}
            payload = stockfish_scan(board, result)
            with self.lock:
                self.stats['stockfish_calls'] += len(phases)
            reference = self.cache.put(board, 'stockfish', request, result)
            self._remember(board, 'stockfish', request, result, reference)
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
        count = min(multipv, len(roots) if roots else max(1, board.legal_moves.count()))
        request = stockfish_exploration_request(self._stockfish_signature(), movetime_ms,
            self.limits.depth, count, [move.uci() for move in roots] if roots else None)

        def present(measurement, executed_request):
            return analysis_result(board, measurement, movetime_ms=executed_request['movetime_ms'],
                                   depth=executed_request['depth'], pv_plies=pv_plies)

        with self.lock:
            cached = self.cache.select_record(board, 'stockfish', request)
            if cached is not None:
                self.stats['stockfish_cache_hits'] += 1
                return present(cached['result'], cached['request'])
            request = promote_request(self.cache.records(board, 'stockfish'), request)
            if board.is_game_over(claim_draw=False):
                result = {'lines': []}
            else:
                with self.cancellation.watch(), self.cancellation.search() as control, self._initial_engine(control) as engine:
                    if engine is None:
                        raise RuntimeError('Stockfish is not running.')
                    result = search_lines(engine, board, seconds=request['movetime_ms'] / 1000,
                        depth=request['depth'], multipv=count, root_moves=roots, control=control)
                self.cancellation.check()
                self.stats['stockfish_calls'] += 1
            self.cache.put(board, 'stockfish', request, result)
            return present(result, request)
