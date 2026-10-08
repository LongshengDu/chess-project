"""Native counterpart of Maia's staged-root-probe search, with real per-move depths."""
from __future__ import annotations

import math
import threading
import time

import chess
import chess.engine
from analysis.settings import CONFIG


BOUNDED_POLICY_VERSION = 2


def search_candidates(depth, strategy, options=None):
    """Return the effective deepening order shared by execution and cache identity."""
    if strategy == 'exhaustive' or (strategy != 'bounded' and depth <= 4):
        return []
    options = options or {}
    forced = list(options.get('forcedCandidateMoves', []))
    human = list(options.get('maiaCandidateMoves', []))[:4]
    ordered = forced + human if strategy == 'bounded' else human + forced
    # Order affects sequential search budgets and hash reuse; only duplicates vanish.
    return list(dict.fromkeys(ordered))


def resolve_search_seconds(seconds=None, maximum=None):
    """Resolve a position allowance independently of its search-depth ceiling."""
    settings = CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']
    seconds = settings['DEFAULT_SEARCH_SECONDS'] if seconds is None else seconds
    maximum = settings['MAX_SEARCH_SECONDS'] if maximum is None else maximum
    if (isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum) or maximum <= 0):
        raise ValueError('Stockfish maximum search time must be a positive finite number of seconds.')
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds <= maximum):
        raise ValueError(f'Stockfish search time must be positive and at most {maximum} seconds.')
    return float(seconds)


class SearchDeadline:
    """Stop an unresponsive UCI process and consistently surface a timeout."""

    def __init__(self, engine, seconds, message):
        self.engine = engine
        self.message = message
        self.expired = threading.Event()
        self.timer = threading.Timer(seconds + 2, self._expire)
        self.timer.daemon = True

    def _expire(self):
        self.expired.set()
        self.engine.close()

    def __enter__(self):
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.timer.cancel()
        self.timer.join()
        if self.expired.is_set():
            raise TimeoutError(self.message) from exc


class SearchControl:
    """Allow an HTTP cancellation to stop UCI even between streamed results."""

    def __init__(self):
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.search = None

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            if self.search is not None:
                self.search.stop()

    def attach(self, search):
        with self.lock:
            self.search = search
            if search is not None and self.cancelled.is_set():
                search.stop()


def snapshots(engine, board, depth, multipv, control, root_moves=None, metrics=None, time_limit=None):
    """Only emit complete iterations; never combine rows from different depths."""
    if control.cancelled.is_set():
        return
    options = {"root_moves": root_moves} if root_moves else {}
    with engine.analysis(board, chess.engine.Limit(depth=depth, time=time_limit), multipv=multipv, **options) as search:
        control.attach(search)
        buckets, last_depth = {}, 0
        try:
            for info in search:
                if metrics is not None:
                    for key in ("nodes", "time", "nps", "hashfull", "seldepth", "tbhits"):
                        if key in info:
                            metrics[key] = max(metrics.get(key, 0), info[key])
                if control.cancelled.is_set():
                    return
                if not info.get("pv") or "score" not in info or "depth" not in info:
                    continue
                if info.get("upperbound") or info.get("lowerbound"):
                    continue
                current_depth = info["depth"]
                if current_depth <= last_depth:
                    continue
                move = info["pv"][0].uci()
                score = info["score"].pov(chess.WHITE)
                mate = score.mate()
                cp = score.score() if mate is None else (10000 if mate > 0 else -10000)
                rows = buckets.setdefault(current_depth, {})
                # The page expects White CP, but UCI side-to-move mate distance.
                rows[move] = (current_depth, cp, info["score"].pov(board.turn).mate())
                if len(rows) == multipv:
                    last_depth = current_depth
                    yield rows.copy()
                    buckets = {d: bucket for d, bucket in buckets.items() if d > last_depth}
        finally:
            control.attach(None)


def stream_evaluations(engine, board, depth, options=None, strategy=None, control=None, on_search=None, seconds=None):
    strategy = CONFIG['ANALYSIS']['STOCKFISH_SEARCH_STRATEGY'] if strategy is None else strategy
    if strategy not in ('bounded', 'staged', 'exhaustive'):
        raise ValueError('Search strategy must be bounded, staged or exhaustive.')
    if strategy == "bounded":
        yield from bounded_evaluations(engine, board, depth, options, control, on_search, seconds)
        return
    control = control or SearchControl()
    options = options or {}
    legal = {move.uci(): move for move in board.legal_moves}
    rows, engine_moves = {}, []
    best_move = None

    def evaluation(completed_depth, complete=False):
        return {"depth": completed_depth, "complete": complete, "strategy": strategy,
                "coverage_complete": len(rows) == len(legal),
                "best_move": best_move, "engine_moves": engine_moves,
                "cp_vec": {move: row[1] for move, row in rows.items()},
                "mate_vec": {move: row[2] for move, row in rows.items() if row[2] is not None},
                "root_move_depth_vec": {move: row[0] for move, row in rows.items()}}

    def search(target, count, moves=None, phase="exhaustive"):
        nonlocal best_move, engine_moves
        achieved = 0
        metrics = {} if on_search else None
        details = {"phase": phase, "target_depth": target, "multipv": count,
                   "root_move": moves[0].uci() if moves else None}
        if on_search:
            on_search({**details, "status": "started"})
        started = time.perf_counter()
        try:
            for snapshot in snapshots(engine, board, target, count, control, moves, metrics):
                achieved = min(row[0] for row in snapshot.values())
                if phase in ('engine_top', 'exhaustive'):
                    def strength(move):
                        _, cp, mate = snapshot[move]
                        return chess.engine.Mate(mate) if mate is not None else chess.engine.Cp(cp * (1 if board.turn else -1))
                    engine_moves = sorted(snapshot, key=strength, reverse=True)[:4]
                    best_move = engine_moves[0]
                for move, row in snapshot.items():
                    if move not in rows or row[0] >= rows[move][0]:
                        rows[move] = row
                # Completion is emitted only after ALL phases, including the played
                # move. A depth-18 top-four search alone is not a finished position.
                yield evaluation(min(depth - 1, achieved))
        finally:
            if on_search:
                on_search({**details, "status": "finished", "achieved_depth": achieved,
                           "wall_ms": (time.perf_counter() - started) * 1000,
                           "cancelled": control.cancelled.is_set(), **metrics})
        if not control.cancelled.is_set() and achieved < target:
            raise RuntimeError(f"Stockfish stopped at depth {achieved}, requested {target}")

    if strategy == "exhaustive" or depth <= 4:
        yield from search(depth, len(legal))
    else:
        # Match the upstream plan: all moves shallow; four likely human moves
        # and the actual played move to 14; top four engine moves to full depth;
        # then finish human/played candidates at full depth.
        screen_depth = max(4, min(10, depth - 6))
        yield from search(screen_depth, len(legal), phase="screening")
        candidates = search_candidates(depth, strategy, options)
        candidates = [move for move in candidates if move in legal]
        def deepen(target, phase):
            for move in candidates:
                if rows.get(move, (0,))[0] < target:
                    yield from search(target, 1, [legal[move]], phase=phase)

        yield from deepen(min(14, depth), "human_mid")
        yield from search(depth, min(4, len(legal)), phase="engine_top")
        yield from deepen(depth, "human_final")
    if not control.cancelled.is_set():
        yield evaluation(depth, complete=True)


def bounded_evaluations(engine, board, depth, options=None, control=None, on_search=None, seconds=None):
    """Depth OR time limits, sharing one per-position wall-time allowance.

    Keep an inexpensive all-root screen for Maia's probability charts. Spend the
    useful time on one unrestricted PV and the played/human candidates, then on
    alternatives. Reuse the engine/hash, and never call a timed search cancelled
    or claim the requested depth was reached when it wasn't.
    """
    control, options = control or SearchControl(), options or {}
    legal = {move.uci(): move for move in board.legal_moves}
    candidates = search_candidates(depth, 'bounded', options)
    candidates = [move for move in candidates if move in legal]
    budget = resolve_search_seconds(seconds)
    started = time.perf_counter()
    deadline = started + budget
    rows, engine_moves = {}, []
    best_move, best_depth = None, 0
    timed_out = False

    def evaluation(complete=False):
        ranked_engine_moves = list(dict.fromkeys(([best_move] if best_move else []) + engine_moves))[:4]
        required = set(candidates + ranked_engine_moves)
        achieved = min((rows.get(move, (0,))[0] for move in required), default=0)
        reached = achieved >= depth and bool(best_move)
        return {"depth": best_depth, "candidate_min_depth": achieved, "target_depth": depth, "target_reached": reached,
                "complete": complete, "strategy": "bounded", "policy_version": BOUNDED_POLICY_VERSION,
                "budget_seconds": budget, "elapsed_seconds": time.perf_counter() - started,
                "stop_reason": ("depth" if reached else "time") if complete else "running",
                "time_limited": timed_out, "coverage_complete": len(rows) == len(legal),
                "candidate_moves": candidates, "engine_moves": ranked_engine_moves, "best_move": best_move,
                "best_depth": best_depth,
                "cp_vec": {move: row[1] for move, row in rows.items()},
                "mate_vec": {move: row[2] for move, row in rows.items() if row[2] is not None},
                "root_move_depth_vec": {move: row[0] for move, row in rows.items()}}

    def search(target, count, seconds, phase, moves=None):
        nonlocal best_move, best_depth, engine_moves, timed_out
        seconds = min(seconds, deadline - time.perf_counter())
        if control.cancelled.is_set() or seconds < .001:
            return
        achieved, metrics = 0, {}
        details = {"phase": phase, "target_depth": target, "multipv": count,
                   "root_move": moves[0].uci() if moves else None, "time_limit_seconds": seconds}
        if on_search:
            on_search({**details, "status": "started"})
        search_started = time.perf_counter()
        try:
            for snapshot in snapshots(engine, board, target, count, control, moves, metrics, seconds):
                achieved = min(row[0] for row in snapshot.values())
                for move, row in snapshot.items():
                    if move not in rows or row[0] >= rows[move][0]:
                        rows[move] = row
                if phase in ("engine_best", "engine_top"):
                    # An unrestricted search's best line is authoritative. A
                    # shallow screening outlier must not become the best move.
                    ordered = sorted(snapshot, key=lambda m: snapshot[m][1], reverse=board.turn == chess.WHITE)
                    if achieved >= best_depth:
                        best_move, best_depth = ordered[0], achieved
                    if phase == "engine_top":
                        engine_moves = list(dict.fromkeys([best_move, *ordered]))[:4]
                    elif not engine_moves:
                        engine_moves = [best_move]
                yield evaluation()
        finally:
            limited = achieved < target
            timed_out = timed_out or limited
            if on_search:
                on_search({**details, "status": "finished", "achieved_depth": achieved,
                           "wall_ms": (time.perf_counter() - search_started) * 1000,
                           "cancelled": control.cancelled.is_set(), "stop_reason": "time" if limited else "depth",
                           **metrics})

    yield from search(min(6, depth), len(legal), min(.30, budget * .05), "screening")
    yield from search(depth, 1, budget * .40, "engine_best")
    # Play the actual game move first; remaining candidates share a phase budget
    # and inherit time saved when an earlier candidate reaches depth quickly.
    candidate_deadline = min(deadline, time.perf_counter() + budget * .40)
    pending = [move for move in candidates if rows.get(move, (0,))[0] < depth]
    for index, move in enumerate(pending):
        share = (candidate_deadline - time.perf_counter()) / (len(pending) - index)
        yield from search(depth, 1, share, "human_final", [legal[move]])
    yield from search(depth, min(4, len(legal)), deadline - time.perf_counter(), "engine_top")
    if not control.cancelled.is_set():
        if not best_move or any(move not in rows for move in candidates):
            raise RuntimeError("Stockfish produced no usable evaluation within the position budget; increase the search time.")
        yield evaluation(complete=True)
