"""Bounded Stockfish continuation investigation with legal-line validation."""
from __future__ import annotations

import math

import chess
import chess.engine

from analysis.settings import CONFIG


def validate_exploration_seconds(seconds, maximum):
    if (isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum) or maximum < .05):
        raise ValueError('ANALYSIS.STOCKFISH_EXPLORATION.MAX_SEARCH_SECONDS must be finite and at least 0.05.')
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not .05 <= seconds <= maximum):
        raise ValueError(f'Exploration seconds must be between 0.05 and {maximum:g}')


def explore(engine, board, *, move=None, seconds=None, depth=None):
    """A legal principal variation, with an explicit depth OR time ceiling."""
    settings = CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION']
    seconds = settings['DEFAULT_SEARCH_SECONDS'] if seconds is None else seconds
    depth = settings['MAX_DEPTH'] if depth is None else depth
    validate_exploration_seconds(seconds, settings['MAX_SEARCH_SECONDS'])
    root = board.copy(stack=True)
    prefix = []
    if move:
        candidate = chess.Move.from_uci(move)
        if candidate not in root.legal_moves:
            raise ValueError("Exploration move must be legal")
        prefix.append(candidate)
        root.push(candidate)
    if root.is_game_over():
        score = chess.engine.PovScore(chess.engine.Mate(0) if root.is_checkmate() else chess.engine.Cp(0), root.turn)
        info = {"score": score, "pv": [], "depth": 0}
    else:
        info = None
        # The last UCI update at a time cutoff can be an aspiration bound or
        # an incomplete iteration. Keep the last exact score with its own PV.
        with engine.analysis(root, chess.engine.Limit(time=seconds, depth=depth), multipv=1,
                             info=chess.engine.INFO_SCORE | chess.engine.INFO_PV | chess.engine.INFO_BASIC) as search:
            for update in search:
                if (update.get("pv") and "score" in update and "depth" in update
                        and not update.get("upperbound") and not update.get("lowerbound")
                        and (info is None or update["depth"] >= info["depth"])):
                    info = update.copy()
        if info is None:
            raise RuntimeError("Stockfish produced no complete scored continuation within the exploration limit")
    walk = board.copy()
    ucis, sans = [], []
    for step in prefix + list(info.get("pv", []))[:16]:
        if step not in walk.legal_moves:
            raise RuntimeError("Engine returned an illegal continuation")
        sans.append(walk.san(step))
        ucis.append(step.uci())
        walk.push(step)
    white_score = info["score"].white()
    return {"fen": board.fen(), "move": move, "pv_uci": ucis, "pv_san": sans,
            "white_cp": white_score.score(), "white_mate": white_score.mate(),
            "depth": info.get("depth", 0), "target_depth": depth, "time_limit_seconds": seconds,
            "terminal": root.is_game_over(), "bound": bool(info.get("upperbound") or info.get("lowerbound"))}
