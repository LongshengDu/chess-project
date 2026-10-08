"""Completeness and chess validity of reusable position measurements."""
from __future__ import annotations

import math

import chess

from analysis.cache.requests import validate_maia_prediction


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_measurement(board, namespace, request, result):
    """Reject incomplete engine evidence; leave other evidence categories extensible.

    A time-bounded search can be complete without attaining its depth ceiling.
    Completeness concerns the requested scored moves, not a depth-only target.
    """
    if not isinstance(request, dict):
        return
    if namespace == 'maia' and 'own_rating' in request:
        try:
            validate_maia_prediction(board, result)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        return
    if namespace != 'stockfish' or 'kind' not in request:
        return
    if not isinstance(result, dict) or result.get('available') is False:
        raise ValueError('Cannot cache unavailable Stockfish evidence.')
    terminal = board.is_game_over(claim_draw=False)
    legal = set() if terminal else {move.uci() for move in board.legal_moves}
    if request['kind'] == 'evaluation':
        scores, depths, mates = (result.get(key) for key in ('cp_vec', 'root_move_depth_vec', 'mate_vec'))
        engine_moves = result.get('engine_moves')
        if (result.get('complete') is not True or result.get('coverage_complete') is not True
                or not isinstance(scores, dict) or set(scores) != legal
                or not isinstance(depths, dict) or set(depths) != legal
                or not isinstance(mates, dict) or not set(mates) <= (legal | ({''} if terminal else set()))
                or any(not _number(value) for value in scores.values())
                or any(type(value) is not int or value < 0 for value in depths.values())
                or any(value is not None and type(value) is not int for value in mates.values())
                or not isinstance(engine_moves, list)
                or any(not isinstance(move, str) or move not in legal for move in engine_moves)
                or len(engine_moves) != len(set(engine_moves))
                or (not terminal and not engine_moves) or (terminal and bool(engine_moves))
                or (not terminal and result.get('best_move') not in legal)):
            raise ValueError('Cannot cache incomplete or invalid Stockfish move evaluations.')
        return
    if request['kind'] != 'exploration' or request.get('version') != 2:
        raise ValueError('Unsupported Stockfish measurement request.')
    from analysis.stockfish_exploration import validate_search_lines
    validate_search_lines(board, result)
    roots = [line['pv_uci'][0] for line in result['lines']]
    allowed = set(request['root_moves']) if request.get('root_moves') else legal
    count = 0 if terminal else min(request['multipv'], len(allowed))
    if len(roots) != count or not set(roots) <= allowed:
        raise ValueError('Cannot cache incomplete or mismatched Stockfish continuation roots.')
