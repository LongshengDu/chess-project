"""One-time normalization of saved continuation observations, without engine work."""
from __future__ import annotations

import math

import chess

from analysis.cache.requests import stockfish_exploration_request
from analysis.stockfish_exploration import exploration_board, validate_search_lines


def _complete_pv(board, pv, cutoff):
    if not isinstance(pv, list) or type(cutoff) is not int or cutoff < 1:
        return False
    walk = board.copy(stack=True)
    for uci in pv:
        move = chess.Move.from_uci(uci)
        if move not in walk.legal_moves:
            return False
        walk.push(move)
    # A cutoff-length record may have omitted later engine PV moves. Only a
    # terminal board proves that no unrecorded continuation could be required.
    return len(pv) < cutoff or walk.is_game_over(claim_draw=False)


def convert_exploration(board, request, result):
    """Return (searched board, new request, raw full PV) or None if insufficient.

    Supports old session exploration and backend continuation records only for
    this offline cleanup. Runtime readers never infer missing PV information.
    """
    try:
        if request.get('kind') == 'exploration' and request.get('version') == 2:
            validate_search_lines(board, result)
            if not board.is_game_over(claim_draw=False):
                if len(result['lines']) != request['multipv']:
                    return None
                roots = request.get('root_moves')
                if roots is not None and any(line['pv_uci'][0] not in roots for line in result['lines']):
                    return None
            return board, request, result
        if request.get('version') != 1 or request.get('kind') not in ('exploration', 'continuation'):
            return None
        if request['kind'] == 'continuation':
            move = request.get('move')
            root = exploration_board(board, move)
            ms = request['seconds'] * 1000
            count, roots = 1, None
            if result.get('bound'):
                return None
            pv = result['pv_uci']
            if move is not None:
                if not pv or pv[0] != move:
                    return None
                pv = pv[1:]
            if root.is_game_over(claim_draw=False) and pv:
                return None
            raw = {'lines': [] if root.is_game_over(claim_draw=False) else [{
                'cp': result['white_cp'], 'mate': result['white_mate'],
                'depth': result['depth'], 'pv_uci': pv}]}
            cutoff = 16
        else:
            root = board
            ms = request['movetime_ms']
            roots = request.get('root_moves')
            if roots is not None:
                if not isinstance(roots, list) or not roots:
                    return None
                roots = sorted(set(roots))
                if any(chess.Move.from_uci(uci) not in root.legal_moves for uci in roots):
                    return None
            if type(request['multipv']) is not int or request['multipv'] < 1:
                return None
            count = min(request['multipv'], len(roots) if roots else max(1, root.legal_moves.count()))
            raw = {'lines': [{key: line[key] for key in ('cp', 'mate', 'depth', 'pv_uci')}
                             for line in result['lines']]}
            cutoff = request['pv_plies']
        if (type(ms) not in (int, float) or not math.isfinite(ms) or ms < 1
                or type(request['depth']) is not int or request['depth'] < 1
                or not isinstance(request['engine'], dict)):
            return None
        validate_search_lines(root, raw)
        if not root.is_game_over(claim_draw=False):
            if len(raw['lines']) != count:
                return None
            for line in raw['lines']:
                if not _complete_pv(root, line['pv_uci'], cutoff):
                    return None
                if roots is not None and line['pv_uci'][0] not in roots:
                    return None
        canonical = stockfish_exploration_request(request['engine'], ms, request['depth'], count, roots)
        return root, canonical, raw
    except (KeyError, TypeError, ValueError):
        return None
