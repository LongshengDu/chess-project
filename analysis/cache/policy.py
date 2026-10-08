"""Reuse complete observations by requested search limits and engine semantics."""
from __future__ import annotations

import math

from analysis.cache.storage import identity
from analysis.stockfish_search import BOUNDED_POLICY_VERSION


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _time(request):
    if request.get('kind') == 'exploration':
        return request.get('movetime_ms')
    return request.get('budget_seconds') if request.get('strategy') == 'bounded' else request.get('max_budget_seconds', request.get('budget_seconds'))


def _family(request):
    if not isinstance(request, dict) or not isinstance(request.get('engine'), dict) or not request['engine']:
        return None
    if 'own_rating' in request:
        return 'maia' if (request.get('version') == 1
            and all(type(request.get(key)) is int and 600 <= request[key] <= 3000
                    for key in ('own_rating', 'opponent_rating'))) else None
    if type(request.get('depth')) is not int or request['depth'] < 1 or not _positive(_time(request)):
        return None
    if request.get('kind') == 'evaluation':
        candidates = request.get('candidate_moves')
        return 'evaluation' if (request.get('policy_version') == BOUNDED_POLICY_VERSION
            and request.get('strategy') in ('bounded', 'staged', 'exhaustive')
            and isinstance(candidates, list) and all(isinstance(move, str) for move in candidates)) else None
    if request.get('kind') == 'exploration':
        roots = request.get('root_moves')
        return 'exploration' if (request.get('version') == 2
            and type(request.get('multipv')) is int and request['multipv'] >= 1
            and (roots is None or isinstance(roots, list) and roots
                 and all(isinstance(move, str) for move in roots))) else None
    return None


def _same_profile(first, second):
    # Engine resources and unknown future options remain meaningful. Only these
    # requested limits/candidates may be promoted without changing the profile.
    variable = {'depth', 'budget_seconds', 'max_budget_seconds', 'candidate_moves', 'movetime_ms'}
    return ({key: value for key, value in first.items() if key not in variable}
            == {key: value for key, value in second.items() if key not in variable})


def _complete(request, result):
    """Storage validates legal moves/scores; selection also requires full coverage."""
    if not isinstance(result, dict):
        return False
    family = _family(request)
    if family == 'maia':
        return isinstance(result.get('policy'), dict) and type(result.get('value')) in (int, float)
    if family == 'evaluation':
        scores, depths = result.get('cp_vec'), result.get('root_move_depth_vec')
        if (result.get('complete') is not True or result.get('coverage_complete') is not True
                or result.get('strategy') != request['strategy']
                or not isinstance(scores, dict) or not isinstance(depths, dict)
                or set(scores) != set(depths) or not set(request['candidate_moves']) <= scores.keys()):
            return False
        if not scores:
            return type(result.get('terminal_cp')) in (int, float) and math.isfinite(result['terminal_cp'])
        engine_moves = result.get('engine_moves')
        return (result.get('best_move') in scores and isinstance(engine_moves, list)
                and bool(engine_moves) and set(engine_moves) <= scores.keys())
    if family == 'exploration':
        lines = result.get('lines')
        if not isinstance(lines, list):
            return False
        # PositionCache separately confirms an empty continuation is terminal.
        if not lines:
            return True
        roots = [line.get('pv_uci', [None])[0] if isinstance(line, dict) and line.get('pv_uci') else None for line in lines]
        return (len(lines) == request['multipv'] and None not in roots and len(set(roots)) == len(roots)
                and (request['root_moves'] is None or set(roots) <= set(request['root_moves'])))
    return False


def request_satisfies(stored_request, result, requested_request):
    """Reuse a complete observation when BOTH requested limits are sufficient.

    Achieved depth is diagnostic only. Validation of finite/legal scores belongs
    to PositionCache. Obsolete engine policies are available through exact pins,
    never active lookup; unknown namespaces use storage's exact-only fallback.
    """
    family = _family(stored_request)
    if (family is None or family != _family(requested_request)
            or not _same_profile(stored_request, requested_request) or not _complete(stored_request, result)):
        return False
    if family == 'maia':
        return stored_request == requested_request
    if stored_request['depth'] < requested_request['depth'] or _time(stored_request) < _time(requested_request):
        return False
    return family != 'evaluation' or set(requested_request['candidate_moves']) <= set(stored_request['candidate_moves'])


def promote_request(records, requested_request):
    """Join requested limits/candidates of validated history-compatible records.

    Callers filter records by namespace/history and validate new input limits.
    Historical larger limits survive promotion. This function never starts
    work, combines scores or clips a limit.
    """
    promoted = dict(requested_request)
    family = _family(promoted)
    if family not in ('evaluation', 'exploration'):
        return promoted
    compatible = [record for record in records
                  if isinstance(record, dict) and _family(record.get('request')) == family
                  and _same_profile(record['request'], promoted)
                  and _complete(record['request'], record.get('result'))]
    candidates = list(promoted.get('candidate_moves', []))
    budget = _time(promoted)
    for record in sorted(compatible, key=lambda item: quality_key(item['request'], item['result']), reverse=True):
        request = record['request']
        promoted['depth'] = max(promoted['depth'], request['depth'])
        budget = max(budget, _time(request))
        if family == 'evaluation':
            candidates.extend(request['candidate_moves'])
    if family == 'evaluation':
        promoted.update(budget_seconds=float(budget), max_budget_seconds=float(budget),
                        candidate_moves=list(dict.fromkeys(candidates)))
    else:
        promoted['movetime_ms'] = budget
    return promoted


def _history(board):
    return board.root().fen(), tuple(move.uci() for move in board.move_stack)


def _outcome(board):
    outcome = board.outcome(claim_draw=False)
    return None if outcome is None else (outcome.termination, outcome.winner)


def _recent_positions(board, length):
    walk, positions = board.copy(stack=True), []
    for _ in range(length):
        positions.append(walk.fen())
        if not walk.move_stack:
            break
        walk.pop()
    return tuple(positions)


def _reversible_history(board):
    walk, moves = board.copy(stack=True), []
    while walk.move_stack:
        move = walk.pop()
        if walk.is_irreversible(move):
            walk.push(move)
            break
        moves.append(move.uci())
    return walk.fen(), tuple(reversed(moves))


def history_compatible(board, stored_board, namespace, request):
    """Compare actual model input/history; never erase live repetition context."""
    if board.fen() != stored_board.fen():
        return False
    if _history(board) == _history(stored_board):
        return True
    family = _family(request)
    if family is None or _outcome(board) != _outcome(stored_board):
        return False
    engine = request['engine']
    if namespace == 'maia' and family == 'maia' and 'maia' in engine:
        length = engine.get('history_window')
        return (type(length) is int and length > 0
                and _recent_positions(board, length) == _recent_positions(stored_board, length))
    if namespace == 'stockfish' and family in ('evaluation', 'exploration') and 'stockfish' in engine:
        return _reversible_history(board) == _reversible_history(stored_board)
    return False


def quality_key(request, result):
    """Stable preference by requested depth/time, never achieved search depth."""
    family = _family(request)
    if family in ('evaluation', 'exploration'):
        return (request['depth'], _time(request), len(request.get('candidate_moves', [])), identity([request, result]))
    return (0, 0, 0, identity([request, result]))


def dominates(new_request, new_result, old_request, old_result):
    """Whether the new complete observation covers the old requested contract."""
    return request_satisfies(new_request, new_result, old_request)
