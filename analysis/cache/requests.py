"""Engine-specific identities shared by analysis and interactive requests."""
from __future__ import annotations

import math

from analysis.stockfish_search import BOUNDED_POLICY_VERSION, search_candidates


def maia_signature(signature):
    """Maia inference does not depend on Stockfish resources or its executable."""
    return {key: value for key, value in (signature or {}).items()
            if key not in ('stockfish', 'threads', 'hash_mb', 'cache_version')}


def stockfish_signature(signature):
    """Stockfish evidence does not depend on Maia's checkpoint or device."""
    signature = signature or {}
    if 'stockfish' in signature:
        return {key: signature[key] for key in ('stockfish', 'threads', 'hash_mb') if key in signature}
    return {key: value for key, value in signature.items()
            if key not in ('maia', 'history_window', 'device', 'cache_version')}


def maia_request(signature, own, opponent):
    return {'version': 1, 'engine': maia_signature(signature),
            'own_rating': own, 'opponent_rating': opponent}


def validate_maia_prediction(board, prediction):
    """Shared cached policies must contain every legal move and finite probabilities."""
    policy = prediction.get('policy') if isinstance(prediction, dict) else None
    legal = set() if board.is_game_over(claim_draw=False) else {move.uci() for move in board.legal_moves}
    if (not isinstance(policy, dict) or set(policy) != legal
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in policy.values())
            or (legal and not .99 <= math.fsum(policy.values()) <= 1.01)):
        raise RuntimeError('Maia returned an invalid legal-move distribution.')
    value = prediction.get('value')
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise RuntimeError('Maia returned an invalid expected score.')


def stockfish_initial_request(signature, depth, budget, max_budget, strategy, options):
    """Identify the executed search, including its effective allowance and ceiling."""
    # Depth-driven strategies ignore the nominal bounded budget; only their
    # maximum watchdog allowance affects execution in web and session callers.
    budget = budget if strategy == 'bounded' else max_budget
    # The bounded search and its watchdog use budget, not the request ceiling.
    # Keep the real configured ceiling in result metadata, not engine identity.
    max_budget = budget if strategy == 'bounded' else max_budget
    return {'kind': 'evaluation', 'policy_version': BOUNDED_POLICY_VERSION,
            'engine': stockfish_signature(signature), 'depth': depth,
            'budget_seconds': float(budget), 'max_budget_seconds': float(max_budget),
            'strategy': strategy, 'candidate_moves': search_candidates(depth, strategy, options)}


def stockfish_request_metadata(request):
    """Describe the executed request for display without altering its measurement."""
    return {'target_depth': request['depth'], 'policy_version': request['policy_version'],
            'strategy': request['strategy'], 'candidate_moves': list(request['candidate_moves']),
            'budget_seconds': request['budget_seconds'], 'max_budget_seconds': request['max_budget_seconds']}


def stockfish_exploration_request(signature, time_ms, depth, multipv, root_moves):
    """One executed root search; PV display length does not change engine work."""
    return {'kind': 'exploration', 'version': 2, 'engine': stockfish_signature(signature),
            'movetime_ms': round(time_ms), 'depth': depth, 'multipv': multipv,
            'root_moves': sorted(set(root_moves)) if root_moves else None}
