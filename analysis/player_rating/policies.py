"""Batched full-history, equal-rating Maia policies shared by all estimators."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.player_rating.parameters import RATINGS
from analysis.cache import write_json
from analysis.settings import CONFIG


def normalized_policies(policies, legal):
    """Validate complete legal distributions at every supported Maia rating."""
    if not isinstance(policies, dict) or len(policies) != len(RATINGS) or {str(r) for r in policies} != {str(r) for r in RATINGS}:
        raise ValueError('Rating evidence needs every exact diagonal Maia rating.')
    result = {}
    for rating in RATINGS:
        policy = policies.get(rating, policies.get(str(rating)))
        if not isinstance(policy, dict) or set(policy) != set(legal):
            raise ValueError('Rating evidence needs a complete legal policy at each rating.')
        probabilities = np.asarray(list(policy.values()), dtype=float)
        total = probabilities.sum()
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not .99 <= total <= 1.01:
            raise ValueError('Invalid equal-opponent Maia probability distribution.')
        result[rating] = {move: float(probability/total) for move, probability in policy.items()}
    return result


def infer_policies(boards, evaluate, *, evaluate_many=None):
    """Infer complete policies; forced moves do not require a model call."""
    policies = [None]*len(boards)
    pending = []

    def flush():
        requests = [(boards[index], list(RATINGS), list(RATINGS)) for index in pending]
        batches = evaluate_many(requests) if evaluate_many else [evaluate(*request) for request in requests]
        if len(batches) != len(pending):
            raise ValueError('Incomplete equal-opponent Maia position batch.')
        for index, entries in zip(pending, batches, strict=True):
            if len(entries) != len(RATINGS):
                raise ValueError('Incomplete equal-opponent Maia rating policies.')
            policies[index] = normalized_policies(
                {rating: entry['policy'] for rating, entry in zip(RATINGS, entries, strict=True)},
                {move.uci() for move in boards[index].legal_moves})
        pending.clear()

    for index, board in enumerate(boards):
        legal = list(board.legal_moves)
        if len(legal) == 1:
            policies[index] = {rating: {legal[0].uci(): 1.} for rating in RATINGS}
            continue
        if not legal:
            raise ValueError('A played position must have at least one legal move.')
        if pending and (len(pending)+1)*len(RATINGS) > CONFIG['MAIA']['BATCH_SIZE']:
            flush()
        pending.append(index)
    if pending:
        flush()
    return policies


def prepare_rating_policies(game, records, evaluate, directory, model_signature, *, evaluate_many=None):
    """Attach complete ``policies[rating][uci]`` maps without fitting a rating.

    Evidence is independent of the selected estimator, interval probability,
    account ratings and reference labels. Existing diagonal policy caches use
    this same content signature and can be reused unchanged.
    """
    moves = [move.uci() for move in game.mainline_moves()]
    if [row['move'] for row in records] != moves:
        raise ValueError("Rating policies do not match this game's mainline.")
    boards, board = [], game.board()
    for row in records:
        if row.get('side', 'White' if board.turn else 'Black') != ('White' if board.turn else 'Black'):
            raise ValueError('Rating record has the wrong player side.')
        boards.append(board.copy(stack=True))
        board.push_uci(row['move'])
    signature = [1, 'equal_opponent', model_signature, game.board().fen(), moves, RATINGS]
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(directory)/f'diagonal-policies-{key}.json'
    policies = None
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding='utf-8'))
            if not isinstance(cached, list) or len(cached) != len(records):
                raise ValueError("Cached rating policies do not match this game's mainline.")
            policies = [normalized_policies(policy, {move.uci() for move in board.legal_moves})
                        for policy, board in zip(cached, boards, strict=True)]
        except (ValueError, TypeError):
            policies = None  # Full preparation rebuilds malformed/incomplete caches.
    if policies is None:
        policies = infer_policies(boards, evaluate, evaluate_many=evaluate_many)
        write_json(cache, policies)
    for row, policy in zip(records, policies, strict=True):
        row['policies'] = policy
