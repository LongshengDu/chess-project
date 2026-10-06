"""Attach declared account ratings and before-position evaluations to evidence."""
from __future__ import annotations

import hashlib
import json

from analysis.lichess_accuracy import win_percent
from analysis.position_evaluation import centipawns, header_elo


SIDES = ('White', 'Black')


def account_ratings(headers, overrides=None):
    """Only actual Elo headers and explicit overrides may supply account ratings."""
    result = {side: header_elo(headers, side) for side in SIDES}
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides)-set(SIDES):
            raise ValueError('Rating overrides must use White and Black keys.')
        result.update(overrides)
    return result


def account_signature(ratings):
    """Keep account changes out of reusable engine-evidence cache identities."""
    values = [None if ratings.get(side) is None else float(ratings[side]) for side in SIDES]
    return hashlib.sha256(json.dumps(values, allow_nan=False).encode()).hexdigest()[:16]


def attach_context(evidence, ratings, position_scores=None):
    """Return a copy with optional actual ratings and aligned White-relative CP scores."""
    result = {}
    for side in SIDES:
        rows = evidence[side]['observations']
        scores = position_scores.get(side) if position_scores is not None else None
        if scores is not None and len(scores) != len(rows):
            raise ValueError('Before-position evaluations must cover every rating observation.')
        copied = [dict(row) for row in rows]
        if scores is not None:
            for row, score in zip(copied, scores, strict=True):
                probability = win_percent(score)
                if probability is None:
                    raise ValueError('A before-position evaluation is unavailable.')
                row['position_win_probability'] = probability/100.
        result[side] = {**evidence[side], 'actual_rating': ratings.get(side), 'observations': copied}
    return result


def game_scores(game, records):
    """Group canonical full-analysis scores by the side actually moving."""
    scores = {side: [] for side in SIDES}
    board = game.board()
    for row in records:
        scores['White' if board.turn else 'Black'].append(row['position_score'])
        board.push_uci(row['move'])
    return scores


def saved_ratings(analysis):
    """Explicit saved overrides take precedence; absent ratings stay absent."""
    return account_ratings(analysis.get('headers', {}), analysis.get('rating_account_overrides', {}))


def saved_context(evidence, analysis):
    scores = None
    if 'moves' in analysis:
        scores = {side: [] for side in SIDES}
        for row in analysis['moves']:
            scores[row['side'].title()].append(centipawns(row['position_eval']))
    return attach_context(evidence, saved_ratings(analysis), scores)
