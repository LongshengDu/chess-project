"""Estimator-independent complete Maia move-quality evidence and validation."""
from __future__ import annotations

from copy import deepcopy
import statistics

import chess
import numpy as np

from analysis.lichess_accuracy import INITIAL_CP, move_accuracy, win_percent
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.policies import infer_policies, normalized_policies

SCHEMA_VERSION = 1
CONDITIONING = 'equal_opponent'


def validate_evidence(evidence):
    """Normalize compatible diagonal evidence; reject conditional legacy rows.

    Existing diagonal caches predate the schema_version field but explicitly
    identify equal-opponent conditioning and the complete rating grid. Those
    markers identify the same numeric contract, so they are accepted unchanged.
    """
    if not isinstance(evidence, dict) or set(evidence) != {'White', 'Black'}:
        raise ValueError('Rating evidence must contain both White and Black records.')
    result = {}
    for side, record in evidence.items():
        if (not isinstance(record, dict) or record.get('conditioning') != CONDITIONING
                or record.get('rating_grid') != list(RATINGS)
                or type(record.get('schema_version', SCHEMA_VERSION)) is not int
                or record.get('schema_version', SCHEMA_VERSION) != SCHEMA_VERSION):
            raise ValueError('Saved rating evidence is incompatible. Rerun full analysis (--analysis-only) to rebuild Maia evidence.')
        observations = record.get('observations')
        if not isinstance(observations, list):
            raise ValueError('Rating observations must be a list.')
        clean = []
        for row in observations:
            if not isinstance(row, dict):
                raise ValueError('Each rating observation must be a numeric evidence record.')
            qualities = row.get('qualities')
            if not isinstance(qualities, dict) or set(qualities) != {'position', 'root'}:
                raise ValueError('Complete position and root move qualities are required.')
            q = np.asarray(qualities['position'], dtype=float)
            index = row.get('played_index')
            if (q.ndim != 1 or len(q) == 0 or type(index) is not int or not 0 <= index < len(q)):
                raise ValueError('Played index must identify one legal move.')
            for values in qualities.values():
                values = np.asarray(values, dtype=float)
                if values.shape != q.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
                    raise ValueError('Move accuracies must be finite bounded percentages.')
            policy = np.asarray(row.get('maia_probabilities'), dtype=float)
            if (policy.shape != (len(RATINGS), len(q)) or not np.isfinite(policy).all()
                    or np.any(policy < 0) or not np.allclose(policy.sum(axis=1), 1., atol=1e-6, rtol=0)):
                raise ValueError('Complete normalized Maia policies are required.')
            weight = row.get('weight')
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
                raise ValueError('Rating observation weights must be positive finite numbers.')
            # No arbitrary metadata is passed into estimators; only chess evidence.
            clean.append({key: deepcopy(row[key]) for key in
                          ('played_index', 'qualities', 'weight', 'maia_probabilities')})
            if 'position_win_probability' in row:
                probability = row['position_win_probability']
                if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                        or not np.isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError('Before-position winning probability must be finite within [0, 1].')
                clean[-1]['position_win_probability'] = float(probability)
        result[side] = {'schema_version': SCHEMA_VERSION, 'conditioning': CONDITIONING,
                        'rating_grid': list(RATINGS), 'observations': clean}
        if 'actual_rating' in record:
            rating = record['actual_rating']
            if rating is not None and (isinstance(rating, bool) or not isinstance(rating, (int, float))
                    or not np.isfinite(rating) or not 0 <= rating <= 4000):
                raise ValueError('Actual account ratings must be finite within [0, 4000] or absent.')
            result[side]['actual_rating'] = None if rating is None else float(rating)
    return result


def evidence_compatible(evidence):
    try:
        validate_evidence(evidence)
        return True
    except (ValueError, TypeError, KeyError):
        return False


def evidence_matches_game(evidence, game):
    """Check cached observation coverage, side, legal ordering and played indices."""
    if not evidence_compatible(evidence):
        return False
    offsets = {'White': 0, 'Black': 0}
    board = game.board()
    for move in game.mainline_moves():
        side = 'White' if board.turn else 'Black'
        observations = evidence[side]['observations']
        if offsets[side] >= len(observations):
            return False
        row = observations[offsets[side]]
        legal = sorted(candidate.uci() for candidate in board.legal_moves)
        if len(row['qualities']['position']) != len(legal) or row['played_index'] != legal.index(move.uci()):
            return False
        offsets[side] += 1
        board.push(move)
    return all(offsets[side] == len(evidence[side]['observations']) for side in offsets)


def collect_evidence(game, records, evaluate, *, evaluate_many=None):
    """Combine all-legal White CP/mate scores and exact equal-rating policies."""
    if [row['move'] for row in records] != [move.uci() for move in game.mainline_moves()]:
        raise ValueError('Rating evidence does not match the game.')
    result = {side: {'schema_version': SCHEMA_VERSION, 'conditioning': CONDITIONING,
                     'rating_grid': list(RATINGS), 'observations': []} for side in ('White', 'Black')}
    if not records:
        return result
    boards, after_scores, policies, missing = [], [], [], []
    board = game.board()
    for index, row in enumerate(records):
        choices = {move.uci() for move in board.legal_moves}
        if set(row['scores']) != choices:
            raise ValueError('Rating evidence requires scores for every legal move.')
        boards.append(board.copy(stack=True))
        policies.append(normalized_policies(row['policies'], choices) if row.get('policies') is not None else None)
        if policies[-1] is None:
            missing.append(index)
        board.push_uci(row['move'])
        score = records[index+1]['position_score'] if index+1 < len(records) else row['scores'][row['move']]
        if board.is_checkmate():
            score = '#-0' if board.turn else '#0'
        elif board.is_game_over(claim_draw=False):
            score = 0
        after_scores.append(score)
    for index, policy in zip(missing, infer_policies([boards[i] for i in missing], evaluate, evaluate_many=evaluate_many), strict=True):
        policies[index] = policy
    initial = INITIAL_CP if game.board().board_fen() == chess.Board().board_fen() else records[0]['position_score']
    wins = [win_percent(score) for score in [initial, *after_scores]]
    size = max(2, min(8, len(records)//10))
    windows = [wins[:size]]*(min(size, len(wins))-2)
    windows += [wins[i:i+size] for i in range(max(1, len(wins)-size+1))]
    for index, (board, row, policy) in enumerate(zip(boards, records, policies, strict=True)):
        side = 'White' if board.turn else 'Black'
        choices = sorted(row['scores'])
        before = win_percent(row['position_score'])
        qualities, position_qualities = [], []
        for move in choices:
            child = board.copy(stack=False)
            child.push_uci(move)
            after = row['scores'][move]
            if child.is_checkmate():
                after = '#-0' if child.turn else '#0'
            elif child.is_game_over(claim_draw=False):
                after = 0
            nxt = win_percent(after)
            qualities.append(move_accuracy(before, nxt) if board.turn else move_accuracy(nxt, before))
            position_after = win_percent(after_scores[index]) if move == row['move'] else nxt
            position_before = wins[index]
            position_qualities.append(move_accuracy(position_before, position_after) if board.turn
                                      else move_accuracy(position_after, position_before))
        result[side]['observations'].append({
            'played_index': choices.index(row['move']),
            'position_win_probability': before/100.,
            'qualities': {'root': qualities, 'position': position_qualities},
            'weight': max(.5, min(12., statistics.pstdev(windows[index]))),
            'maia_probabilities': [[policy[rating][move] for move in choices] for rating in RATINGS]})
    return validate_evidence(result)
