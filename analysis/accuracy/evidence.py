"""Complete legal-move accuracy and Maia probability measurements."""
from __future__ import annotations

import math

import chess

from analysis.accuracy.lichess import INITIAL_CP, move_accuracy, win_percent
from analysis.position_evaluation import centipawns

RATINGS = tuple(range(600, 2601, 100))


def expected_accuracy(qualities, policy):
    """Probability-weighted accuracy over one position's complete legal moves."""
    return max(0., min(100., math.fsum(policy[move]*qualities[move] for move in qualities)))


def accuracy_moments(row):
    """Prepared accuracy and absolute deviation at every native Maia anchor."""
    quality = row['qualities']
    moments = {}
    for rating in RATINGS:
        policy = row['policies'][rating]
        average = expected_accuracy(quality, policy)
        moments[str(rating)] = {'expected_accuracy': average,
            'absolute_deviation': math.fsum(policy[m]*abs(quality[m]-average) for m in quality)}
    return moments


def normalized_policies(policies, legal):
    """Require all native anchors and all legal moves, including rare moves."""
    if not isinstance(policies, dict) or {str(r) for r in policies} != {str(r) for r in RATINGS}:
        raise ValueError('Accuracy curve requires every Maia anchor from 600 to 2600.')
    result = {}
    for rating in RATINGS:
        policy = policies.get(rating, policies.get(str(rating)))
        if not isinstance(policy, dict) or set(policy) != set(legal):
            raise ValueError('Accuracy curve requires a complete legal policy at each Maia anchor.')
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
               for p in policy.values()):
            raise ValueError('Maia probabilities must be finite and nonnegative.')
        total = sum(policy.values())
        if not .99 <= total <= 1.01:
            raise ValueError('The full legal Maia policy must sum to one.')
        result[rating] = {move: float(p/total) for move, p in policy.items()}
    return result


def collect_evidence(analysis):
    """Reconstruct aligned move qualities from a complete saved game analysis.

    Observed moves use the next unrestricted position score (or the last played
    score). Alternatives use their complete root scores. This matches the game
    performance metric's position sequence and retains every legal alternative.
    """
    rows, positions = analysis['moves'], analysis.get('positions', [])
    if len(positions) != len(rows)+1:
        raise ValueError('Accuracy curve needs complete saved positions; rerun game analysis using the engine cache.')
    result = {'white': [], 'black': []}
    board = chess.Board(analysis['start_fen'])
    initial = INITIAL_CP if board.board_fen() == chess.Board().board_fen() else (
        centipawns(rows[0]['position_eval']) if rows else INITIAL_CP)
    before_score = initial
    for index, row in enumerate(rows):
        position = positions[index]
        if chess.Board(position['fen']).fen() != board.fen():
            raise ValueError('Saved positions do not match the played game.')
        legal = {move.uci() for move in board.legal_moves}
        scores = position.get('stockfish', {}).get('cp_vec', {})
        if set(scores) != legal:
            raise ValueError('Accuracy curve needs Stockfish scores for all legal moves; rerun game analysis using the engine cache.')
        maia = position.get('maia', {})
        policies = ({rating: {next(iter(legal)): 1.} for rating in RATINGS} if len(legal) == 1 else
                    normalized_policies({r: maia.get(f'maia_kdd_{r}', {}).get('policy') for r in RATINGS}, legal))
        side = 'white' if board.turn else 'black'
        if row['side'] != side or row['played']['move'] not in legal:
            raise ValueError('Saved move side or legality does not match its position.')
        played_score = centipawns(rows[index+1]['position_eval'] if index+1 < len(rows) else row['played']['eval'])
        before = win_percent(before_score)
        qualities = {}
        mate_scores = position['stockfish'].get('mate_vec', {})
        for move in sorted(legal):
            child = board.copy(stack=False)
            child.push_uci(move)
            score = scores[move]
            if move in mate_scores:
                mate = mate_scores[move] * (1 if board.turn else -1)
                score = f'#{mate}'
            if move == row['played']['move']:
                score = played_score
            if child.is_checkmate():
                score = '#-0' if child.turn else '#0'
            elif child.is_game_over(claim_draw=False):
                score = 0
            after = win_percent(score)
            qualities[move] = move_accuracy(before, after) if board.turn else move_accuracy(after, before)
            if move == row['played']['move']:
                played_score = score
        result[side].append({'played': row['played']['move'], 'qualities': qualities, 'policies': policies})
        before_score = played_score
        board.push_uci(row['played']['move'])
    return result
