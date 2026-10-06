"""Cheap attention labels, not verified tactical or psychological conclusions."""
from __future__ import annotations

import math
import chess

from analysis.lichess_accuracy import judgment
from analysis.player_rating.scale import native_player_rating

LABELS = (
    'inaccuracy', 'mistake', 'blunder', 'missed_win', 'winning_to_losing',
    'saved_game', 'critical_move', 'only_move', 'missed_opponent_error',
    'punished_opponent_error', 'conversion_error', 'sacrifice', 'natural_but_bad',
    'higher_elo_improvement', 'above_level_move', 'stronger_maia_converges',
    'engine_only_move', 'human_consensus', 'forcing_best_move', 'mate_found',
    'missed_mate', 'allowed_mate',
)
RATING_LABELS = frozenset(('natural_but_bad', 'higher_elo_improvement', 'above_level_move'))


def mate_for(value, side):
    if not isinstance(value, str) or not value.startswith('#'):
        return False
    return ('black' if value.startswith('#-') else 'white') == side


def expected_score(value, side):
    """Smooth CP proxy, not a calibrated Elo-specific expected-points model."""
    if isinstance(value, str):
        return float(mate_for(value, side))
    white = 1 / (1 + math.exp(-max(-30., min(30., .368208 * value))))
    return white if side == 'white' else 1-white


def outcome(score):
    return 1 if score >= .75 else -1 if score <= .25 else 0


def near_best(best, played, side):
    # A large numeric advantage is not a substitute for preserving forced mate.
    if mate_for(best, side):
        return mate_for(played, side)
    return expected_score(best, side) - expected_score(played, side) <= .02


def probability(candidate, elo):
    """Interpolate exported Maia probabilities at actual Elo; clamp to the grid."""
    curve = candidate['maia_p']
    ratings = sorted(map(int, curve))
    lower = max((r for r in ratings if r <= elo), default=ratings[0])
    upper = min((r for r in ratings if r >= elo), default=ratings[-1])
    if lower == upper:
        return curve[str(lower)]
    weight = (elo-lower)/(upper-lower)
    return curve[str(lower)]*(1-weight) + curve[str(upper)]*weight


def material(board, color):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(v*(len(board.pieces(p, color))-len(board.pieces(p, not color))) for p, v in values.items())


def sacrifice_hint(row, reply=None):
    """A moved piece is offered, or another piece is actually given up.

    Acceptance must cost >=2 material points before recovery and leave at
    least one point sacrificed after any immediate legal recovery. This keeps
    bishop-for-two-pawns offers but excludes ordinary exchanges. Check legal
    captures even when the opponent declined (or the game ended at the offer).
    Soundness is gated by move_flags; this is still an attention hint, not proof
    of intent or a verified continuation after every possible acceptance.
    """
    board = chess.Board(row['fen'])
    color, initial = board.turn, material(board, board.turn)
    move = chess.Move.from_uci(row['played']['move'])
    board.push(move)
    actual_reply = chess.Move.from_uci(reply['played']['move']) if reply else None
    for response in board.generate_legal_captures():
        # Do not label unrelated quiet moves just because a previously offered
        # piece remains hanging. Retain accepted sacrifices of another piece.
        if response.to_square != move.to_square and response != actual_reply:
            continue
        victim = board.piece_at(response.to_square)
        if victim is None or victim.color != color or victim.piece_type in (chess.PAWN, chess.KING):
            continue
        accepted = board.copy(stack=False); accepted.push(response)
        if material(accepted, color) > initial-2:
            continue
        for recovery in accepted.legal_moves:
            child = accepted.copy(stack=False); child.push(recovery)
            if material(child, color) >= initial:
                break
        else:
            return True
    return False


def rating_flags(row, level, judgments):
    """Re-evaluate only level-dependent labels from saved candidate probabilities."""
    candidates, side = row['candidate_moves'], row['side']
    actual = next((c for c in candidates if c['move'] == row['played']['move']), None)
    if actual is None:
        return set(row.get('flags', ())) & RATING_LABELS
    flags, base = set(), probability(actual, level)
    stronger = sorted({min(2600, level+step) for step in (200, 400, 600) if min(2600, level+step) > level})
    def higher(candidate):
        return sum(probability(candidate, rating) for rating in stronger)/len(stronger) if stronger else probability(candidate, level)
    if set(judgments) & {'mistake', 'blunder'} and base >= .10:
        flags.add('natural_but_bad')
    if stronger and near_best(row['position_eval'], row['played']['eval'], side) and higher(actual) >= max(.15, base+.10, base*1.5):
        flags.add('above_level_move')
    after = expected_score(row['played']['eval'], side)
    for candidate in candidates:
        if stronger and candidate['move'] != actual['move'] and expected_score(candidate['eval'], side)-after >= .05 \
                and higher(candidate) >= max(.15, probability(candidate, level)+.10, probability(candidate, level)*1.5):
            flags.add('higher_elo_improvement')
    return flags


def move_flags(row, level, *, previous=None, reply=None, all_moves=None):
    side = row['side']
    other = 'black' if side == 'white' else 'white'
    best, played = row['position_eval'], row['played']['eval']
    before, after = expected_score(best, side), expected_score(played, side)
    loss = max(0., before-after)
    good = near_best(best, played, side)
    flags = set()
    allowed = mate_for(played, other) and not mate_for(best, other)
    # Lichess advice: 5/10/15 percentage-point losses, plus mate transitions.
    # The completed game pass aligns these against the post-move score sequence.
    severity = judgment(best*100 if isinstance(best, (int, float)) else best,
                        played*100 if isinstance(played, (int, float)) else played, side == 'white')
    if severity:
        flags.add(severity)
    if before >= .75 and after <= .55:
        flags.add('missed_win')
    if outcome(before) == 1 and outcome(after) == -1:
        flags.add('winning_to_losing')
    # Before/after root scores normally cannot improve under best play. Positive
    # transitions may be finite-search discoveries; labels remain provisional.
    if before <= .25 and after >= .45:
        flags.update(('saved_game', 'critical_move'))
    if .45 <= before <= .55 and after >= .75:
        flags.add('critical_move')
    if before >= .75 and after >= .75 and loss >= .05:
        flags.add('conversion_error')
    board = chess.Board(row['fen'])
    candidates = row['candidate_moves']
    scores = all_moves if all_moves is not None else candidates
    # Never infer uniqueness from the truncated Maia candidate union.
    complete = {c['move'] for c in scores} == {m.uci() for m in board.legal_moves}
    unique = False
    if complete and len(scores) > 1 and before >= .45:
        retained = [c for c in scores if (mate_for(c['eval'], side) if mate_for(best, side)
                    else expected_score(c['eval'], side) >= before-.05)]
        unique = (len(retained) == 1 and retained[0]['move'] == row['played']['move']
                  and all(expected_score(c['eval'], side) <= before-.10
                          for c in scores if c['move'] != retained[0]['move']))
        if unique and good:
            flags.update(('only_move', 'critical_move'))
    if previous is not None and previous['side'] != side:
        old = expected_score(previous['position_eval'], side)
        opportunity = expected_score(previous['played']['eval'], side)
        if opportunity-old >= .10 and opportunity >= .55:
            if loss >= .10:
                flags.add('missed_opponent_error')
            elif good and after >= opportunity-.05:
                flags.add('punished_opponent_error')
    if good and after >= .45 and sacrifice_hint(row, reply):
        flags.add('sacrifice')
    flags.update(rating_flags(row, level, flags))
    for c in candidates:
        curve = c['maia_p']
        low = sum(curve[str(r)] for r in (1000, 1100, 1200))/3
        high = sum(curve[str(r)] for r in (2400, 2500, 2600))/3
        if high >= .50 and high >= low+.20 and all(
                row['maia'][str(r)] and row['maia'][str(r)][0]['move'] == c['move'] for r in (2400, 2500, 2600)):
            blocks = [sum(curve[str(r)] for r in range(start, start+300, 100))/3 for start in (1000, 1400, 1800, 2200)]
            if all(b >= a-.03 for a, b in zip(blocks, blocks[1:])):
                flags.add('stronger_maia_converges')
        if sum(curve[str(r)] >= .50 and row['maia'][str(r)][0]['move'] == c['move']
               for r in range(1000, 2601, 100)) >= 14:
            flags.add('human_consensus')
    # Strong preference requires all roots, and must name the actual engine best,
    # not an arbitrary near-best move. This is a position-level attention hint.
    if complete and len(scores) > 1:
        ranked = sorted(scores, key=lambda c: expected_score(c['eval'], side), reverse=True)
        top = next((c for c in candidates if c['move'] == ranked[0]['move']), None)
        if top and expected_score(ranked[0]['eval'], side)-expected_score(ranked[1]['eval'], side) >= .10 \
                and max(top['maia_p'][str(r)] for r in (2400, 2500, 2600)) < .05:
            flags.add('engine_only_move')
    move = chess.Move.from_uci(row['played']['move'])
    if good and (board.gives_check(move) or board.is_capture(move) or move.promotion or board.legal_moves.count() == 1):
        flags.add('forcing_best_move')
    if mate_for(played, side):
        flags.add('mate_found')
    if mate_for(best, side) and not mate_for(played, side):
        flags.add('missed_mate')
    if allowed:
        flags.add('allowed_mate')
    return [label for label in LABELS if label in flags]


def add_flags(analysis, all_scores):
    """Called only by analyze_game, while the complete legal-root scores exist."""
    rows = analysis['moves']
    for index, row in enumerate(rows):
        level = (native_player_rating(analysis, row['side'])
                 or native_player_rating(analysis, row['side'], fitted=True) or 1500)
        row['flags'] = move_flags(row, level, previous=rows[index-1] if index else None,
            reply=rows[index+1] if index+1 < len(rows) else None,
            all_moves=all_scores.get(row['ply']) if all_scores else None)
    return analysis
