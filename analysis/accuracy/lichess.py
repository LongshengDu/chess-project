"""Python port of Lichess accuracy, advice and phase division (standard chess).

Sources: Lila f4da67dc45bba6769600db33af925c05ae21a8d0 (AGPL-3.0),
deps/lichess-lila/modules/analyse/src/main/{AccuracyPercent,AccuracyCP}.scala,
modules/tree/src/main/Advice.scala; scalachess 17.16.2 eval.scala/Divider.scala
and scalalib 11.10.12 Maths.scala. Scores are White-relative centipawns,
or '#N'/'#-N' mate notation; '#0' and '#-0' preserve the winner at checkmate.
"""
from __future__ import annotations

import math
import statistics

import chess

INITIAL_CP = 15
PHASES = ('opening', 'middlegame', 'endgame')
JUDGMENTS = ('inaccuracy', 'mistake', 'blunder')


def round_percent(value):
    """Scala/Java Math.round, rather than Python's ties-to-even round."""
    return None if value is None else math.floor(value + .5)


def mate_sign(score):
    if isinstance(score, str) and score.startswith('#'):
        int(score[1:])  # Reject malformed scores; keep the explicit sign of zero.
        return -1 if score.startswith('#-') else 1
    return None


def force_cp(score):
    """Lila forceAsCp followed by Cp.ceiled; mate saturates at +/-1000."""
    if score is None:
        return None
    sign = mate_sign(score)
    if sign is not None:
        return sign * 1000
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('Expected a finite centipawn score or signed mate.')
    return max(-1000, min(1000, score))


def winning_chances(cp):
    """Lichess advice uses UNCLIPPED CP on a -1..1 scale."""
    x = .00368208 * cp
    # Stable logistic, including unusually large engine scores.
    return 2 / (1 + math.exp(-max(-700, min(700, x)))) - 1


def win_percent(score):
    cp = force_cp(score)
    return None if cp is None else 50 + 50 * winning_chances(cp)


def move_accuracy(before, after):
    """Arguments are mover-relative winning percentages (0..100)."""
    if before is None or after is None:
        return None
    if after >= before:
        return 100.
    return max(0., min(100., 103.1668100711649 * math.exp(-.04354415386753951 *
                      (before-after)) - 3.166924740191411 + 1))


def judgment(before, after, white=True):
    """Lila CpAdvice then MateAdvice; at most one severity per move."""
    if before is None or after is None:
        return None
    bmate, amate = mate_sign(before), mate_sign(after)
    pov = 1 if white else -1
    if bmate is None and amate is None:
        delta = (winning_chances(before)-winning_chances(after)) * pov
        for threshold, label in ((.3, 'blunder'), (.2, 'mistake'), (.1, 'inaccuracy')):
            if delta >= threshold:
                return label
    elif bmate is None and amate * pov < 0:
        cp = before * pov
        return 'inaccuracy' if cp < -999 else 'mistake' if cp < -700 else 'blunder'
    elif bmate is not None and bmate * pov > 0 and (amate is None or amate * pov < 0):
        cp = after * pov if amate is None else 0
        return 'inaccuracy' if cp > 999 else 'mistake' if cp > 700 else 'blunder'
    return None


def move_metrics(scores, start_white=True, initial=INITIAL_CP):
    """One ordered, aligned result for every played half-move, including gaps."""
    sequence = [initial, *scores]
    rows = []
    for index, (before, after) in enumerate(zip(sequence, sequence[1:])):
        white = (index % 2 == 0) == start_white
        prev, nxt = win_percent(before), win_percent(after)
        if not white:
            prev, nxt = nxt, prev
        bcp, acp = force_cp(before), force_cp(after)
        rows.append({'side': 'white' if white else 'black',
                     'accuracy': move_accuracy(prev, nxt),
                     'centipawn_loss': None if bcp is None or acp is None else
                         max(0, (bcp-acp) * (1 if white else -1)),
                     'judgment': judgment(before, after, white)})
    return rows


def game_accuracy(scores, start_white=True, initial=INITIAL_CP):
    """Exact Lila window alignment, weight clamps and harmonic floor of 1.

    Return None when either side lacks valid weighted observations, as upstream.
    Missing evaluations stay in place: never compact the sequence across a gap.
    """
    if not scores:
        return None
    wins = [win_percent(s) for s in [initial, *scores]]
    size = max(2, min(8, len(scores)//10))
    windows = [wins[:size]] * (min(size, len(wins))-2)
    windows += [wins[i:i+size] for i in range(max(1, len(wins)-size+1))]
    by_side = {'white': [], 'black': []}
    for row, window in zip(move_metrics(scores, start_white, initial), windows):
        if row['accuracy'] is not None and all(x is not None for x in window):
            weight = max(.5, min(12., statistics.pstdev(window)))
            by_side[row['side']].append((row['accuracy'], weight))
    if any(not values for values in by_side.values()):
        return None
    result = {}
    for side, values in by_side.items():
        weighted = sum(a*w for a, w in values)/sum(w for _, w in values)
        harmonic = len(values)/sum(1/max(1., a) for a, _ in values)
        result[side] = (weighted+harmonic)/2
    return result


def _mixedness_score(y, white, black):
    # The scalachess 2x2-region lookup, ranks numbered 1 through 7.
    table = {
        (0, 1): 1+y, (0, 2): 8-y if y < 6 else 0,
        (0, 3): 10-y if y < 7 else 0, (0, 4): 10-y if y < 7 else 0,
        (1, 0): 9-y, (1, 1): 5+abs(4-y), (1, 2): 11-y, (1, 3): 12-y,
        (2, 0): y if y > 2 else 0, (2, 1): 3+y, (2, 2): 7,
        (3, 0): 2+y if y > 1 else 0, (3, 1): 4+y,
        (4, 0): 2+y if y > 1 else 0,
    }
    return table.get((white, black), 0)


def _mixedness(board):
    return sum(_mixedness_score(y+1,
        chess.popcount(board.occupied_co[chess.WHITE] & (0x0303 << (x+8*y))),
        chess.popcount(board.occupied_co[chess.BLACK] & (0x0303 << (x+8*y))))
        for y in range(7) for x in range(7))


def division(boards):
    """Boards include the initial position; boundaries are relative played plies."""
    middle = end = None
    for ply, board in enumerate(boards):
        pieces = chess.popcount(board.occupied & ~(board.kings | board.pawns))
        if end is None and pieces <= 6:
            end = ply
        if middle is None and (pieces <= 10 or
                chess.popcount(board.occupied_co[chess.WHITE] & chess.BB_RANK_1) < 4 or
                chess.popcount(board.occupied_co[chess.BLACK] & chess.BB_RANK_8) < 4 or
                _mixedness(board) > 150):
            middle = ply
    if middle is None:
        end = None
    if middle is not None and end is not None and middle >= end:
        middle = None
    return {'middle': middle, 'end': end}


def phase_of(ply, div):
    if div['end'] is not None and ply >= div['end']:
        return 'endgame'
    if div['middle'] is not None and ply >= div['middle']:
        return 'middlegame'
    return 'opening'


def phase_accuracies(scores, div, start_white=True):
    """Match Lila: each phase restarts gameAccuracy at Cp.initial (15).

    Do not substitute the evaluation immediately before a phase. That would be
    a different metric. Without a middle boundary, Lila returns no phase scores.
    """
    result = {'white': {}, 'black': {}}
    if div['middle'] is None:
        return result
    for phase in PHASES:
        indices = [i for i in range(len(scores)) if phase_of(i+1, div) == phase]
        if not indices:
            continue
        values = game_accuracy([scores[i] for i in indices],
                               (indices[0] % 2 == 0) == start_white)
        if values:
            for side in result:
                result[side][phase] = values[side]
    return result
