"""Local, reusable Lichess statistics and saved quality hints."""
from __future__ import annotations

import statistics

import chess

from analysis.position_evaluation import centipawns
from analysis.player_rating.scale import native_player_rating
from analysis.cache import identity

from analysis.lichess_accuracy import (INITIAL_CP, JUDGMENTS, PHASES, division,
    game_accuracy, move_metrics, phase_accuracies, phase_of, round_percent)

PERFORMANCE_VERSION = 1


def _refresh_rating_flags(analysis):
    """Refresh level-sensitive labels; preserve tactical/complete-root evidence."""
    from analysis.move_hints import LABELS, RATING_LABELS, rating_flags
    levels = {side: (native_player_rating(analysis, side)
                    or native_player_rating(analysis, side, fitted=True) or 1500)
              for side in ('white', 'black')}
    signature = identity(['native-rating-hints-v1', levels])
    if analysis['performance'].get('rating_context_signature') == signature:
        return
    for row in analysis['moves']:
        flags = set(row.get('flags', ()))
        flags = flags - RATING_LABELS | rating_flags(row, levels[row['side']], flags)
        row['flags'] = [label for label in LABELS if label in flags]
    analysis['performance']['rating_context_signature'] = signature


def refresh_performance(analysis):
    """Upgrade saved statistics/quality hints locally, without any engine or LLM.

    Preserve other existing hints: their all-legal-root evidence is not necessarily
    available in an older saved file. Summaries/tools do not call this function.
    """
    if analysis.get('performance', {}).get('version') == PERFORMANCE_VERSION:
        _refresh_rating_flags(analysis)
        return analysis
    from analysis.move_hints import LABELS

    rows = analysis['moves']
    board = chess.Board(analysis['start_fen'])
    start_white = board.turn
    boards = [board.copy(stack=False)]
    for row in rows:
        board.push_uci(row['played']['move'])
        boards.append(board.copy(stack=False))
    div = division(boards)
    # An unrestricted search at the next played position is stronger evidence
    # than the previous root-restricted candidate search. Use it consistently
    # for both game accuracy and severity; the final move has no next root.
    scores = [centipawns(rows[i+1]['position_eval'] if i+1 < len(rows)
                        else row['played']['eval']) for i, row in enumerate(rows)]
    for i, after in enumerate(boards[1:]):
        if after.is_checkmate():
            scores[i] = '#-0' if after.turn else '#0'
        elif after.is_game_over(claim_draw=False):
            scores[i] = 0
    initial = INITIAL_CP
    # Unlike a standard opening, an arbitrary FEN can begin already lost.
    # Use its actual evaluation rather than inventing a +0.15 starting score.
    if boards[0].board_fen() != chess.Board().board_fen() and rows:
        initial = centipawns(rows[0]['position_eval'])
    metrics = move_metrics(scores, start_white, initial)
    overall = game_accuracy(scores, start_white, initial) or {}
    phases = phase_accuracies(scores, div, start_white)
    for row, quality in zip(rows, metrics, strict=True):
        row['stage'] = phase_of(row['ply'], div)
        row['accuracy'] = quality['accuracy']
        row['centipawn_loss'] = quality['centipawn_loss']
        flags = set(row.get('flags', [])) - set(JUDGMENTS) - {'natural_but_bad'}
        if quality['judgment']:
            flags.add(quality['judgment'])
        row['flags'] = [label for label in LABELS if label in flags]
    players = {}
    for side in ('white', 'black'):
        moves = [q for q in metrics if q['side'] == side]
        losses = [q['centipawn_loss'] for q in moves if q['centipawn_loss'] is not None]
        players[side] = {
            'name': analysis['headers'].get(side.title(), side.title()),
            'inaccuracies': sum(q['judgment'] == 'inaccuracy' for q in moves),
            'mistakes': sum(q['judgment'] == 'mistake' for q in moves),
            'blunders': sum(q['judgment'] == 'blunder' for q in moves),
            'average_centipawn_loss': round_percent(statistics.mean(losses)) if losses else None,
            'accuracy': overall.get(side),
            'phases': {phase: phases[side].get(phase) for phase in PHASES},
            'moves_scored': len(losses), 'moves_total': len(moves),
        }
    analysis['performance'] = {'version': PERFORMANCE_VERSION, 'method': 'lichess',
        'evaluation_source': 'next_position_or_final_played', 'initial_cp': initial,
        'division': div, 'players': players}
    _refresh_rating_flags(analysis)
    return analysis
