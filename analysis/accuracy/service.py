"""Native expected accuracy, absolute deviation, and observed game summaries."""
from __future__ import annotations

import math
from statistics import mean

import chess

from analysis.accuracy.performance import refresh_performance
from analysis.accuracy.evidence import RATINGS, collect_evidence, accuracy_moments

SCHEMA_VERSION = 1


def summarize_evidence(evidence, performance=None):
    """Summarize raw evidence through the same prepared-metric aggregation."""
    return _summarize_metrics({side: [
        {'forced': len(row['qualities']) == 1, 'accuracy': row['qualities'][row['played']],
         'maia': accuracy_moments(row)} for row in rows] for side, rows in evidence.items()}, performance)


def _summarize_metrics(evidence, performance):
    """Average position metrics within each side, then give sides equal weight."""
    players = {}
    for side in ('white', 'black'):
        all_rows = evidence[side]
        rows = [row for row in all_rows if not row['forced']]
        expected, deviation = [], []
        for rating in RATINGS:
            expected.append(mean(row['maia'][str(rating)]['expected_accuracy'] for row in rows) if rows else None)
            deviation.append(mean(row['maia'][str(rating)]['absolute_deviation'] for row in rows) if rows else None)
        players[side] = {
            'average_accuracy': mean(row['accuracy'] for row in rows) if rows else None,
            'lichess_accuracy': (performance or {}).get('players', {}).get(side, {}).get('accuracy'),
            'moves_used': len(rows), 'forced_moves_excluded': len(all_rows)-len(rows),
            'expected_accuracy': expected, 'absolute_deviation': deviation,
        }
    nonempty = [player for player in players.values() if player['moves_used']]
    shared = lambda key: [mean(player[key][i] for player in nonempty) if nonempty else None
                          for i in range(len(RATINGS))]
    return {
        'schema_version': SCHEMA_VERSION,
        'rating_scale': 'lb', 'rating_name': 'Lichess Blitz', 'ratings': list(RATINGS),
        'expected_accuracy': shared('expected_accuracy'),
        'absolute_deviation': shared('absolute_deviation'), 'players': players,
        'position_selection': 'Played positions with more than one legal move; full legal Maia distributions.',
        'pooling': 'Arithmetic average per side, then equal-weight average of nonempty sides.',
        'deviation_definition': 'Mean probability-weighted absolute deviation from each position expected accuracy; descriptive spread, not a confidence interval.',
    }


def summarize_moves(analysis):
    """Aggregate final move records without raw positions or an engine cache."""
    evidence = {'white': [], 'black': []}
    board = chess.Board(analysis['start_fen'])
    for row in analysis['moves']:
        side = 'white' if board.turn else 'black'
        move = chess.Move.from_uci(row['played']['move'])
        if row['side'] != side or move not in board.legal_moves:
            raise ValueError('Saved move side or legality does not match its position.')
        for rating in RATINGS:
            record = row.get('maia', {}).get(str(rating), {})
            for field in ('expected_accuracy', 'absolute_deviation'):
                value = record.get(field) if isinstance(record, dict) else None
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f'Missing prepared Maia {field}; regenerate the analysis from cached engine evidence.')
        evidence[side].append({**row, 'forced': board.legal_moves.count() == 1})
        board.push(move)
    return _summarize_metrics(evidence, analysis.get('performance'))


def refresh_saved_curve(analysis, *, output_dir=None):
    """Prepare raw measurements once, or aggregate an existing prepared analysis.

    Plotting consumes saved data. Raw positions are an internal pipeline input;
    ordinary saved-analysis refresh needs only the prepared move measurements.
    """
    refresh_performance(analysis)
    if 'positions' in analysis:
        from analysis.accuracy.by_move import record_move_accuracy
        record_move_accuracy(analysis, collect_evidence(analysis))
    analysis['accuracy_curve'] = summarize_moves(analysis)
    if output_dir is not None:
        from analysis.accuracy.figures import export_saved_figures
        export_saved_figures(analysis, output_dir)
    return analysis
