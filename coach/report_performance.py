"""Deterministic Markdown performance tables from saved analysis statistics."""
from __future__ import annotations

import html
import re

from analysis.accuracy.lichess import PHASES, round_percent


def performance_table(performance, game, accuracy_curve=None):
    players = performance['players']
    def name(side):
        # Game names are data, not Markdown/HTML instructions.
        value = html.escape(str(game[side]['name'] or side.title())).replace('\n', ' ').replace('\r', ' ')
        return re.sub(r'([\\`*_{}\[\]()#+.!|<>-])', r'\\\1', value)
    lines = [f"| | {name('white')} (White) | {name('black')} (Black) |",
             '|---|---:|---:|']
    if any(p['moves_scored'] != p['moves_total'] for p in players.values()):
        counts = [f"{players[side]['moves_scored']} / {players[side]['moves_total']}" for side in ('white', 'black')]
        lines.append('| Moves analyzed | ' + ' | '.join(counts) + ' |')
    fields = [('Inaccuracies', 'inaccuracies'), ('Mistakes', 'mistakes'), ('Blunders', 'blunders'),
              ('Average centipawn loss', 'average_centipawn_loss'),
              *([('Arithmetic average accuracy', 'average_accuracy')] if accuracy_curve else []),
              ('Lichess accuracy', 'accuracy'),
              *[(phase.title(), phase) for phase in PHASES]]
    for label, key in fields:
        values = []
        for side in ('white', 'black'):
            player = players[side]
            if key == 'average_accuracy':
                value = accuracy_curve.get('players', {}).get(side, {}).get(key)
            else:
                value = player['phases'].get(key) if key in PHASES else player[key]
            percent = key == 'accuracy' or key in PHASES
            values.append('—' if value is None else f'{value:.2f}%' if key == 'average_accuracy'
                          else f'{round_percent(value)}%' if percent else str(value))
        lines.append(f'| {label} | ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def insert_performance_snapshot(report, analysis):
    """Insert the local table once as plain Markdown beneath its report heading."""
    performance = analysis.get('performance')
    if not performance:
        return report
    curve = analysis.get('accuracy_curve')
    block = performance_table(performance, analysis['game'], curve)
    heading = re.search(r'^[ \t]{0,3}(#{2,6})[ \t]+[^\n]*\b(?:performance|ratings?|elo)\b[^\n]*$', report, re.M | re.I)
    if not heading:
        return report
    following = report[heading.end():]
    next_heading = re.search(r'^[ \t]{0,3}#{1,' + str(len(heading[1])) + r'}[ \t]+', following, re.M)
    section = following[:next_heading.start()] if next_heading else following
    if block in section:
        return report
    return report[:heading.end()] + '\n\n' + block + '\n' + report[heading.end():]
