"""Deterministic Markdown performance tables from saved analysis statistics."""
from __future__ import annotations

import html
import re

from analysis.lichess_accuracy import PHASES, round_percent


def performance_table(performance):
    players = performance['players']
    def name(side):
        # PGN names are data, not Markdown/HTML instructions.
        value = html.escape(str(players[side]['name'])).replace('\n', ' ').replace('\r', ' ')
        return re.sub(r'([\\`*_{}\[\]()#+.!|<>-])', r'\\\1', value)
    lines = [f"| | {name('white')} (White) | {name('black')} (Black) |",
             '|---|---:|---:|']
    fields = [('Inaccuracies', 'inaccuracies'), ('Mistakes', 'mistakes'), ('Blunders', 'blunders'),
              ('Average centipawn loss', 'average_centipawn_loss'), ('Accuracy', 'accuracy'),
              *[(phase.title(), phase) for phase in PHASES]]
    for label, key in fields:
        values = []
        for side in ('white', 'black'):
            player = players[side]
            value = player['phases'].get(key) if key in PHASES else player[key]
            percent = key == 'accuracy' or key in PHASES
            values.append('—' if value is None else f'{round_percent(value)}%' if percent else str(value))
        lines.append(f'| {label} | ' + ' | '.join(values) + ' |')
    if any(p['moves_scored'] != p['moves_total'] for p in players.values()):
        lines += ['', 'Some positions lack evaluations; statistics cover only scored moves.']
    if any(v is None for p in players.values() for v in p['phases'].values()):
        lines += ['', '—: this phase has no Lichess accuracy result.']
    return '\n'.join(lines)


def insert_performance_snapshot(report, analysis):
    """Idempotently fill the local table; never ask the model to calculate it."""
    performance = analysis.get('performance')
    if not performance:
        return report
    start, end = '<!-- performance-statistics -->', '<!-- /performance-statistics -->'
    scale = analysis.get('played_elo_scale', {}).get('name', 'Lichess Blitz')
    scale_note = f'Played-level ratings: {html.escape(scale)}. Maia comparison levels: Lichess Blitz.'
    block = start + '\n' + scale_note + '\n\n' + performance_table(performance) + '\n' + end
    if start in report:
        return re.sub(re.escape(start) + r'(?:.*?' + re.escape(end) + r')?',
                      lambda _: block, report, flags=re.S)
    heading = re.search(r'^\s{0,3}#{2,6}\s+[^\n]*\b(?:performance|ratings?|elo)\b[^\n]*$', report, re.M | re.I)
    if not heading:
        return report
    return report[:heading.end()] + '\n\n' + block + '\n' + report[heading.end():]
