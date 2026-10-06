"""Render per-game shared curves and exact candidate decision maps as SVGs.

Candidate functions are reused without fitting new parameters. The renderer
never reads commercial reference headers or alters normal game outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Callable

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.player_rating.bayesian_shared_curve import SharedCurve
from analysis.settings import CONFIG
from tests.analysis import curve_anchor_candidates as anchor
from tests.analysis import curve_hierarchical_candidates as hierarchical
from tests.analysis import curve_regularized_candidates as regularized
from tests.analysis.compare_blitz_rating_methods import current_pgn_context, intersection
from tests.analysis.compare_rating_methods import load_cases


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/intuitive-curves/diagnostics'
SIDES = ('White', 'Black')
COLORS = {'White': '#16828a', 'Black': '#bd6334'}
METHODS = {
    'curve_anchor_cauchy': ('Common Cauchy prior', '#6d4bbe'),
    'hierarchical_affine_translated_prior': ('Hierarchical affine', '#298251'),
    'curve_regularized_cauchy': ('Regularized Cauchy', '#2d6eac'),
}
DEFAULT_METHODS = ('curve_anchor_cauchy', 'hierarchical_affine_translated_prior')


@dataclass(frozen=True)
class RatingMap:
    """One common accuracy-to-rating function, already prepared for this game."""

    method: str
    label: str
    color: str
    point: Callable[[float], float | None]


def prepare_plot(evidence, actual, methods=DEFAULT_METHODS):
    """Prepare reference-free curve data and exact reusable candidate functions."""
    if not methods or len(set(methods)) != len(methods) or set(methods)-set(METHODS):
        raise ValueError('Choose distinct supported candidate methods.')
    context = anchor.prepare(evidence, actual)
    observed = {side: row['average_accuracy']
                for side, row in zip(SIDES, context['curve']['players'], strict=True)}
    maps = []
    for method in methods:
        if not context['curve']['identifiable']:
            evaluate = lambda accuracy: None
        elif method == 'curve_anchor_cauchy':
            evaluate = lambda accuracy, prepared=context: anchor.point(accuracy, prepared, 'curve_anchor_cauchy')
        elif method == 'hierarchical_affine_translated_prior':
            prepared = hierarchical.prepare(evidence, actual)

            def evaluate(accuracy, prepared=prepared, ratings=dict(actual)):
                # Only replace the observation; all integrated context moments
                # and candidate formulas remain those of the research module.
                item = {**prepared, 'observed': dict.fromkeys(SIDES, accuracy)}
                return hierarchical.predict_prepared(item, ratings)['hierarchical_affine_translated_prior']['White']
        else:
            prepared = regularized.prepare(evidence, actual)
            evaluate = lambda accuracy, prepared=prepared, ratings=dict(actual): regularized.point(accuracy, prepared, ratings)
        maps.append(RatingMap(method, *METHODS[method], evaluate))
    return {
        'knots': context['curve']['monotone_expected_accuracy'],
        'sigma': float(np.sqrt(context['curve']['likelihood']['accuracy_variance'])),
        'observed': observed,
        'actual': dict(actual),
        'anchor': context['anchor'] if context['has_account'] else None,
        'identifiable': context['curve']['identifiable'],
        'maps': maps,
    }


def _number(value, decimals=0):
    if value is None:
        return '—'
    return f'{int(np.floor(value+.5)):,}' if decimals == 0 else f'{value:,.{decimals}f}'


def _crossing_label(crossing):
    point = crossing['estimate']
    if point is not None:
        suffix = '\n(extrapolated)' if crossing['intersection_status'] == 'extrapolated' else ''
        if not 200 <= point <= 3000:
            suffix += '\n(outside axis)'
        return _number(point)+suffix
    status = crossing['intersection_status']
    if status == 'above_support':
        return 'No crossing\nabove 0–3200 curve'
    if status == 'below_support':
        return 'No crossing\nbelow 0–3200 curve'
    if crossing['intersection_interval'] is not None:
        low, high = crossing['intersection_interval']
        return f'Nonunique\n{low:,.0f}–{high:,.0f}'
    return 'No observation'


def _table(axis, rows, columns):
    table = axis.table(cellText=rows, colLabels=columns, cellLoc='center',
                       bbox=(0., -.35, 1., .22))
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d8dde3')
        cell.set_linewidth(.5)
        if row == 0:
            cell.set_facecolor('#eef1f5')
        elif row <= len(SIDES):
            cell.get_text().set_color(COLORS[SIDES[row-1]])
    return table


def build_figure(game, data):
    """Build a figure from fixed numerical data and arbitrary RatingMap callables."""
    curve = SharedCurve(data['knots'])
    grid = np.linspace(200., 3000., 561)
    expected = curve(grid)
    crossings = {side: intersection(data['knots'], data['observed'][side]) for side in SIDES}
    figure = Figure(figsize=(16., 8.), facecolor='white')
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    figure.subplots_adjust(left=.06, right=.98, top=.86, bottom=.30, wspace=.24)
    figure.suptitle(f'{game} — shared accuracy curve and common rating decisions', fontsize=16, fontweight='bold', y=.97)
    figure.text(.5, .925, 'Arithmetic accuracy · same mapping for White and Black · no commercial ratings used',
                ha='center', fontsize=10, color='#4b5563')
    for axis in (left, right):
        axis.grid(alpha=.18)
        axis.spines[['top', 'right']].set_visible(False)
    left.axvspan(200., 600., color='#d8dee6', alpha=.45)
    left.axvspan(2600., 3000., color='#d8dee6', alpha=.45)
    left.fill_between(grid, np.clip(expected-data['sigma'], 0., 100.),
                      np.clip(expected+data['sigma'], 0., 100.),
                      color='#8b9bae', alpha=.17)
    left.plot(grid, expected, color='#334155', lw=2.)
    left.scatter(np.arange(600., 2601., 100.), data['knots'], color='#334155', s=13, zorder=3)
    left.set(xlim=(200., 3000.), ylim=(50., 100.), xlabel='Maia reference rating',
             ylabel='Arithmetic move accuracy (%)')
    left.set_xticks(np.arange(200, 3001, 400))
    left.set_title(f'Shared curve C(r) · conditional σ = {data["sigma"]:.3f} accuracy points\nMeasured Maia range: 600–2600', fontsize=11)
    for boundary in (600., 2600.):
        left.axvline(boundary, color='#8793a2', ls=':', lw=.8)
    left_handles = [Line2D([], [], color='#334155', lw=2, label='Shared curve'),
                    Patch(facecolor='#8b9bae', alpha=.25, label='Conditional ±1σ')]
    for side in SIDES:
        accuracy, actual, crossing = data['observed'][side], data['actual'].get(side), crossings[side]
        color = COLORS[side]
        if accuracy is not None:
            left.axhline(accuracy, color=color, ls='--', lw=1.1)
            left_handles.append(Line2D([], [], color=color, ls='--', label=f'{side} accuracy'))
        if actual is not None:
            left.axvline(actual, color=color, ls=':', lw=1., alpha=.7)
        if crossing['estimate'] is not None and 200 <= crossing['estimate'] <= 3000:
            left.scatter(crossing['estimate'], accuracy, color=color, s=44, edgecolor='white', zorder=5)
            left.vlines(crossing['estimate'], 50., accuracy, color=color, ls=':', lw=.8)
    if data['anchor'] is not None:
        left.axvline(data['anchor'], color='#6b6252', ls='-.', lw=1.)
        left_handles.append(Line2D([], [], color='#6b6252', ls='-.', label=f'Mean actual: {data["anchor"]:,.1f}'))
    left.legend(handles=left_handles, fontsize=8.5, loc='lower left', ncols=2)
    _table(left, [[side, _number(data['actual'].get(side)),
                   _number(data['observed'][side], 2)+'%' if data['observed'][side] is not None else '—',
                   _crossing_label(crossings[side])] for side in SIDES],
           ['Player', 'Actual Elo', 'Observed accuracy', 'Raw intersection'])

    accuracies = np.linspace(50., 100., 251)
    for mapping in data['maps']:
        points = np.array([mapping.point(float(value)) for value in accuracies], dtype=float)
        right.plot(accuracies, points, color=mapping.color, lw=2., label=mapping.label)
        for side in SIDES:
            observed = data['observed'][side]
            point = mapping.point(observed) if observed is not None else None
            if point is not None:
                right.scatter(observed, point, color=COLORS[side], marker='o' if side == 'White' else 's',
                              s=46, edgecolor='white', linewidth=.8, zorder=5)
    for side in SIDES:
        if data['observed'][side] is not None:
            right.axvline(data['observed'][side], color=COLORS[side], ls=':', lw=.9)
    if data['anchor'] is not None:
        right.axhline(data['anchor'], color='#6b6252', ls='-.', lw=1.)
    right.set(xlim=(50., 100.), ylim=(200., 3000.), xlabel='Observed arithmetic accuracy (%)',
              ylabel='Estimated rating')
    right.set_yticks(np.arange(200, 3001, 400))
    right.set_title('Accuracy → estimated rating\nOne common map per method for both players', fontsize=11)
    right.legend(fontsize=9, loc='upper left')
    _table(right, [[side, *[_number(mapping.point(data['observed'][side]))
                            if data['observed'][side] is not None else '—' for mapping in data['maps']]]
                  for side in SIDES], ['Player', *[mapping.label for mapping in data['maps']]])
    figure.text(.5, .055,
                'Gray regions are curve extrapolation. The ±1σ band is conditional accuracy noise, not a rating interval.\n'
                'Dotted colored rating lines mark actual Elo; dash-dot lines mark the common mean account anchor. '
                'Undefined intersections remain undefined.',
                ha='center', va='center', fontsize=9, color='#4b5563')
    return figure


def export(game, data, output):
    """Atomically write only this game’s requested diagnostic SVG."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'curves.svg'
    figure = build_figure(game, data)
    temporary = None
    try:
        with NamedTemporaryFile(dir=output, prefix='.curves-', suffix='.svg', delete=False) as file:
            temporary = Path(file.name)
        with rc_context({'svg.fonttype': 'none'}):
            figure.savefig(temporary, format='svg', facecolor='white')
        temporary.replace(path)
    finally:
        figure.clear()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def run(output=OUTPUT, *, numbers=None, methods=DEFAULT_METHODS):
    """Read existing game evidence and write diagnostic SVGs only."""
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'tests/analysis/output'):
        raise ValueError('Diagnostic output must stay under tests/analysis/output.')
    if numbers is None:
        numbers = sorted(int(match[1]) for path in (ROOT/'games').glob('game*.pgn')
                         if (match := re.fullmatch(r'game(\d+)\.pgn', path.name)))
    cases = load_cases(ROOT/'games', numbers, Path(CONFIG['ANALYSIS']['CACHE_DIR']))
    paths = []
    for case in cases:
        evidence = current_pgn_context(case)
        actual = {side: evidence[side]['actual_rating'] for side in SIDES}
        path = export(case['name'], prepare_plot(evidence, actual, methods), output/case['name'])
        paths.append(path)
        print(path, flush=True)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--games', type=int, nargs='+', help='Game numbers; default discovers all game PGNs.')
    parser.add_argument('--methods', nargs='+', choices=tuple(METHODS), default=list(DEFAULT_METHODS))
    args = parser.parse_args()
    run(args.output, numbers=args.games, methods=args.methods)
