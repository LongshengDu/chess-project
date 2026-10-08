"""SVG views of native expected accuracy and probability-weighted dispersion."""
from __future__ import annotations

import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import fill

from analysis.accuracy.by_move import DEFAULT_MAIA_ELO, DEFAULT_MAIA_ELOS, export_move_curve
from analysis.accuracy.plot_style import SHARED_COLOR, SIDE_COLORS, SVG_STYLE

DEFAULT_FIGURE_NAMES = ('accuracy-curve.svg', *(f'accuracy-by-move-{elo}.svg' for elo in DEFAULT_MAIA_ELOS))


def _intersection_ranges(ratings, expected, observed):
    """Crossings of the drawn segments, including flat overlaps; never extrapolate."""
    if observed is None or not math.isfinite(observed):
        return []
    matches = []
    for rating, value in zip(ratings, expected, strict=True):
        if value is not None and math.isfinite(value) and math.isclose(value, observed, abs_tol=1e-9, rel_tol=0):
            matches.append((float(rating), float(rating)))
    for x0, x1, y0, y1 in zip(ratings, ratings[1:], expected, expected[1:]):
        if any(value is None or not math.isfinite(value) for value in (y0, y1)):
            continue
        a, b = y0-observed, y1-observed
        if abs(a) <= 1e-9 and abs(b) <= 1e-9:
            matches.append((float(x0), float(x1)))
        elif a*b < 0:
            crossing = float(x0+(x1-x0)*(-a)/(b-a))
            matches.append((crossing, crossing))
    merged = []
    for low, high in sorted(matches):
        if merged and low <= merged[-1][1]+1e-7:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    return merged


def _mark_intersections(axis, ranges, average, color):
    """Project crossings with faint guides; coordinates belong in the legend."""
    for low, high in ranges:
        axis.vlines([low] if low == high else [low, high], 50, average,
                    color=color, lw=.8, ls=':', alpha=.65)


def export_saved_figures(analysis, output_dir):
    """Replace the overview and all default by-move SVGs from saved evidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    data = analysis.get('accuracy_curve')
    if not isinstance(data, dict):
        raise ValueError('Accuracy curve is unavailable; run the common game analysis first.')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    x = np.asarray(data['ratings'], dtype=float)
    mu = np.asarray(data['expected_accuracy'], dtype=float)
    deviation = np.asarray(data['absolute_deviation'], dtype=float)
    with plt.rc_context(SVG_STYLE):
        fig, ((white, black), (shared, spread)) = plt.subplots(2, 2, figsize=(13.6, 10))
        try:
            shared.plot(x, mu, color=SHARED_COLOR, lw=2.4, ls=':', label='Shared curve')
            shared.fill_between(x, np.maximum(0, mu-deviation), np.minimum(100, mu+deviation),
                                color=SHARED_COLOR, alpha=.14, label='± absolute deviation')
            spread.plot(x, deviation, color=SHARED_COLOR, lw=2.4, ls=':', label='Shared absolute deviation')
            for side, player in data['players'].items():
                curve = white if side == 'white' else black
                average = player['average_accuracy']
                expected = np.asarray(player['expected_accuracy'], dtype=float)
                own_deviation = np.asarray(player['absolute_deviation'], dtype=float)
                curve.plot(x, expected, color=SIDE_COLORS[side], lw=2.4,
                           label=f'{side.title()} positions')
                shared.plot(x, expected, color=SIDE_COLORS[side], lw=1.4,
                            label=f'{side.title()} positions')
                curve.fill_between(x, np.maximum(0, expected-own_deviation),
                                   np.minimum(100, expected+own_deviation), color=SIDE_COLORS[side],
                                   alpha=.14, label='± absolute deviation')
                if average is not None:
                    lichess = player.get('lichess_accuracy')
                    lichess_label = f'{lichess:.1f}' if lichess is not None else 'unavailable'
                    decisions = player.get('moves_used')
                    label = f'{side.title()}: avg {average:.1f}; lc {lichess_label}'
                    if decisions is not None:
                        label += f'; {decisions} pos'
                    for axis, values in ((curve, expected), (shared, mu)):
                        ranges = _intersection_ranges(x, values, average)
                        coordinates = ', '.join(f'{low:.0f}' if low == high else f'{low:.0f}–{high:.0f}'
                                                for low, high in ranges)
                        intersection = f'; xelo {coordinates}' if ranges else ''
                        axis.axhline(average, color=SIDE_COLORS[side], lw=.8, ls=':', alpha=.65,
                                     label=label+intersection)
                        _mark_intersections(axis, ranges, average, SIDE_COLORS[side])
                spread.plot(x, player['absolute_deviation'], color=SIDE_COLORS[side], lw=1.2,
                            label=f'{side.title()} positions')
            for axis, title in ((white, 'White position accuracy curve'),
                                (black, 'Black position accuracy curve'), (shared, 'Shared accuracy curve')):
                axis.set(title=title, ylabel='Accuracy (%)', ylim=(50, 100))
            spread.set(title='Probability-weighted absolute deviation', ylabel='Accuracy points',
                       ylim=(0, 50))
            for axis in (white, black, shared, spread):
                axis.set(xlim=(600, 2600), xlabel='Maia Elo anchor · native Lichess Blitz')
                axis.set_xticks(range(600, 2601, 200))
                axis.grid(alpha=.18)
                handles, labels = axis.get_legend_handles_labels()
                labels = [fill(label, width=60, break_long_words=False, break_on_hyphens=False)
                          for label in labels]
                axis.legend(handles, labels, loc='best' if axis is spread else 'lower left',
                            fontsize=8, framealpha=.95, borderpad=.7)
            game = analysis['game']
            fig.suptitle(f'{game["white"]["name"] or "White"} — {game["black"]["name"] or "Black"}', fontsize=14, y=.98)
            fig.subplots_adjust(left=.06, right=.985, top=.92, bottom=.07, wspace=.22, hspace=.3)
            target = directory/DEFAULT_FIGURE_NAMES[0]
            with NamedTemporaryFile(dir=directory, suffix='.svg', delete=False) as temp:
                temporary = Path(temp.name)
            try:
                fig.savefig(temporary, format='svg', metadata={'Date': None})
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            plt.close(fig)
    paths = {'curve': str(target)}
    for elo in DEFAULT_MAIA_ELOS:
        key = 'by_move' if elo == DEFAULT_MAIA_ELO else f'by_move_{elo}'
        paths[key] = export_move_curve(analysis, directory, elo)['curve']
    return paths
