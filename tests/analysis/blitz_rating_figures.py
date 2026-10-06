"""Equal-scale SVG comparison of saved rating-method results on the Blitz scale."""
from __future__ import annotations

import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


NATIVE_COLOR = '#2563a6'
EDGE_COLOR = '#ce6d16'
LIMIT_COLOR = '#5a5363'
RATING_LIMITS = (0., 3200.)
FILE_NAME = 'commercial-estimated-elo.svg'


def _number(value, *, missing=False):
    if value is None and missing:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Rating figure values must be finite numbers or explicitly absent.')
    return float(value)


def _identifier(row):
    return f"g{row['number']}{row['side'][0]}"


def _overlap(first, second):
    width = max(0., min(first.x1, second.x1)-max(first.x0, second.x0))
    height = max(0., min(first.y1, second.y1)-max(first.y0, second.y0))
    return width*height


def _label_points(axis, entries, points):
    """Place short edge labels in display space, avoiding labels and markers."""
    if not entries:
        return
    renderer = axis.figure.canvas.get_renderer()
    font = FontProperties(size=8.5)
    scale = axis.figure.dpi/72.
    bounds = axis.get_window_extent().padded(-7)
    markers = [Bbox.from_bounds(x-5, y-5, 10, 10) for x, y in axis.transData.transform(points)]
    occupied = []
    angles = np.array([45, 135, 315, 225, 0, 180, 90, 270])*np.pi/180
    # Boundary points have fewer feasible positions, so place them first.
    entries = sorted(entries, key=lambda row: (not row['boundary'], -row['y'], row['label']))
    for row in entries:
        origin = axis.transData.transform((row['x'], row['y']))
        width, height, _ = renderer.get_text_width_height_descent(row['label'], font, ismath=False)
        choices = []
        for radius in (15, 23, 33, 46, 62, 82, 108):
            for angle in angles:
                offset = radius*np.array([np.cos(angle), np.sin(angle)])
                center = origin+scale*offset
                box = Bbox.from_bounds(center[0]-width/2-3, center[1]-height/2-3, width+6, height+6)
                outside = box.width*box.height-_overlap(box, bounds)
                collisions = sum(_overlap(box, existing) for existing in occupied)
                marker_hits = sum(_overlap(box, marker) for marker in markers)
                cost = 10000*outside+100*collisions+30*marker_hits+radius
                choices.append((cost, offset, box))
        _, offset, box = min(choices, key=lambda value: value[0])
        occupied.append(box)
        axis.annotate(row['label'], (row['x'], row['y']), xytext=offset,
                      textcoords='offset points', ha='center', va='center', fontsize=8.5,
                      color=LIMIT_COLOR if row['boundary'] else '#a54e0a', zorder=7,
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': .35},
                      arrowprops={'arrowstyle': '-', 'color': '#787878', 'linewidth': .6,
                                  'shrinkA': 2, 'shrinkB': 4}, annotation_clip=False)


def _metric(value):
    return 'n/a' if value is None else f'{_number(value):.1f}'


def build_figure(result):
    """Build a figure from completed predictions; never fit or infer a rating."""
    methods = result['method_order']
    if len(methods) != 3 or len(set(methods)) != 3:
        raise ValueError('The comparison requires exactly three distinct methods.')
    games = result['games']
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise ValueError('A positive game count is required.')
    rows_by_method = {method: [] for method in methods}
    identities = set()
    for row in result['players']:
        method = row['method']
        if method not in rows_by_method or row['side'] not in ('White', 'Black'):
            raise ValueError('Unknown method or player side in comparison rows.')
        identity = (method, row['game'], row['side'])
        if identity in identities:
            raise ValueError('Duplicate method/game/player row.')
        identities.add(identity)
        reference, estimate = _number(row['reference']), _number(row['estimate'], missing=True)
        if not 0 <= reference <= 3200 or estimate is not None and not 0 <= estimate <= 3200:
            raise ValueError('Rating comparison points must fit the declared 0–3200 axes.')
        if row.get('intersection_bound') is not None and row['intersection_bound'] not in RATING_LIMITS:
            raise ValueError('An unavailable intersection must use a declared support boundary.')
        rows_by_method[method].append(row)
    if any(len(rows) != games*2 for rows in rows_by_method.values()):
        raise ValueError('Every method must retain both players from every game, including missing estimates.')

    fig = Figure(figsize=(21, 8), facecolor='white')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=.047, right=.987, bottom=.17, top=.80, wspace=.18)
    fig.suptitle('Commercial reference Elo vs Estimated Elo', fontsize=21, fontweight='bold', y=.973)
    fig.text(.5, .925, f'Lichess Blitz rating scale · {games} games / {2*games} players · identical 0–3200 axes',
             ha='center', fontsize=12, color='#424a55')
    all_missing, has_boundary, pending = [], False, []
    for axis, method in zip(axes, methods, strict=True):
        rows = rows_by_method[method]
        summary = result['summary'][method]
        available = [row for row in rows if row['estimate'] is not None]
        missing = [row for row in rows if row['estimate'] is None]
        if summary['estimated_players'] != len(available) or summary['missing_players'] != len(missing):
            raise ValueError('Summary counts do not match the plotted player rows.')
        axis.set(xlim=RATING_LIMITS, ylim=RATING_LIMITS,
                 xlabel='Commercial reference Elo', ylabel='Estimated Elo')
        axis.set_box_aspect(1)
        axis.set_xticks(np.arange(0, 3201, 400))
        axis.set_yticks(np.arange(0, 3201, 400))
        axis.tick_params(labelsize=9)
        axis.grid(alpha=.17, zorder=0)
        axis.spines[['top', 'right']].set_visible(False)
        axis.plot(RATING_LIMITS, RATING_LIMITS, color='#9b9fa5', lw=1.1, ls='--', zorder=1)
        for edge, color, marker in ((False, NATIVE_COLOR, 'o'), (True, EDGE_COLOR, 'D')):
            selected = [row for row in available if bool(row['edge']) == edge]
            if selected:
                axis.scatter([row['reference'] for row in selected], [row['estimate'] for row in selected],
                             s=38 if edge else 29, marker=marker, color=color,
                             edgecolors='white', linewidths=.55, alpha=.95 if edge else .75, zorder=4)
        entries = [{'x': row['reference'], 'y': row['estimate'], 'label': _identifier(row), 'boundary': False}
                   for row in available if row['edge']]
        points = [(row['reference'], row['estimate']) for row in available]
        unplaced = []
        for row in missing:
            bound = row.get('intersection_bound')
            if bound is None:
                unplaced.append(_identifier(row))
                continue
            has_boundary = True
            axis.scatter([row['reference']], [bound], s=65, marker='^' if bound == 3200 else 'v',
                         facecolors='none', edgecolors=LIMIT_COLOR, linewidths=1.3,
                         zorder=5, clip_on=False)
            entries.append({'x': row['reference'], 'y': bound, 'label': _identifier(row)+'*', 'boundary': True})
            points.append((row['reference'], bound))
        if unplaced:
            all_missing.append(f"{result['method_names'][method]}: "+', '.join(unplaced))
        title = (f"{result['method_names'][method]}\n"
                 f"MAE {_metric(summary['mae'])} · native {_metric(summary['native_mae'])} · edge {_metric(summary['edge_mae'])}\n"
                 f"{len(available)}/{len(rows)} estimates · max error {_metric(summary['maximum_error'])}")
        axis.set_title(title, fontsize=11, pad=13, linespacing=1.55)
        pending.append((axis, entries, points))
    handles = [Line2D([], [], marker='o', ls='none', color=NATIVE_COLOR, markersize=6, label='Native-range observation'),
               Line2D([], [], marker='D', ls='none', color=EDGE_COLOR, markersize=6, label='Outside native Maia accuracy range'),
               Line2D([], [], ls='--', color='#9b9fa5', label='Equal reference and estimate')]
    if has_boundary:
        handles.append(Line2D([], [], marker='^', markerfacecolor='none', ls='none', color=LIMIT_COLOR,
                              markersize=7, label='No crossing: support boundary only*'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .071), ncols=len(handles),
               frameon=False, fontsize=10)
    footer = ('Edge labels identify game and side (g1W = game1 White). Native range is Maia 600–2600. '
              'MAE uses available estimates only.')
    fig.text(.5, .051, footer, ha='center', fontsize=9, color='#555')
    if has_boundary:
        fig.text(.5, .027, '* Hollow triangles mark 0/3200 curve-support boundaries, not Elo estimates; these players are excluded from MAE.',
                 ha='center', fontsize=9, color=LIMIT_COLOR)
    if all_missing:
        fig.text(.5, .006, 'No unique crossing and no plotted estimate: '+'; '.join(all_missing),
                 ha='center', fontsize=8, color=LIMIT_COLOR, wrap=True)
    fig.canvas.draw()
    for axis, entries, points in pending:
        _label_points(axis, entries, points)
    return fig


def export(result, output: Path):
    """Atomically save the comparison SVG in an output directory and return it."""
    figure = build_figure(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination, temporary = output/FILE_NAME, None
    try:
        with NamedTemporaryFile(dir=output, prefix='.blitz-comparison-', suffix='.svg', delete=False) as stream:
            temporary = Path(stream.name)
        with rc_context({'svg.fonttype': 'none'}):
            figure.savefig(temporary, format='svg', facecolor='white')
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        figure.clear()
    return destination
