"""Render a selected three-method research comparison as SVG figures and HTML.

The completed common-runner result is the only input. No estimators, engines,
PGNs or external services are called. Error summaries are displayed as saved;
missing predictions remain missing and are excluded from scatter points.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import fill

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tests.analysis.blitz_rating_figures import EDGE_COLOR, NATIVE_COLOR, _label_points


LIMITS = (200., 3000.)
SIDES = ('White', 'Black')
FILES = {'scatter': 'commercial-estimated-elo.svg', 'balance': 'reference-error-balance.svg', 'html': 'index.html'}


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Figure values must be finite numeric values.')
    return float(value)


def _label(method, labels):
    return str(labels.get(method, method.replace('_', ' ').capitalize()))


def _identifier(row):
    return f"g{row['number']}{row['side'][0]}"


def _format(value, decimals=1):
    return '—' if value is None else f'{_finite(value):,.{decimals}f}'


def _validate(result, methods, labels):
    methods, labels = tuple(methods), dict(labels or {})
    if len(methods) != 3 or len(set(methods)) != 3 or any(method not in result['summary'] for method in methods):
        raise ValueError('Select exactly three distinct methods present in the completed result.')
    games = result['games']
    if isinstance(games, bool) or not isinstance(games, int) or games < 1:
        raise ValueError('A positive game count is required.')
    rows, seen, shared = {method: [] for method in result['summary']}, set(), {}
    for row in result['players']:
        key = (row['method'], row['game'], row['side'])
        if key in seen or row['method'] not in rows or row['side'] not in SIDES:
            raise ValueError('Duplicate or unknown method/game/player record.')
        seen.add(key)
        _finite(row['reference'])
        if row['estimate'] is not None:
            _finite(row['estimate'])
        if row.get('actual') is not None:
            _finite(row['actual'])
        player = (row['game'], row['side'])
        values = (row['reference'], row.get('actual'))
        if player in shared and shared[player] != values:
            raise ValueError('Methods must use the same reference and actual rating for each player.')
        shared[player] = values
        rows[row['method']].append(row)
    for method in methods:
        if len(rows[method]) != games*2:
            raise ValueError('Each selected method must retain all players, including missing estimates.')
        available = sum(row['estimate'] is not None for row in rows[method])
        if result['summary'][method]['players'] != available:
            raise ValueError('Saved summary count does not match available predictions.')
    return methods, labels, rows


def build_scatter(result, methods, labels=None):
    """Three equal-scale square panels; out-of-axis values stay explicitly marked."""
    methods, labels, rows = _validate(result, methods, labels)
    fig = Figure(figsize=(21, 8), facecolor='white')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=.047, right=.987, top=.79, bottom=.18, wspace=.18)
    fig.suptitle('Commercial reference Elo vs estimated Elo', y=.98, fontsize=21, fontweight='bold')
    fig.text(.5, .935, f"Lichess Blitz rating scale · {result['games']} games / {2*result['games']} players · identical 200–3000 axes",
             ha='center', fontsize=12, color='#424a55')
    pending, outside = [], False
    for axis, method in zip(axes, methods, strict=True):
        summary = result['summary'][method]
        available = [row for row in rows[method] if row['estimate'] is not None]
        missing = [row for row in rows[method] if row['estimate'] is None]
        axis.set(xlim=LIMITS, ylim=LIMITS, xlabel='Commercial reference Elo', ylabel='Estimated Elo')
        axis.set_box_aspect(1)
        axis.set_xticks(np.arange(200., 3001., 400.))
        axis.set_yticks(np.arange(200., 3001., 400.))
        axis.tick_params(labelsize=9)
        axis.grid(alpha=.17)
        axis.spines[['top', 'right']].set_visible(False)
        axis.plot(LIMITS, LIMITS, ls='--', color='#9b9fa5', lw=1.1, zorder=1)
        entries, points = [], []
        for row in available:
            x, y = row['reference'], row['estimate']
            plotted_x, plotted_y = np.clip(x, *LIMITS), np.clip(y, *LIMITS)
            clipped = plotted_x != x or plotted_y != y
            edge = bool(row['edge'])
            color, marker = (EDGE_COLOR, 'D') if edge else (NATIVE_COLOR, 'o')
            if clipped:
                marker = '^' if y > LIMITS[1] else 'v' if y < LIMITS[0] else '>' if x > LIMITS[1] else '<'
                outside = True
            axis.scatter([plotted_x], [plotted_y], s=40 if edge or clipped else 29,
                         marker=marker, color=color, edgecolors='white', linewidths=.55,
                         alpha=.95 if edge or clipped else .75, zorder=4, clip_on=not clipped)
            points.append((plotted_x, plotted_y))
            if edge or clipped:
                label = _identifier(row)
                if clipped:
                    label += f" ({x:,.0f}, {y:,.0f})" if plotted_x != x else f" {y:,.0f}"
                entries.append({'x': plotted_x, 'y': plotted_y, 'label': label, 'boundary': clipped})
        name = fill(_label(method, labels), width=39)
        axis.set_title(f"{name}\nMAE {_format(summary['mae'])} · native {_format(summary['native_mae'])} · edge {_format(summary['edge_mae'])}\n"
                       f"{len(available)}/{len(rows[method])} estimates · max error {_format(summary['maximum_error'])}",
                       fontsize=11, pad=13, linespacing=1.45)
        if missing:
            identifiers = ', '.join(_identifier(row) for row in missing)
            axis.text(.02, .02, fill('No estimate; not plotted: '+identifiers, 58),
                      transform=axis.transAxes, fontsize=8, color='#62606a', va='bottom',
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
        pending.append((axis, entries, points))
    handles = [Line2D([], [], marker='o', ls='none', color=NATIVE_COLOR, markersize=6, label='Native-range observation'),
               Line2D([], [], marker='D', ls='none', color=EDGE_COLOR, markersize=6, label='Outside native Maia accuracy range'),
               Line2D([], [], ls='--', color='#9b9fa5', label='Equal reference and estimate')]
    if outside:
        handles.append(Line2D([], [], marker='^', ls='none', color='#62606a', markersize=7,
                              label='Outside axis: label gives actual value'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .078), ncols=len(handles), frameon=False, fontsize=10)
    fig.text(.5, .055, 'Edge labels identify game and side (g1W = game1 White). Native Maia range: 600–2600. Missing estimates are excluded from MAE.',
             ha='center', fontsize=9, color='#555')
    fig.canvas.draw()
    for axis, entries, points in pending:
        _label_points(axis, entries, points)
    return fig


def build_balance(result, methods, labels=None):
    """Compare both sign counts and mean signed error, without changing results."""
    methods, labels, _ = _validate(result, methods, labels)
    fig = Figure(figsize=(15, 5), facecolor='white')
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.65, 1.]})
    fig.subplots_adjust(left=.22, right=.97, top=.78, bottom=.23, wspace=.22)
    fig.suptitle('Error balance against commercial reference ratings', fontsize=17, fontweight='bold', y=.965)
    fig.text(.5, .895, 'Counts and signed error answer different questions; equal counts do not guarantee small errors.',
             ha='center', fontsize=10, color='#555')
    count_limit = 2*result['games']
    bias_limit = max(100., math.ceil(max(abs(result['summary'][method]['bias']) for method in methods)/100.)*100.)
    for index, method in enumerate(methods):
        summary = result['summary'][method]
        y = len(methods)-1-index
        below, above, equal = summary['below_reference'], summary['above_reference'], summary['equal_reference']
        left.barh(y, -below, color=NATIVE_COLOR, height=.5)
        left.barh(y, above, color=EDGE_COLOR, height=.5)
        if below:
            left.text(-below/2, y, str(below), color='white', ha='center', va='center', fontsize=11)
        if above:
            left.text(above/2, y, str(above), color='white', ha='center', va='center', fontsize=11)
        left.text(0., y-.38, f"{equal} equal · {summary['players']}/{count_limit} available", ha='center', fontsize=8, color='#555')
        bias = summary['bias']
        right.barh(y, bias, height=.5, color=EDGE_COLOR if bias >= 0 else NATIVE_COLOR)
        right.annotate(f'{bias:+.1f}', (bias, y), xytext=(5 if bias >= 0 else -5, 0),
                       textcoords='offset points', ha='left' if bias >= 0 else 'right', va='center', fontsize=10)
    positions = np.arange(len(methods))
    names = [fill(_label(method, labels), 31) for method in reversed(methods)]
    left.set_yticks(positions, names, fontsize=10)
    right.set_yticks(positions, ['']*len(methods))
    left.set(xlim=(-count_limit*1.04, count_limit*1.04), xlabel='Players below ← reference → players above', title='Direction of error')
    ticks = np.linspace(-count_limit, count_limit, 9)
    left.set_xticks(ticks, [f'{abs(value):g}' for value in ticks], fontsize=9)
    right.set(xlim=(-bias_limit*1.25, bias_limit*1.25), xlabel='Mean estimated Elo − reference Elo', title='Mean signed error')
    for axis in (left, right):
        axis.set_ylim(-.65, len(methods)-.5)
        axis.axvline(0, color='#56606b', lw=.8)
        axis.grid(axis='x', alpha=.17)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.tick_params(axis='y', length=0)
    handles = [Line2D([], [], color=NATIVE_COLOR, lw=7, label='Below reference'),
               Line2D([], [], color=EDGE_COLOR, lw=7, label='Above reference')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncols=2, frameon=False, fontsize=10)
    return fig


def _player_value(row):
    if row['estimate'] is None:
        return '<span class="missing" title="No available estimate">—</span>'
    value = str(math.floor(row['estimate']+.5))
    if row['edge']:
        return f'<span class="edge" title="Observed accuracy outside the native Maia curve">{value}<sup>◆</sup></span>'
    return value


def html_report(result, methods, labels=None, *, numerical_results=None):
    """Create a local report with all game pairs and every method's saved metrics."""
    methods, labels, rows = _validate(result, methods, labels)
    groups = {method: {} for method in methods}
    for method in methods:
        for row in rows[method]:
            groups[method].setdefault(row['number'], {})[row['side']] = row
    game_rows = []
    for number in sorted(groups[methods[0]]):
        baseline = groups[methods[0]][number]
        actual = ' / '.join(_format(baseline[side].get('actual'), 0) for side in SIDES)
        reference = ' / '.join(_format(baseline[side]['reference'], 0) for side in SIDES)
        values = [' / '.join(_player_value(groups[method][number][side]) for side in SIDES) for method in methods]
        game_rows.append('<tr><th scope="row">game'+str(number)+'</th><td>'+actual+'</td><td><strong>'+reference+
                         '</strong></td>'+''.join('<td>'+value+'</td>' for value in values)+'</tr>')
    metric_rows, diagnostic_rows = [], []
    all_methods = [*methods, *(method for method in result['summary'] if method not in methods)]
    for method in all_methods:
        summary = result['summary'][method]
        name = escape(_label(method, labels))
        if method in methods:
            name = '<strong>'+name+'</strong>'
        counts = f"{summary['above_reference']} / {summary['below_reference']} / {summary['equal_reference']}"
        metrics = [f"{summary['players']}/{result['games']*2}", _format(summary['mae']),
                   _format(summary['native_mae'])+' / '+_format(summary['edge_mae']),
                   _format(summary['maximum_error']), _format(summary.get('edge_maximum_error')),
                   _format(summary['bias']), _format(summary.get('median_signed_error')), counts]
        metric_rows.append('<tr><th scope="row">'+name+'</th>'+''.join('<td>'+value+'</td>' for value in metrics)+'</tr>')
        denominator = summary['comparable_games']
        diagnostics = [f"{summary['reference_order_matches']}/{denominator}",
                       f"{summary.get('display_order_matches', 0)}/{denominator}",
                       f"{summary.get('curve_order_matches', 0)}/{denominator}",
                       _format(summary.get('maximum_own_rating_change')),
                       _format(summary.get('maximum_both_ratings_change'))]
        diagnostic_rows.append('<tr><th scope="row">'+name+'</th>'+''.join('<td>'+value+'</td>' for value in diagnostics)+'</tr>')
    selected_headers = ''.join('<th>'+escape(_label(method, labels))+'<br><small>White / Black</small></th>' for method in methods)
    timestamp = escape(str(result.get('created_utc', '')))
    sensitivity = 'measured' if result.get('sensitivity_evaluated') else 'not measured in this run; stored zeros are not an established bound'
    numerical_link = f'<a href="{escape(numerical_results, quote=True)}">Numerical results</a>' if numerical_results else ''
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shared-curve rating comparison</title><style>
:root{{color-scheme:light;--ink:#263341;--muted:#647080;--line:#dfe5eb;--paper:#fff;--edge:{EDGE_COLOR}}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1600px;margin:auto;padding:32px 28px 48px}}header{{margin-bottom:26px}}h1{{font-size:32px;line-height:1.15;margin:0 0 12px}}
h2{{font-size:22px;margin:0 0 14px}}p{{margin:10px 0}}.muted,small{{color:var(--muted)}}nav a{{margin-right:20px}}a{{color:#1a6394}}
section{{background:var(--paper);padding:25px;margin:22px 0;border:1px solid var(--line);border-radius:10px}}
figure{{margin:10px 0}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;color:var(--muted);margin:10px 0}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}thead th{{background:#edf2f6;color:#334353;font-size:13px;vertical-align:bottom}}
th:first-child{{text-align:left;white-space:normal;min-width:125px}}tbody tr:hover{{background:#f7fafc}}.edge{{color:#a54e0a}}sup{{font-size:8px;margin-left:2px}}
.missing{{color:#777}}.notes{{font-size:13px;color:var(--muted)}}.methods th:first-child{{min-width:260px;max-width:350px}}
@media(max-width:700px){{main{{padding:20px 12px}}section{{padding:15px}}h1{{font-size:26px}}}}
</style></head><body><main>
<header><h1>Shared-curve rating comparison</h1>
<p>Lichess Blitz rating scale · {result['games']} games · {result['games']*2} players</p>
<p class="muted">Fixed research candidates compared against commercial reference ratings. Selection after reviewing these games remains exploratory.</p>
<nav><a href="#scatter">Rating comparison</a><a href="#balance">Error balance</a><a href="#games">All games</a><a href="#methods">All methods</a>{numerical_link}</nav>
<small>{timestamp}</small></header>
<section id="scatter"><h2>Selected methods</h2><p class="notes">The shortlist shows complementary trade-offs in absolute error, edge cases, and above/below balance. No candidate is presented as satisfying every target.</p><figure><a href="{FILES['scatter']}"><img src="{FILES['scatter']}" alt="Three equal-scale scatter plots of estimated versus commercial reference Elo"></a>
<figcaption>Blue circles are native-range observations; orange diamonds are edge cases. Identical 200–3000 axes make the methods directly comparable. Missing estimates are not plotted. Any finite point outside the axis is explicitly marked and labeled with its true value.</figcaption></figure></section>
<section id="balance"><h2>Above and below the reference</h2><figure><a href="{FILES['balance']}"><img src="{FILES['balance']}" alt="Counts above and below reference ratings, alongside mean signed errors"></a>
<figcaption>Balanced counts do not establish accuracy or calibrated uncertainty. Mean signed error retains the magnitude and direction of errors.</figcaption></figure></section>
<section id="games"><h2>Every game</h2><p class="notes">White / Black values are rounded for display; all error metrics use unrounded estimates. Orange ◆ marks observed accuracy outside the native Maia 600–2600 curve. A dash means no available estimate.</p>
<div class="scroll"><table><thead><tr><th>Game</th><th>Actual<br><small>White / Black</small></th><th>Commercial reference<br><small>White / Black</small></th>{selected_headers}</tr></thead><tbody>{''.join(game_rows)}</tbody></table></div></section>
<section id="methods"><h2>All method diagnostics</h2><p class="notes">The selected methods appear first in bold. Each method's MAE uses only its available predictions; methods with missing players are evaluated on a different subset.</p>
<div class="scroll"><table class="methods"><thead><tr><th>Method</th><th>Players</th><th>MAE</th><th>Native / edge MAE</th><th>Max error</th><th>Edge max</th><th>Mean signed</th><th>Median signed</th><th>Above / below / equal</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
<h2 style="margin-top:28px">Ordering and account sensitivity</h2>
<p class="notes">Equal commercial ratings require |estimated White − Black| &lt;50 Elo. Otherwise only the matching direction is required, including estimated gaps below50. Curve order compares the fitted sign with observed arithmetic-accuracy order. Account sensitivity is {escape(sensitivity)}.</p>
<div class="scroll"><table class="methods"><thead><tr><th>Method</th><th>W/B acceptance</th><th>Displayed acceptance</th><th>Curve order</th><th>Max change: own Elo ±200</th><th>Max change: both Elo ±200</th></tr></thead><tbody>{''.join(diagnostic_rows)}</tbody></table></div></section>
<footer class="notes">This report renders completed predictions only. It does not fit parameters, rerun engines, or change any production result.</footer>
</main></body></html>'''


def _save_figure(figure, path):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix='.'+path.stem+'-', suffix='.svg', delete=False) as stream:
            temporary = Path(stream.name)
        with rc_context({'svg.fonttype': 'none'}):
            figure.savefig(temporary, format='svg', facecolor='white')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        figure.clear()


def export(result, output, *, methods, labels=None):
    """Save two SVGs and one HTML report; no PNG or numerical result is written."""
    methods, labels, _ = _validate(result, methods, labels)
    output = Path(output)
    html = html_report(result, methods, labels, numerical_results='comparison.json' if (output/'comparison.json').is_file() else None)
    scatter, balance = build_scatter(result, methods, labels), build_balance(result, methods, labels)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: output/name for key, name in FILES.items()}
    _save_figure(scatter, paths['scatter'])
    _save_figure(balance, paths['balance'])
    temporary = None
    try:
        with NamedTemporaryFile(dir=output, prefix='.index-', suffix='.html', mode='w', encoding='utf-8', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(html)
        temporary.replace(paths['html'])
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs=3, required=True)
    parser.add_argument('--labels', nargs=3)
    options = parser.parse_args()
    result = json.loads(options.input.read_text(encoding='utf-8'))
    labels = dict(zip(options.methods, options.labels, strict=True)) if options.labels else None
    for name, path in export(result, options.output, methods=options.methods, labels=labels).items():
        print(f'{name}: {path}')
