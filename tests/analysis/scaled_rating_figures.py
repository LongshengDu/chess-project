"""SVG and local HTML views of completed scale-aware rating comparisons.

Rendering uses saved results only. Every method appears; population-only
controls are explicitly separate, and scales are never pooled in a panel.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from textwrap import fill

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tests.analysis.blitz_rating_figures import EDGE_COLOR, NATIVE_COLOR, _label_points
from tests.analysis.intuitive_curve_figures import _save_figure


SIDES = ('White', 'Black')
LIMITS = (0., 3200.)
SCOPES = ('baseline', 'target_curve', 'population_control')
SCOPE_NAMES = {'baseline': 'Baselines', 'target_curve': 'Target shared-curve candidates',
               'population_control': 'Population-only controls'}
LABELS = {
    'shared_curve_intersection': 'Shared-curve intersection',
    'bayesian_shared_curve': 'Original Bayesian shared curve',
    'arithmetic_coverage': 'Arithmetic coverage',
    'uncertainty_ensemble': 'Arithmetic uncertainty ensemble',
    'residual_secant_ridge': 'Account-centered secant ridge',
    'residual_logloss_ridge': 'Account-centered log-loss ridge',
    'residual_elo_logodds': 'Account-centered Elo log-odds',
    'curve_anchor_logistic_sd': 'Gaussian account prior',
    'curve_anchor_local_noise_sd': 'Local-noise account prior',
    'curve_anchor_cauchy': 'Cauchy account prior',
    'discrepancy_context_variance': 'Between-context noise',
    'discrepancy_curve_shrinkage': 'Variance-based curve shrinkage',
    'discrepancy_native_center': 'Native curve offset correction',
    'population_affine_common_account5': 'Population affine + weak account',
    'population_affine_account_prior': 'Population affine + account prior',
    'population_percentile_account_prior': 'Population percentile transport',
    'hierarchical_affine_translated_prior': 'Hierarchical affine · translated prior',
    'hierarchical_affine_gaussian315_prior': 'Hierarchical affine · Gaussian prior',
    'curve_regularized_cauchy': 'Shrunk curve + Cauchy prior',
    'curve_laplace_cauchy_account': 'Laplace accuracy + Cauchy account',
    'curve_laplace_common_account5': 'Laplace accuracy + weak account',
    'curve_information_prior': 'Accuracy-coordinate prior',
    'simple_global_local_inverse': 'Local inverse + weak account',
    'simple_global_population_inverse': 'Population inverse + weak account',
    'simple_global_equal_curve_inverse': 'Equal-curve inverse + weak account',
    'simple_global_arithmetic_secant': 'Arithmetic population secant',
    'simple_global_logloss_secant': 'Log-loss population secant',
    'simple_global_logodds_secant': 'Log-odds population secant',
    'simple_global_local_elo_odds': 'Local Elo-odds analogy',
    'simple_global_population_elo_odds': 'Population Elo-odds analogy',
}


def label(method):
    return LABELS.get(method, method.replace('_', ' ').capitalize())


def number(value, decimals=1):
    return '—' if value is None else f'{value:,.{decimals}f}'


def identifier(row):
    return f"g{row['number']}{row['side'][0]}"


def groups(result):
    """Partition all methods by declared scope, without selecting by scores."""
    inventory = result['method_order']
    if len(set(inventory)) != len(inventory):
        raise ValueError('Method inventory must be unique.')
    if any(result['methods'][method]['scope'] not in SCOPES for method in inventory):
        raise ValueError('Every method needs an explicit mathematical scope.')
    output = []
    for scope in SCOPES:
        selected = [method for method in inventory if result['methods'][method]['scope'] == scope]
        for start in range(0, len(selected), 6):
            output.append((scope, 1+start//6, selected[start:start+6]))
    return output


def scale_rows(result, scale, methods):
    if scale not in result['source_scales']:
        raise ValueError('Unknown source rating scale.')
    records = {method: [] for method in methods}
    seen = set()
    for row in result['players']:
        if row['source_scale'] != scale or row['method'] not in records:
            continue
        key = row['method'], row['game'], row['side']
        if key in seen or row['side'] not in SIDES:
            raise ValueError('Duplicate or unknown player row.')
        seen.add(key)
        if not math.isfinite(row['reference']) or row['estimate'] is not None and not math.isfinite(row['estimate']):
            raise ValueError('Plot values must be finite or missing.')
        records[row['method']].append(row)
    expected = sum(context['scale']['scale'] == scale for context in result['contexts'])*2
    for method, rows in records.items():
        if len(rows) != expected or sum(row['estimate'] is not None for row in rows) != result['summary_by_scale'][scale][method]['players']:
            raise ValueError('Every method must retain every player, including missing estimates.')
    return records


def build_scatter(result, scale, methods, *, title):
    """Up to six square, equal-scale panels with explicit edge/missing labels."""
    if not 1 <= len(methods) <= 6 or len(set(methods)) != len(methods):
        raise ValueError('Choose one to six distinct methods.')
    records = scale_rows(result, scale, methods)
    columns = min(3, len(methods))
    rows = math.ceil(len(methods)/columns)
    fig = Figure(figsize=(7*columns, 7*rows+1.5), facecolor='white')
    FigureCanvasAgg(fig)
    axes = np.asarray(fig.subplots(rows, columns, squeeze=False)).ravel()
    fig.subplots_adjust(left=.05, right=.985, top=1.-2.0/(7*rows+1.5), bottom=1.25/(7*rows+1.5), wspace=.19, hspace=.42)
    scale_name = next(context['scale']['name'] for context in result['contexts'] if context['scale']['scale'] == scale)
    fig.suptitle('Commercial reference Elo vs estimated Elo', fontsize=21, fontweight='bold', y=.985)
    fig.text(.5, 1.-.63/(7*rows+1.5), f'{scale_name} · {title} · same scale on both axes', ha='center', fontsize=12, color='#454f5b')
    pending = []
    for axis, method in zip(axes, methods):
        summary = result['summary_by_scale'][scale][method]
        available = [row for row in records[method] if row['estimate'] is not None]
        missing = [row for row in records[method] if row['estimate'] is None]
        axis.set(xlim=LIMITS, ylim=LIMITS, xlabel='Commercial reference Elo', ylabel='Estimated Elo')
        axis.set_box_aspect(1)
        axis.set_xticks(np.arange(0., 3201., 400.))
        axis.set_yticks(np.arange(0., 3201., 400.))
        axis.tick_params(labelsize=9)
        axis.grid(alpha=.17)
        axis.spines[['top', 'right']].set_visible(False)
        axis.plot(LIMITS, LIMITS, '--', color='#9b9fa5', lw=1.1, zorder=1)
        entries, points = [], []
        for row in available:
            x, y = row['reference'], row['estimate']
            px, py = float(np.clip(x, *LIMITS)), float(np.clip(y, *LIMITS))
            outside = px != x or py != y
            color, marker = (EDGE_COLOR, 'D') if row['edge'] else (NATIVE_COLOR, 'o')
            if outside:
                marker = '^' if y > LIMITS[1] else 'v' if y < LIMITS[0] else '>' if x > LIMITS[1] else '<'
            axis.scatter([px], [py], marker=marker, color=color, s=41 if row['edge'] or outside else 29,
                         alpha=.95 if row['edge'] else .75, edgecolors='white', linewidths=.6, zorder=4, clip_on=not outside)
            if row['conversion_extrapolated']:
                axis.scatter([px], [py], marker='o', s=78, facecolors='none', edgecolors='#584368', linewidths=.8, zorder=3)
            points.append((px, py))
            if row['edge'] or outside:
                text = identifier(row)+(f' ({x:.0f}, {y:.0f})' if outside else '')
                entries.append({'x': px, 'y': py, 'label': text, 'boundary': outside})
        axis.set_title(f"{fill(label(method), 39)}\nMAE {number(summary['mae'])} · native {number(summary['native_mae'])} · edge {number(summary['edge_mae'])}\n"
                       f"{len(available)}/{len(records[method])} estimates · W/B {summary['reference_order_matches']}/{summary['comparable_games']}",
                       fontsize=11, pad=13, linespacing=1.4)
        if missing:
            text = 'No crossing; no point or error assigned: '+', '.join(identifier(row) for row in missing)
            axis.text(.025, .025, fill(text, 48), transform=axis.transAxes, fontsize=8, va='bottom', color='#605765',
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .95})
        pending.append((axis, entries, points))
    for axis in axes[len(methods):]:
        axis.set_visible(False)
    handles = [Line2D([], [], marker='o', ls='none', color=NATIVE_COLOR, label='Native Maia accuracy range'),
               Line2D([], [], marker='D', ls='none', color=EDGE_COLOR, label='Outside native Maia accuracy range'),
               Line2D([], [], marker='o', ls='none', markerfacecolor='none', color='#584368', label='Conversion extrapolated'),
               Line2D([], [], ls='--', color='#9b9fa5', label='Equal estimate and reference')]
    fig.legend(handles=handles, ncols=2 if columns == 1 else 4, frameon=False, fontsize=10,
               loc='lower center', bbox_to_anchor=(.5, .45/(7*rows+1.5)))
    fig.text(.5, .19/(7*rows+1.5), 'Maia native ratings remain 600–2600 Lichess Blitz. Triangles mark values beyond plot limits, with their true values labeled.',
             ha='center', color='#555', fontsize=9)
    fig.canvas.draw()
    for axis, entries, points in pending:
        _label_points(axis, entries, points)
    return fig


def build_curves(contexts):
    """Convert the rating coordinate, preserving every accuracy value exactly."""
    if not 1 <= len(contexts) <= 6 or len({row['scale']['scale'] for row in contexts}) != 1:
        raise ValueError('Choose one to six contexts on the same source scale.')
    columns = min(3, len(contexts)); rows = math.ceil(len(contexts)/columns)
    fig = Figure(figsize=(7*columns, 5.3*rows+1.4), facecolor='white')
    FigureCanvasAgg(fig)
    axes = np.asarray(fig.subplots(rows, columns, squeeze=False)).ravel()
    fig.subplots_adjust(left=.05, right=.985, top=.87, bottom=.12, wspace=.2, hspace=.4)
    fig.suptitle('Shared arithmetic-accuracy curves on the source rating scale', y=.98, fontsize=19, fontweight='bold')
    fig.text(.5, .935, contexts[0]['scale']['name']+' · rating coordinates converted; accuracy values and Maia distributions unchanged', ha='center', fontsize=11, color='#555')
    colors = {'White': '#16828a', 'Black': '#bd6334'}
    for axis, context in zip(axes, contexts):
        x, y = np.asarray(context['source_axis']), np.asarray(context['shared_accuracy'])
        native_x = context['source_native_axis']
        axis.axvspan(native_x[0], native_x[-1], color='#e9eff6', alpha=.7)
        axis.plot(x, y, color='#344b71', lw=1.6)
        axis.scatter(native_x, context['knots'], color='#344b71', s=12, zorder=3)
        for side in SIDES:
            observed = context['accuracy'][side]
            if observed is None:
                continue
            axis.axhline(observed, color=colors[side], lw=1., ls='--', alpha=.9)
            axis.scatter([context['actual'][side]], [51.], marker='^', s=40, color=colors[side], zorder=4)
            axis.text(.02, .96 if side == 'White' else .88,
                      f'{side}: accuracy {observed:.2f} · actual {context["actual"][side]:.0f}',
                      transform=axis.transAxes, va='top', fontsize=9, color=colors[side],
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8})
        axis.set(xlim=(min(0., x[0]), max(3200., math.ceil(x[-1]/200.)*200.)), ylim=(50., 100.),
                 title=f'{context["game"]} · accuracy sigma {math.sqrt(context["variance"]):.2f}',
                 xlabel='Rating on source scale', ylabel='Expected / observed arithmetic accuracy')
        axis.grid(alpha=.17)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(labelsize=9)
    for axis in axes[len(contexts):]:
        axis.set_visible(False)
    fig.text(.5, .035, 'Shaded: canonical Maia 600–2600 support. Triangles at the foot show actual ratings only; their vertical location is not an accuracy.',
             ha='center', fontsize=10, color='#555')
    return fig


def _value(row):
    if row['estimate'] is None:
        return '<span class="muted" title="No finite shared-curve intersection">—</span>'
    text = str(int(np.floor(row['estimate']+.5)))
    if row['edge']:
        text = '<span class="edge">'+text+'◆</span>'
    if row['conversion_extrapolated']:
        text += '<sup title="Converter extrapolated beyond its fitted LB400–2800 range">e</sup>'
    return text


def html_report(result, figures):
    sections, metric_sections, pair_sections = [], [], []
    for scale in result['source_scales']:
        contexts = [row for row in result['contexts'] if row['scale']['scale'] == scale]
        scale_name = escape(contexts[0]['scale']['name'])
        metrics_html = []
        for method in result['method_order']:
            summary = result['summary_by_scale'][scale][method]
            values = [SCOPE_NAMES[result['methods'][method]['scope']], f"{summary['players']}/{len(contexts)*2}",
                      number(summary['mae']), number(summary['edge_mae']), number(summary['maximum_error']), number(summary['bias']),
                      f"{summary['above_reference']} / {summary['below_reference']} / {summary['equal_reference']}",
                      f"{summary['reference_order_matches']}/{summary['comparable_games']}"]
            metrics_html.append('<tr><th title="'+escape(result['methods'][method]['description'], quote=True)+'">'+escape(label(method))+'</th>'+''.join('<td>'+escape(value)+'</td>' for value in values)+'</tr>')
        metric_sections.append(f'<h3>{scale_name}</h3><div class="scroll"><table><thead><tr><th>Method</th><th>Scope</th><th>Players</th><th>MAE</th><th>Edge MAE</th><th>Max error</th><th>Mean signed</th><th>Above / below / equal</th><th>W/B accepted</th></tr></thead><tbody>'+''.join(metrics_html)+'</tbody></table></div>')
        for scope_name, page, methods in groups(result):
            records = scale_rows(result, scale, methods)
            lookup = {(row['number'], row['side'], row['method']): row for rows in records.values() for row in rows}
            table_rows = []
            for context in contexts:
                n = context['number']
                base = {side: lookup[n, side, methods[0]] for side in SIDES}
                values = [' / '.join(number(base[side]['actual'], 0) for side in SIDES),
                          ' / '.join(number(base[side]['reference'], 0) for side in SIDES)]
                values += [' / '.join(_value(lookup[n, side, method]) for side in SIDES) for method in methods]
                table_rows.append('<tr><th>'+escape(context['game'])+'</th>'+''.join('<td>'+value+'</td>' for value in values)+'</tr>')
            pair_sections.append(f'<h3>{scale_name} · {SCOPE_NAMES[scope_name]} {page}</h3><div class="scroll"><table><thead><tr><th>Game</th><th>Actual W / B</th><th>Reference W / B</th>'+''.join('<th>'+escape(label(method))+'<br>W / B</th>' for method in methods)+'</tr></thead><tbody>'+''.join(table_rows)+'</tbody></table></div>')
    for item in figures:
        sections.append('<figure><h3>'+escape(item['title'])+'</h3><a href="'+escape(item['file'], quote=True)+'"><img loading="lazy" src="'+escape(item['file'], quote=True)+'" alt="'+escape(item['title'], quote=True)+'"></a></figure>')
    audit = result['scale_invariance']
    maximum = max(audit['maximum_errors'].values())
    scale_names = ', '.join(dict.fromkeys(row['scale']['name'] for row in result['contexts']))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rating scales — all fixed methods</title><style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#253344;font:15px/1.55 system-ui,Segoe UI,sans-serif}}main{{max-width:1720px;margin:auto;padding:28px}}
h1{{font-size:32px;line-height:1.2}}h2{{margin-top:0}}h3{{font-size:18px;margin-top:26px}}section{{background:white;border:1px solid #dce4ed;border-radius:10px;padding:24px;margin:24px 0}}p{{max-width:1150px}}
a{{color:#1a6394}}nav a{{margin-right:18px}}figure{{margin:22px 0}}img{{width:100%;height:auto;display:block}}.scroll{{overflow-x:auto;margin:12px 0 30px}}table{{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
td,th{{text-align:right;padding:9px 11px;border-bottom:1px solid #dfe5ed;white-space:nowrap}}th:first-child{{text-align:left}}thead th{{background:#edf2f7;white-space:normal;min-width:110px}}tbody tr:hover{{background:#f8fafc}}.muted,small{{color:#647080}}.edge{{color:#a54e0a}}sup{{color:#624576}}
</style></head><body><main><header><h1>All fixed rating methods on the PGN rating scale</h1>
<p>{escape(scale_names)} · {result['games']} games · {2*result['games']} players · {len(result['method_order'])} methods</p>
<p>Actual ratings and commercial references come from the current PGNs. Inference uses Lichess Blitz coordinates internally; only the rating coordinate is converted. The Maia policies, arithmetic accuracy, and candidate formulas remain unchanged.</p>
<nav><a href="#figures">Every method graph</a><a href="#metrics">All metrics</a><a href="#games">Every game and method</a><a href="#audit">Scale audit</a><a href="comparison.json">Full JSON</a><a href="comparison.csv">Full CSV</a></nav></header>
<section id="audit"><h2>What was checked</h2><p>The same canonical actual ratings were represented in Lichess Blitz, Lichess Rapid, Chess.com Blitz, and Chess.com Rapid, normalized back, and supplied to every method. All {audit['checks']:,} player/method/scale checks {'passed' if audit['passed'] else 'did not pass'} the {audit['tolerance']:g}-Elo floating-point tolerance; the largest error was {maximum:.3g} Elo. This checks coordinate consistency, not the empirical accuracy of the conversion model.</p>
<p>Conversion model: <code>{escape(result['conversion_model'])}</code>. Values outside its fitted canonical Lichess Blitz 400–2800 range use its explicitly enabled analytic extension and are marked <sup>e</sup> in tables or ringed in graphs. Missing curve intersections stay missing and are excluded from MAE. A converted point is the original canonical decision mapped to a new coordinate; the estimator is not refitted on converted Maia labels.</p>
<p>Population-only controls are shown separately because they do not retain the target game's curve mean. No commercial labels enter predictions or conversion coefficients. These games have been inspected repeatedly; the comparison is exploratory and is not an independent validation set. Account ±200 sensitivity was not rerun.</p><p><a href="scale-invariance.json">Every scale check</a> · <a href="predictions.json">Predictions saved before reference labels</a> · <a href="contexts.json">Converted curve coordinates and source metadata</a></p></section>
<section id="metrics"><h2>All method metrics</h2><p>Every error uses full-precision estimates in the displayed source scale. Scales are summarized separately. An exact reference tie accepts |estimated White − Black| &lt;50 Elo; otherwise only matching sign is required. Estimated gaps below50 are allowed for non-ties.</p>{''.join(metric_sections)}</section>
<section id="figures"><h2>Every method graph and every shared curve</h2><p>Blue circles: observed accuracy within the native Maia range. Orange diamonds: outside it. All scatter panels use identical 0–3200 source-rating axes. Out-of-axis points are marked by triangles with true coordinates labeled.</p>{''.join(sections)}</section>
<section id="games"><h2>Every game and method</h2><p>Values are White / Black, rounded only for display. ◆ means an accuracy edge case relative to canonical Maia 600–2600. <sup>e</sup> means extrapolated conversion. A dash is unavailable, not a boundary estimate.</p>{''.join(pair_sections)}</section>
<footer class="muted">Generated from completed research results; production settings, saved engine analysis, and earlier comparisons are unchanged.</footer></main></body></html>'''


def export(result, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    figures = []
    for scale in result['source_scales']:
        for scope_name, page, methods in groups(result):
            title = f'{SCOPE_NAMES[scope_name]} · page {page}'
            filename = f'{scale}-{scope_name}-{page}.svg'
            _save_figure(build_scatter(result, scale, methods, title=title), output/filename)
            figures.append({'file': filename, 'title': title})
        contexts = [row for row in result['contexts'] if row['scale']['scale'] == scale]
        for start in range(0, len(contexts), 6):
            filename = f'{scale}-shared-curves-{1+start//6}.svg'
            subset = contexts[start:start+6]
            _save_figure(build_curves(subset), output/filename)
            figures.append({'file': filename, 'title': f'Shared curves · {subset[0]["game"]} to {subset[-1]["game"]}'})
    (output/'index.html').write_text(html_report(result, figures), encoding='utf-8')
    source = Path(__file__)
    comparison = output/'comparison.json'
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'renderer': source.name,
                'renderer_sha256': sha256(source.read_bytes()).hexdigest(),
                'comparison_sha256': sha256(comparison.read_bytes()).hexdigest() if comparison.is_file() else None,
                'figures': [{**item, 'sha256': sha256((output/item['file']).read_bytes()).hexdigest()} for item in figures],
                'scope': 'Rendering only; numerical predictions and their run-time source audit are unchanged.'}
    (output/'figure-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return {'html': str(output/'index.html'), 'figures': figures}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(json.loads(args.input.read_text(encoding='utf-8')), args.output)
