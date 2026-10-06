"""Export the completed universal-method comparison as tables and SVG figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

from analysis.cache import write_json
from tests.analysis.experiment_shared_curve_sweep import _write_csv
from tests.analysis.experiment_universal_rating import OUTPUT


def export(source, method='uncertainty_consensus'):
    source = Path(source)
    result = json.loads(source.read_text(encoding='utf-8'))
    rows = [r for r in result['players'] if r['method'] == method]
    if not rows:
        raise ValueError('Selected method has no predictions in this comparison.')
    metrics = next(r for r in result['ranking'] if r['method'] == method)
    games = {}
    for row in rows:
        record = games.setdefault(row['game'], {'game': row['game']})
        for key in ('reference', 'estimate', 'original_estimate', 'actual', 'edge'):
            record[row['side'].lower()+'_'+key] = row[key]
        record[row['side'].lower()+'_absolute_error'] = abs(row['estimate']-row['reference'])
    target = source.parent/method
    target.mkdir(parents=True, exist_ok=True)
    _write_csv(target/'game-comparison.csv', list(games.values()))
    write_json(target/'result.json', {'method': method, 'metrics': metrics, 'games': list(games.values()),
                                     'source': str(source.resolve()), 'scope': result['scope'],
                                     'figure_generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    reference = np.array([r['reference'] for r in rows])
    edge = np.array([r['edge'] for r in rows])
    for axis, key, title in zip(axes, ('original_estimate', 'estimate'),
                               ('Existing shared-curve fit', 'Universal uncertainty estimate'), strict=True):
        values = np.array([r[key] for r in rows])
        axis.plot([200, 3000], [200, 3000], color='#87919e', linewidth=1, label='Equal to reference')
        axis.scatter(reference[~edge], values[~edge], color='#2563a6', s=35, label=f'Other {sum(~edge)} players')
        axis.scatter(reference[edge], values[edge], color='#ce6d16', s=55, marker='D', label=f'{sum(edge)} edge players')
        axis.set(xlim=(200, 3000), ylim=(200, 3000), xlabel='Commercial reference Elo', ylabel='Estimated Elo',
                 title=f'{title}\nMAE {np.mean(abs(values-reference)):.2f}; maximum {max(abs(values-reference)):.2f}')
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.18)
    axes[0].legend(loc='upper left', fontsize=8)
    figure.suptitle(f'All {len(games)} games / {len(rows)} players — exploratory method comparison', fontsize=13)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis, key in zip(axes, ('original_estimate', 'estimate'), strict=True):
        occupied = []
        for row in sorted((r for r in rows if r['edge']), key=lambda r: r[key]):
            label = axis.annotate(row['game']+' '+row['side'][0], (row['reference'], row[key]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8,
                                  arrowprops={'arrowstyle': '-', 'color': '#87919e', 'linewidth': .5})
            for offset in range(5, 78, 12):
                label.set_position((5, offset))
                label.update_positions(renderer)
                box = Text.get_window_extent(label, renderer).expanded(1.02, 1.02)
                if not any(box.overlaps(other) for other in occupied):
                    break
            occupied.append(box)
    figure.savefig(target/'comparison.svg')
    plt.close(figure)

    edge_rows = [r for r in rows if r['edge']]
    edge_rows.sort(key=lambda r: r['reference'], reverse=True)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(edge_rows))
    for key, label, marker, color in (
            ('reference', 'Commercial reference', 'o', '#273549'),
            ('original_estimate', 'Existing fit', 's', '#939cab'),
            ('estimate', 'Universal estimate', 'D', '#2563a6'),
            ('actual', 'Actual Elo', '^', '#ce6d16')):
        axis.plot(x, [r[key] for r in edge_rows], marker=marker, color=color, label=label, linewidth=1.3)
    axis.set(xticks=x, xticklabels=[r['game']+' '+r['side'] for r in edge_rows], ylim=(200, 3000),
             ylabel='Elo', title=f'Fixed {len(edge_rows)} edge players — MAE {metrics["edge_mae"]:.2f} Elo')
    axis.grid(axis='y', alpha=.2)
    axis.legend(loc='lower left')
    figure.savefig(target/'edge-comparison.svg')
    plt.close(figure)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUTPUT/'comparison.json')
    parser.add_argument('--method', default='uncertainty_consensus')
    args = parser.parse_args()
    print(export(args.source, args.method))
