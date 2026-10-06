"""Read-only game17 measurement diagnosis; export research SVG and numeric JSON."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analysis.player_rating.uncertainty_measurement import measure
from analysis.settings import CONFIG
from tests.analysis.compare_rating_methods import load_cases
from tests.analysis.simple_curve_account import predict

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/simple-rating-restart'
SIDES = ('White', 'Black')
COLORS = {'White': '#087f8c', 'Black': '#bd6334'}


def diagnostic_data(case):
    evidence = case['evidence']
    actual = {side: evidence[side]['actual_rating'] for side in SIDES}
    arithmetic, competitive = measure(evidence), measure(evidence, .5)
    simple = predict(evidence, actual)['simple_posterior_mean_account10']
    production = json.loads((case['analysis_path'].parent/'player-rating/fit.json').read_text(encoding='utf-8'))
    bayesian = json.loads((ROOT/'tests/analysis/output/rating-methods-games0-18/game17/bayesian_shared_curve/fit.json').read_text(encoding='utf-8'))
    curve = arithmetic['curve']
    grid, accuracy_curve = np.asarray(curve['fine_ratings']), np.asarray(curve['shared_accuracy'])
    players = {}
    for side in SIDES:
        rows = evidence[side]['observations']
        moves = [move for move in case['analysis']['moves'] if move['side'].title() == side]
        if len(rows) != len(moves):
            raise ValueError('Evidence and played moves must align.')
        included = [(row, move) for row, move in zip(rows, moves, strict=True) if len(row['qualities']['position']) > 1]
        weights = np.asarray([np.sqrt(4*row['position_win_probability']*(1-row['position_win_probability']))
                              for row, _ in included])
        if weights.sum() <= 0:
            weights[:] = 1.
        normalized = weights/weights.sum()
        ordinary_weight = 1./len(included)
        changes = []
        for (row, move), weight, fraction in zip(included, weights, normalized, strict=True):
            accuracy = row['qualities']['position'][row['played_index']]
            changes.append({'ply': move['ply'], 'move': move['label'], 'accuracy': accuracy,
                            'before_white_win_probability': row['position_win_probability'],
                            'raw_competitive_weight': float(weight), 'normalized_weight': float(fraction),
                            'arithmetic_weight': ordinary_weight,
                            'weight_multiplier': float(fraction/ordinary_weight),
                            'weighted_minus_arithmetic_loss_contribution': float((fraction-ordinary_weight)*(100.-accuracy)),
                            'saved_flags': move.get('flags', [])})
        observed = arithmetic['sides'][side]['accuracy']
        intersects = bool(accuracy_curve[0] <= observed <= accuracy_curve[-1])
        intersection = float(np.interp(observed, accuracy_curve, grid)) if intersects else None
        weighted = competitive['sides'][side]['accuracy']
        if not np.isclose(sum(row['weighted_minus_arithmetic_loss_contribution'] for row in changes),
                          observed-weighted, atol=1e-10):
            raise AssertionError('Per-move contribution decomposition does not match the aggregate change.')
        players[side] = {'actual_rating': actual[side], 'moves_used': len(included),
                         'forced_positions_removed': len(rows)-len(included),
                         'arithmetic_accuracy': observed, 'competitive_accuracy': weighted,
                         'accuracy_change_from_weighting': weighted-observed,
                         'shared_curve_intersection': intersection,
                         'effective_competitive_moves': competitive['sides'][side]['effective_moves'],
                         'largest_weighting_changes': sorted(changes, key=lambda row: -abs(row['weighted_minus_arithmetic_loss_contribution']))[:8]}
    # Reference ratings are display data only; no calculation above uses them.
    rating_points = {
        'commercial_reference': {side: int(case['game'].headers[side+'EloEstimate']) for side in SIDES},
        'current_production': {side: production['players'][side]['unrounded_estimate'] for side in SIDES},
        'original_bayesian': {side: bayesian['players'][side]['unrounded_estimate'] for side in SIDES},
        'simple_posterior_mean_account10': simple,
    }
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'game': 'game17', 'players': players,
            'rating_points': rating_points,
            'rating_gaps_white_minus_black': {name: pair['White']-pair['Black'] for name, pair in rating_points.items()},
            'accuracy_gaps_white_minus_black': {name: players['White'][name]-players['Black'][name]
                                               for name in ('arithmetic_accuracy', 'competitive_accuracy')},
            'curve': {'ratings': grid.tolist(), 'arithmetic_accuracy': accuracy_curve.tolist()},
            'scope': 'Diagnostic only. Actual ratings are positions on the rating axis, not measured accuracies. The figure isolates the observed-measurement change; competitive reference curves and component inference also change. No estimator is selected or tuned.'}


def build_figure(data):
    fig = Figure(figsize=(12, 5), facecolor='white')
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.42, 1.]})
    fig.subplots_adjust(left=.07, right=.98, top=.81, bottom=.24, wspace=.27)
    fig.suptitle('Game17: similar arithmetic accuracy, different reweighted accuracy', fontsize=14, y=.98)
    fig.text(.5, .915, 'The same saved move qualities are used in both measurements; single-legal-move positions are excluded.',
             ha='center', fontsize=10, color='#4b5563')
    grid = np.asarray(data['curve']['ratings'])
    curve = np.asarray(data['curve']['arithmetic_accuracy'])
    left.axvspan(200, 600, color='#f1f3f5', zorder=0)
    left.axvspan(2600, 3000, color='#f1f3f5', zorder=0)
    left.plot(grid, curve, color='#263548', lw=2, label='Shared arithmetic Maia curve')
    for side, offset in (('White', (-115, -42)), ('Black', (35, 18))):
        player, color = data['players'][side], COLORS[side]
        observed, intersection = player['arithmetic_accuracy'], player['shared_curve_intersection']
        left.axhline(observed, color=color, ls='--', lw=1.1, alpha=.7)
        if intersection is not None:
            left.scatter(intersection, observed, color=color, s=34, zorder=5)
            left.annotate(f'{side} {observed:.2f}%\nintersection {intersection:.0f}',
                          (intersection, observed), xytext=offset, textcoords='offset points',
                          color=color, fontsize=9, ha='left',
                          arrowprops={'arrowstyle': '-', 'color': color, 'lw': .8},
                          bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 1.5})
        actual = player['actual_rating']
        left.plot([actual, actual], [50, 52], color=color, lw=2, clip_on=False)
    left.set(xlim=(200, 3000), ylim=(50, 100), xlabel='Rating hypothesis', ylabel='Move accuracy (%)')
    left.set_xticks(np.arange(200, 3001, 400))
    left.set_title('Shared arithmetic accuracy curve', fontsize=11, pad=12)
    left.legend(loc='upper left', frameon=False, fontsize=9)
    left.text(.02, .03, f"Actual Elo axis markers: White {data['players']['White']['actual_rating']:.0f} / "
              f"Black {data['players']['Black']['actual_rating']:.0f}", transform=left.transAxes,
              fontsize=8.5, color='#4b5563', va='bottom')
    left.grid(alpha=.18)
    for side in SIDES:
        player, color = data['players'][side], COLORS[side]
        values = [player['arithmetic_accuracy'], player['competitive_accuracy']]
        right.plot([0, 1], values, 'o-', color=color, lw=2.3, ms=6, label=side)
        for x, value in enumerate(values):
            dy = -17 if side == 'White' else 10
            right.annotate(f'{value:.2f}', (x, value), xytext=(0, dy), textcoords='offset points',
                           ha='center', color=color, fontsize=10)
    gaps = data['accuracy_gaps_white_minus_black']
    right.text(.5, .96, f"Absolute W/B gap: {abs(gaps['arithmetic_accuracy']):.2f} → "
               f"{abs(gaps['competitive_accuracy']):.2f} points", ha='center', va='top',
               transform=right.transAxes, fontsize=10, color='#263548')
    right.set(xlim=(-.23, 1.23), ylim=(88, 93), ylabel='Observed mean accuracy (%) · enlarged scale')
    right.set_xticks([0, 1], ['Arithmetic\nequal position weights', 'Competitive\nweights √[4p(1−p)]'])
    right.set_title('Effect on the observed means', fontsize=11, pad=12)
    right.legend(loc='lower center', ncols=2, frameon=False, fontsize=9)
    right.grid(axis='y', alpha=.18)
    labels = [('commercial_reference', 'Reference'), ('current_production', 'Production ensemble'),
              ('original_bayesian', 'Arithmetic Bayesian median'),
              ('simple_posterior_mean_account10', 'Arithmetic posterior mean + 10% actual')]
    summaries = [f"{label}: {data['rating_points'][name]['White']:.0f} / {data['rating_points'][name]['Black']:.0f}"
                 for name, label in labels]
    fig.text(.07, .1, 'Rating points (White / Black)   |   '+'   |   '.join(summaries[:2]), fontsize=9.5, color='#263548')
    fig.text(.07, .057, '   |   '.join(summaries[2:]), fontsize=9.5, color='#263548')
    fig.text(.07, .014, 'This is a measurement diagnosis, not evidence that any candidate is the better estimator.',
             fontsize=8.5, color='#5b6470')
    return fig


def main():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    case = load_cases(ROOT/'games', [17], CONFIG['ANALYSIS']['CACHE_DIR'])[0]
    paths = set(case['inputs']) | {ROOT/'config.yaml', ROOT/'analysis/player_rating/data/maia_accuracy_calibration.json',
                                 case['analysis_path'].parent/'player-rating/fit.json',
                                 ROOT/'tests/analysis/output/rating-methods-games0-18/game17/bayesian_shared_curve/fit.json'}
    before = {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}
    data = diagnostic_data(case)
    with rc_context({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 10}):
        figure = build_figure(data)
        if any(sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in before.items()):
            raise RuntimeError('Saved evidence changed during diagnostic generation.')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        figure.savefig(OUTPUT/'game17-diagnosis.svg', format='svg')
    data['protected_input_hashes'] = before
    (OUTPUT/'game17-diagnosis.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'svg': str(OUTPUT/'game17-diagnosis.svg'), 'accuracy_gaps': data['accuracy_gaps_white_minus_black'],
                      'rating_gaps': data['rating_gaps_white_minus_black']}, indent=2))


if __name__ == '__main__':
    main()
