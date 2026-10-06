"""Compare declared accuracy measurements under one unchanged affine estimator.

Every population curve is recomputed from the same hash-verified frozen sixteen
numeric contexts. Actual PGN Elo is converted to native Lichess Blitz; reference
labels are read only after every estimate is saved. No production files change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import csv

import numpy as np

from analysis.cache import write_json
from analysis.player_rating.bayesian_shared_curve import SharedCurve
from analysis.player_rating.calibration import DEFAULT_PATH, evidence_fingerprint, load_calibration
from analysis.player_rating.shared_curve_affine import (CURVE_ARGS, _account_anchor,
    affine_moments, translated_prior)
from tests.analysis.curve_hierarchical_candidates import shrinkage
from analysis.player_rating.scale import from_native, normalize_evidence, rating_context
from analysis.settings import CONFIG
from tests.analysis.compare_blitz_rating_methods import current_pgn_context, intersection
from tests.analysis.compare_joint_rating_directions import matches_pair
from tests.analysis.compare_rating_methods import load_cases
from tests.analysis.test_player_rating_arithmetic_coverage import recorded_rating_cases
from tests.analysis import lichess_measurement
from tests.analysis import hierarchical_accuracy_fusion as fusion
from tests.analysis import hierarchical_joint_accuracy as joint


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/hierarchical-accuracy'
SIDES = ('White', 'Black')
METHODS = ('arithmetic', 'volatility_weighted', 'lichess_plugin', 'lichess_policy_expectation', 'lichess_reciprocal_plugin')
FUSION_METHODS = {**fusion.METHODS, 'volatility_center_arithmetic_contrast': 'volatility_weighted'}
ALL_METHODS = (*METHODS, *FUSION_METHODS, *joint.METHODS)
PRIMARY_METHODS = ('arithmetic', 'volatility_weighted', 'volatility_center_arithmetic_contrast',
                   'lichess_policy_expectation', *joint.METHODS)
NAMES = {'arithmetic': 'Arithmetic baseline', 'lichess_plugin': 'Lichess plug-in',
         'volatility_weighted': 'Volatility-weighted mean',
         'volatility_center_arithmetic_contrast': 'Volatility-weighted center + arithmetic gap',
         'lichess_policy_expectation': 'Expected Lichess aggregate',
         'lichess_reciprocal_plugin': 'Reciprocal plug-in (diagnostic)',
         'lichess_plugin_center_arithmetic_contrast': 'Lichess plug-in center + arithmetic gap',
         'lichess_policy_center_arithmetic_contrast': 'Expected Lichess center + arithmetic gap',
         'hierarchical_joint_accuracy': 'Joint arithmetic + Lichess',
         'hierarchical_joint_center_arithmetic_contrast': 'Joint center + arithmetic gap'}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def population_evidence(evidence):
    """Keep numeric contexts and volatility weights; erase played choices/accounts."""
    return {side: {key: value for key, value in record.items() if key != 'actual_rating'} |
            {'observations': [{**row, 'played_index': 0} for row in record['observations']]}
            for side, record in evidence.items()}


def measurement(evidence, method, cache, *, samples, seed):
    fingerprint = evidence_fingerprint(evidence)
    context_seed = (seed+int(fingerprint[:8], 16)) % (2**32)
    module_hash = sha256(Path(lichess_measurement.__file__).read_bytes()).hexdigest()
    key = sha256(json.dumps([method, samples, context_seed, module_hash, jsonable(evidence)],
                           sort_keys=True, allow_nan=False).encode()).hexdigest()
    path = cache/(key+'.json')
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    value = jsonable(lichess_measurement.measure(evidence, method=method, samples=samples, seed=context_seed))
    write_json(path, value)
    return value


def fit_measurement(measured, actual, records, fingerprint):
    """Change only measurement; reuse production shrinkage, prior and action."""
    retained = [value for key, value in records.items() if key != fingerprint]
    if not retained:
        raise ValueError('No population context remains after target exclusion.')
    curve = measured['curve']
    knots = np.asarray([row['curve']['monotone_expected_accuracy'] for row in retained])
    population = np.mean([SharedCurve(row)(CURVE_ARGS.grid) for row in knots], axis=0)
    between = float(np.var(knots, axis=0, ddof=int(len(knots) > 1)).mean())
    variance = float(curve['likelihood']['accuracy_variance'])
    model, weight, noise = shrinkage(curve['shared_accuracy'], population, between, variance)
    anchor, _ = _account_anchor(actual)
    prior = translated_prior(anchor)
    affine = affine_moments(model, noise, prior)
    observed = {side: measured['sides'][side]['accuracy'] if measured['sides'][side] else None for side in SIDES}
    unbounded = {side: (prior['prior_mean']+affine['affine_slope']*(accuracy-affine['accuracy_mean'])
                       if curve['identifiable'] and accuracy is not None else None)
                 for side, accuracy in observed.items()}
    points = {side: float(np.clip(value, *CURVE_ARGS.rating_range)) if value is not None else None for side, value in unbounded.items()}
    return {'players': points, 'unbounded_players': unbounded, 'observed': observed,
            'hierarchy': {'measurement_variance': variance, 'between_context_variance': between,
                          'residual_variance': noise, 'local_weight': weight,
                          'contexts_used': len(retained), 'contexts_excluded': len(records)-len(retained)},
            'affine': {**{key: value for key, value in prior.items() if not isinstance(value, np.ndarray)}, **affine},
            'curve': {'grid': CURVE_ARGS.grid.tolist(), 'local': curve['shared_accuracy'],
                      'population': population.tolist(), 'model': model.tolist()}}


def metrics(rows):
    valid = [row for row in rows if row['estimate'] is not None]
    errors = np.array([row['estimate']-row['reference'] for row in valid])
    by_game = {}
    for row in valid:
        by_game.setdefault(row['game'], {})[row['side']] = row
    pairs = []
    for game, pair in by_game.items():
        if len(pair) != 2:
            continue
        white, black = (pair[side] for side in SIDES)
        gap, reference_gap = white['estimate']-black['estimate'], white['reference']-black['reference']
        original_gap = white['arithmetic_accuracy']-black['arithmetic_accuracy']
        pairs.append({'game': game, 'estimate_gap': gap, 'reference_gap': reference_gap,
                      'arithmetic_accuracy_gap': original_gap,
                      'arithmetic_order_match': bool(np.sign(gap) == np.sign(original_gap)),
                      'reference_order_match': bool(matches_pair(reference_gap, gap)),
                      'reference_tie': reference_gap == 0})
    def subset(flag):
        values = [row['estimate']-row['reference'] for row in valid if row[flag]]
        return {'players': len(values), 'mae': float(np.mean(np.abs(values))) if values else None,
                'bias': float(np.mean(values)) if values else None,
                'maximum_error': max(map(abs, values), default=None)}
    return {'players': len(valid), 'mae': float(np.mean(abs(errors))), 'bias': float(errors.mean()),
            'maximum_error': float(abs(errors).max()), 'over': int((errors > 0).sum()),
            'under': int((errors < 0).sum()), 'equal': int((errors == 0).sum()),
            'edge': subset('edge'), 'low_accuracy_quartile': subset('low_accuracy_quartile'),
            'arithmetic_order_matches': sum(pair['arithmetic_order_match'] for pair in pairs),
            'reference_order_matches': sum(pair['reference_order_match'] for pair in pairs),
            'games': len(pairs), 'pairs': pairs}


def figures(predictions, scored, output):
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    colors = {'White': '#16828a', 'Black': '#bd6334'}
    plotted = [method for method in METHODS if method in scored]
    for game, record in predictions.items():
        columns = min(3, len(plotted)); height = (len(plotted)+columns-1)//columns
        fig = Figure(figsize=(6.7*columns, 4.8*height), layout='constrained'); FigureCanvasAgg(fig)
        fig.suptitle(game+' · same hierarchical-affine rule, different accuracy measurements', fontsize=14)
        axes = list(fig.subplots(height, columns, squeeze=False).flat)
        for axis, method in zip(axes, plotted):
            fit = record['methods'][method]
            curve = fit['curve']
            x = from_native(curve['grid'], record['rating_scale'])
            for key, label, color, style in (('local', 'Game curve', '#344154', '-'),
                    ('population', 'Population', '#8f79a8', '--'), ('model', 'Blended curve', '#217758', '-')):
                axis.plot(x, curve[key], label=label, color=color, ls=style, lw=1.6)
            for side in SIDES:
                value = fit['observed'][side]
                if value is not None:
                    axis.axhline(value, color=colors[side], ls=':', label=f'{side} observed {value:.2f}')
            displayed = [from_native(fit['players'][side], record['rating_scale']) for side in SIDES]
            pair = ' / '.join('—' if value is None else f'{value:.0f}' for value in displayed)
            axis.set_title(NAMES[method]+f' · W / B: {pair}\n'+
                f'Game weight {fit["hierarchy"]["local_weight"]:.3f}; σ={np.sqrt(fit["hierarchy"]["measurement_variance"]):.3f}', fontsize=10)
            axis.set(xlim=from_native([200., 3000.], record['rating_scale']), ylim=(0., 100.),
                     xlabel='Rating ('+record['rating_scale']['name']+')', ylabel='Measurement score (0–100)')
            axis.grid(alpha=.2); axis.spines[['top', 'right']].set_visible(False)
            axis.legend(fontsize=8, loc='lower left')
        for axis in axes[len(plotted):]:
            axis.set_visible(False)
        directory = output/game; directory.mkdir(exist_ok=True)
        with rc_context({'svg.fonttype': 'none'}):
            fig.savefig(directory/'measurement-curves.svg', facecolor='white')
        fig.clear()
    fig = Figure(figsize=(19., 6.), layout='constrained'); FigureCanvasAgg(fig)
    error, bias, counts = fig.subplots(1, 3)
    all_methods = [method for method in ALL_METHODS if method in scored]
    short = {'arithmetic': 'Arithmetic', 'volatility_weighted': 'Weighted', 'lichess_plugin': 'Plug-in',
             'lichess_policy_expectation': 'Expected aggregate', 'lichess_reciprocal_plugin': 'Reciprocal diagnostic',
             'lichess_plugin_center_arithmetic_contrast': 'Plug-in center/gap',
             'lichess_policy_center_arithmetic_contrast': 'Expected center/gap',
             'volatility_center_arithmetic_contrast': 'Weighted center/gap',
             'hierarchical_joint_accuracy': 'Joint', 'hierarchical_joint_center_arithmetic_contrast': 'Joint center/gap'}
    labels = [short[method] for method in all_methods]
    for axis, key, title in ((error, 'mae', 'Mean absolute error'), (bias, 'bias', 'Mean signed error')):
        axis.bar(range(len(all_methods)), [scored[method][key] for method in all_methods], color='#647b96')
        axis.set_title(title); axis.set_ylabel('Chess.com Rapid Elo'); axis.axhline(0., color='#444', lw=.8)
        axis.set_xticks(range(len(all_methods)), labels, fontsize=9, rotation=60, ha='right'); axis.spines[['top', 'right']].set_visible(False)
    over = np.array([scored[method]['over'] for method in all_methods])
    under = np.array([scored[method]['under'] for method in all_methods])
    counts.bar(range(len(all_methods)), over, label='Above reference', color='#bd6334')
    counts.bar(range(len(all_methods)), under, bottom=over, label='Below reference', color='#16828a')
    counts.set_title('Directional balance'); counts.set_ylabel('Players'); counts.set_xticks(range(len(all_methods)), labels, fontsize=9, rotation=60, ha='right')
    counts.legend(fontsize=9); counts.spines[['top', 'right']].set_visible(False)
    with rc_context({'svg.fonttype': 'none'}):
        fig.savefig(output/'comparison.svg', facecolor='white')
    fig.clear()


def scatter_figure(rows, scored, output, *, methods=None, filename='scatter.svg'):
    """Compare saved estimates without recalculating any measurement or fit."""
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    methods = tuple(methods or [method for method in ALL_METHODS if method in scored])
    columns = min(3, len(methods)); height = (len(methods)+columns-1)//columns
    fig = Figure(figsize=(5.2*columns, 4.8*height), layout='constrained'); FigureCanvasAgg(fig)
    fig.suptitle('Same hierarchical-affine estimator · alternative accuracy measurements\n'
                 'Original arithmetic native-curve edge cases are highlighted', fontsize=13)
    axes = list(fig.subplots(height, columns, squeeze=False).flat)
    finite = [row for row in rows if row['method'] in methods and row['estimate'] is not None]
    values = [value for row in finite for value in (row['reference'], row['estimate'])]
    limits = (200.*np.floor(min(values)/200.), 200.*np.ceil(max(values)/200.))
    for axis, method in zip(axes, methods):
        points = [row for row in finite if row['method'] == method]
        for edge, color, label in ((False, '#34678c', 'Native intersection'), (True, '#c56828', 'Original edge case')):
            group = [row for row in points if row['edge'] == edge]
            axis.scatter([row['reference'] for row in group], [row['estimate'] for row in group],
                         c=color, s=27 if not edge else 42, alpha=.8, label=label, edgecolors='white', linewidths=.4)
            if edge:
                for row in group:
                    axis.annotate(row['game'].replace('game', 'g')+row['side'][0],
                                  (row['reference'], row['estimate']), xytext=(4, 3),
                                  textcoords='offset points', fontsize=6.5, color=color)
        axis.plot(limits, limits, '--', color='#777777', lw=1.)
        score = scored[method]
        axis.set_title(NAMES[method]+f'\nMAE {score["mae"]:.1f}; bias {score["bias"]:+.1f}; '
                       f'order {score["reference_order_matches"]}/{score["games"]}', fontsize=10)
        axis.set(xlim=limits, ylim=limits, xlabel='Commercial reference (Chess.com Rapid)',
                 ylabel='Estimated rating (Chess.com Rapid)')
        axis.set_aspect('equal', adjustable='box'); axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(loc='upper left', fontsize=7)
    for axis in axes[len(methods):]:
        axis.set_visible(False)
    with rc_context({'svg.fonttype': 'none'}):
        fig.savefig(Path(output)/filename, facecolor='white')
    fig.clear()


def write_tables(rows, output):
    """Mark original arithmetic non-intersections consistently in saved tables."""
    games = sorted({row['game'] for row in rows}, key=lambda name: int(name.removeprefix('game')))
    for filename, methods in (('comparison.md', ALL_METHODS), ('primary-comparison.md', PRIMARY_METHODS)):
        methods = [method for method in methods if any(row['method'] == method for row in rows)]
        columns = ['Game', 'Reference W / B', *(NAMES[method]+' W / B' for method in methods)]
        table = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
        for game in games:
            baseline = {row['side']: row for row in rows if row['game'] == game and row['method'] == 'arithmetic'}
            def pair(values):
                return ' / '.join(('—' if values[side] is None else str(int(np.floor(values[side]+.5))))+
                                  ('*' if baseline[side]['edge'] else '') for side in SIDES)
            name = game+('*' if any(row['edge'] for row in baseline.values()) else '')
            values = [name, pair({side: baseline[side]['reference'] for side in SIDES}),
                      *(pair({row['side']: row['estimate'] for row in rows if row['game'] == game and row['method'] == method}) for method in methods)]
            table.append('| '+' | '.join(values)+' |')
        table.extend(['', '* marks a player whose original arithmetic accuracy has no intersection within the measured Maia 600–2600 shared curve; the game name is marked when either player qualifies.',
                      '', 'All displayed ratings are Chess.com Rapid, rounded only for presentation. Metrics use unrounded estimates.'])
        (Path(output)/filename).write_text('\n'.join(table)+'\n', encoding='utf-8')


def run(output=OUTPUT, *, samples=16384, seed=0, plots=True):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    # Historical body is unreachable; its estimator was removed.
    from analysis.player_rating.hierarchical_affine import Rating
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'tests/analysis/output'):
        raise ValueError('Research output must remain inside tests/analysis/output.')
    output.mkdir(parents=True, exist_ok=True)
    cache = output/'measurements'; cache.mkdir(exist_ok=True)
    started = perf_counter()
    corpus = load_calibration()
    frozen = recorded_rating_cases([f'game{i}' for i in range(16)])
    if {evidence_fingerprint(case['evidence']) for case in frozen} != {record.fingerprint for record in corpus.records}:
        raise ValueError('Archived numeric contexts do not match the frozen population.')
    cases = load_cases(ROOT/'games', range(24), Path(CONFIG['ANALYSIS']['CACHE_DIR']))
    protected = {path: digest for case in [*frozen, *cases] for path, digest in case['inputs'].items()}
    for path in (ROOT/'config.yaml', DEFAULT_PATH, Path(__file__), Path(lichess_measurement.__file__), Path(fusion.__file__), Path(joint.__file__),
                 *sorted((ROOT/'analysis').rglob('*.py'))):
        protected[path] = sha256(path.read_bytes()).hexdigest()
    write_json(output/'design.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'methods': {**lichess_measurement.describe(), **fusion.describe(), **joint.describe(),
                    'volatility_center_arithmetic_contrast': 'Volatility-weighted native pair center plus original arithmetic affine gap; the same declared constrained point decision.'}, 'samples': samples, 'base_seed': seed,
        'calibration_hash': corpus.content_hash, 'low_accuracy_subset': 'Lowest twelve of forty-eight original arithmetic accuracies, chosen without labels.',
        'rule': 'One unchanged production hierarchical-affine calculation; all legal moves; frozen16 numeric contexts remeasured consistently; references join after inference.',
        'input_hashes': {str(path): digest for path, digest in protected.items()}})
    populations = {method: {} for method in METHODS}
    for index, case in enumerate(frozen, 1):
        fingerprint = evidence_fingerprint(case['evidence'])
        evidence = population_evidence(case['evidence'])
        for method in METHODS:
            populations[method][fingerprint] = measurement(evidence, method, cache, samples=samples, seed=seed)
        expected = next(record.arithmetic.knots for record in corpus.records if record.fingerprint == fingerprint)
        np.testing.assert_allclose(populations['arithmetic'][fingerprint]['curve']['monotone_expected_accuracy'], expected, atol=1e-10, rtol=0.)
        print(f'Frozen population {index}/16: all measurements prepared.', flush=True)
    write_json(output/'population-moments.json', populations)
    predictions, rows = {}, []
    for case in cases:
        source_evidence = current_pgn_context(case)
        context = rating_context(case['game'].headers)
        evidence = normalize_evidence(source_evidence, context)
        actual = {side: evidence[side]['actual_rating'] for side in SIDES}
        fingerprint = evidence_fingerprint(evidence)
        measurements = {method: measurement(evidence, method, cache, samples=samples, seed=seed) for method in METHODS}
        fits = {method: fit_measurement(value, actual, populations[method], fingerprint) for method, value in measurements.items()}
        baseline = Rating().fit(evidence)
        for side in SIDES:
            np.testing.assert_allclose(fits['arithmetic']['players'][side], baseline['players'][side]['unrounded_estimate'], atol=1e-8, rtol=0.)
        for method, source in FUSION_METHODS.items():
            combined = fusion.preserve_arithmetic_contrast(fits['arithmetic']['unbounded_players'], fits[source]['unbounded_players'])
            fits[method] = {**fits[source], 'players': combined['players'], 'unbounded_players': combined['diagnostics']['unclipped'],
                            'fusion': combined['diagnostics']}
        fits[joint.METHODS[0]] = joint.fit_joint(measurements['lichess_policy_expectation'], actual,
                                              populations['lichess_policy_expectation'], fingerprint)
        combined = fusion.preserve_arithmetic_contrast(fits['arithmetic']['unbounded_players'],
                                                       fits[joint.METHODS[0]]['unbounded_players'])
        fits[joint.METHODS[1]] = {**fits[joint.METHODS[0]], 'players': combined['players'],
                                'unbounded_players': combined['diagnostics']['unclipped'], 'fusion': combined['diagnostics']}
        fits = jsonable(fits)
        predictions[case['name']] = {'rating_scale': context, 'methods': fits}
        for method, fit in fits.items():
            for side in SIDES:
                original = measurements['arithmetic']['sides'][side]['accuracy']
                crossing = intersection(measurements['arithmetic']['curve']['monotone_expected_accuracy'], original)
                rows.append({'game': case['name'], 'side': side, 'method': method,
                    'actual': source_evidence[side]['actual_rating'], 'actual_native': actual[side],
                    'arithmetic_accuracy': original, 'observed_score': fit['observed'][side],
                    'native_estimate': fit['players'][side], 'estimate': from_native(fit['players'][side], context),
                    'edge': crossing['edge'], 'intersection_status': crossing['intersection_status']})
        print(case['name']+': all estimates saved without reference labels.', flush=True)
    lowest = sorted([row for row in rows if row['method'] == 'arithmetic'], key=lambda row: (row['arithmetic_accuracy'], row['game'], row['side']))[:12]
    low_keys = {(row['game'], row['side']) for row in lowest}
    for row in rows:
        row['low_accuracy_quartile'] = (row['game'], row['side']) in low_keys
    write_json(output/'predictions.json', predictions)
    write_json(output/'predicted-players.json', rows)
    references = {case['name']: {side: int(case['game'].headers[side+'EloEstimate']) for side in SIDES} for case in cases}
    rows = [{**row, 'reference': references[row['game']][row['side']]} for row in rows]
    summary = {method: metrics([row for row in rows if row['method'] == method]) for method in ALL_METHODS}
    for path, digest in protected.items():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Protected source changed: {path}')
    result = {'summary': summary, 'players': rows, 'runtime_seconds': perf_counter()-started,
              'calibration_hash': corpus.content_hash, 'samples': samples, 'seed': seed,
              'evaluation_note': 'Exploratory repeated comparison, not independent external validation; reference labels entered only after every fit.'}
    write_json(output/'comparison.json', result)
    with (output/'players.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_tables(rows, output)
    if plots:
        figures(predictions, summary, output)
        scatter_figure(rows, summary, output)
        scatter_figure(rows, summary, output, methods=('arithmetic', *joint.METHODS), filename='scatter-joint.svg')
        scatter_figure(rows, summary, output, methods=PRIMARY_METHODS, filename='scatter-primary.svg')
        scatter_figure(rows, summary, output, methods=('arithmetic', 'volatility_center_arithmetic_contrast',
                                                      'lichess_policy_expectation'), filename='scatter-focus.svg')
    return result


if __name__ == '__main__':
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--samples', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--plots-only', action='store_true', help='Render existing saved results without fitting again.')
    args = parser.parse_args()
    if args.plots_only:
        result = json.loads((args.output/'comparison.json').read_text(encoding='utf-8'))
        predictions = json.loads((args.output/'predictions.json').read_text(encoding='utf-8'))
        write_tables(result['players'], args.output)
        figures(predictions, result['summary'], args.output)
        scatter_figure(result['players'], result['summary'], args.output)
        if joint.METHODS[0] in result['summary']:
            scatter_figure(result['players'], result['summary'], args.output,
                           methods=('arithmetic', *joint.METHODS), filename='scatter-joint.svg')
        if all(method in result['summary'] for method in PRIMARY_METHODS):
            scatter_figure(result['players'], result['summary'], args.output,
                           methods=PRIMARY_METHODS, filename='scatter-primary.svg')
            scatter_figure(result['players'], result['summary'], args.output,
                           methods=('arithmetic', 'volatility_center_arithmetic_contrast',
                                    'lichess_policy_expectation'), filename='scatter-focus.svg')
    else:
        result = run(args.output, samples=args.samples, seed=args.seed, plots=not args.no_plots)
    print(json.dumps(result['summary'], indent=2))
