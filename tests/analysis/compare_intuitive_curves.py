"""Audit current PGNs, predict without labels, then score fixed research models.

No engine, coaching API, production cache migration, or parameter fitting runs.
Each module's mathematical declarations and source hash are saved before any
prediction, and numerical predictions are saved before labels are joined.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from analysis.player_rating.bayesian_shared_curve import fit_pair
from analysis.settings import CONFIG
from tests.analysis.compare_blitz_rating_methods import current_pgn_context, intersection
from tests.analysis.compare_joint_rating_directions import matches_pair
from tests.analysis.compare_rating_methods import load_cases

ROOT = Path(__file__).resolve().parents[2]
SIDES = ('White', 'Black')
DEFAULT_MODULES = (
    'curve_production_baselines', 'curve_residual_candidates',
    'curve_anchor_candidates', 'curve_discrepancy_candidates',
    'curve_population_candidates', 'curve_hierarchical_candidates',
    'curve_robust_candidates', 'curve_regularized_candidates',
    'curve_information_candidates', 'simple_accuracy_scale',
)
SENSITIVITY_OFFSETS = (-200., -150., -100., -50., 50., 100., 150., 200.)


def dump(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def metrics(rows):
    valid = [row for row in rows if row['estimate'] is not None]
    error = np.array([row['estimate']-row['reference'] for row in valid])
    pairs = {}
    for row in valid:
        pairs.setdefault(row['game'], {})[row['side']] = row
    checks = []
    for game, pair in pairs.items():
        if len(pair) != 2:
            continue
        w, b = (pair[s] for s in SIDES)
        gap = w['estimate']-b['estimate']
        checks.append({'game': game, 'gap': gap, 'reference_gap': w['reference']-b['reference'],
                       'accuracy_gap': w['accuracy']-b['accuracy'],
                       'curve_order_match': bool(np.sign(gap) == np.sign(w['accuracy']-b['accuracy'])),
                       'reference_order_match': bool(matches_pair(w['reference']-b['reference'], gap)),
                       'display_order_match': bool(matches_pair(w['reference']-b['reference'],
                                                               int(np.floor(w['estimate']+.5))-int(np.floor(b['estimate']+.5))))})
    edges = [row for row in valid if row['edge']]
    subset = lambda group: float(np.mean([abs(row['estimate']-row['reference']) for row in group])) if group else None
    observed_max = lambda name: max((row[name] for row in valid if row.get(name) is not None), default=None)
    return {'players': len(valid), 'mae': float(abs(error).mean()) if len(error) else None, 'maximum_error': float(abs(error).max()) if len(error) else None,
            'bias': float(error.mean()) if len(error) else None, 'median_signed_error': float(np.median(error)) if len(error) else None,
            'above_reference': int(sum(error > 0)), 'below_reference': int(sum(error < 0)), 'equal_reference': int(sum(error == 0)),
            'edge_mae': subset(edges), 'native_mae': subset([row for row in valid if not row['edge']]),
            'edge_maximum_error': max((abs(row['estimate']-row['reference']) for row in edges), default=None),
            'curve_order_matches': sum(row['curve_order_match'] for row in checks),
            'reference_order_matches': sum(row['reference_order_match'] for row in checks),
            'display_order_matches': sum(row['display_order_match'] for row in checks),
            'comparable_games': len(checks),
            'maximum_own_rating_change': observed_max('own_rating_change'),
            'maximum_both_ratings_change': observed_max('both_ratings_change'),
            'pairs': checks}


def run(names, output, *, sensitivity=True):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'tests/analysis/output'):
        raise ValueError('Research output must remain under tests/analysis/output.')
    output.mkdir(parents=True, exist_ok=True)
    modules = [import_module('tests.analysis.'+name) for name in names]
    declarations = {module.__name__: {'methods': module.describe(),
                    'sha256': sha256(Path(module.__file__).read_bytes()).hexdigest()} for module in modules}
    dump(output/'design.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'declarations': declarations,
                             'rule': 'No commercial labels or benchmark-fitted parameters enter candidate prediction.'})
    numbers = sorted(int(match[1]) for path in (ROOT/'games').glob('game*.pgn')
                     if (match := re.fullmatch(r'game(\d+)\.pgn', path.name)))
    cases = load_cases(ROOT/'games', numbers, Path(CONFIG['ANALYSIS']['CACHE_DIR']))
    protected = {path: value for case in cases for path, value in case['inputs'].items()}
    for path in [ROOT/'config.yaml', *sorted((ROOT/'analysis').rglob('*.py')),
                 *sorted((ROOT/'tests/analysis').glob('*.py')),
                 ROOT/'analysis/player_rating/data/maia_accuracy_calibration.json']:
        protected[path] = sha256(path.read_bytes()).hexdigest()
    predicted, contexts = [], []
    started = perf_counter()
    for case in cases:
        evidence = current_pgn_context(case)
        actual = {side: evidence[side]['actual_rating'] for side in SIDES}
        curve = fit_pair(evidence['White'], evidence['Black'])
        accuracies = {side: curve['players'][i]['average_accuracy'] for i, side in enumerate(SIDES)}
        crossing = {side: intersection(curve['monotone_expected_accuracy'], accuracies[side]) for side in SIDES}
        contexts.append({'game': case['name'], 'actual': actual, 'accuracy': accuracies,
                         'knots': curve['monotone_expected_accuracy'],
                         'variance': curve['likelihood']['accuracy_variance'], 'intersection': crossing})
        base = {'shared_curve_intersection': {side: crossing[side]['estimate'] for side in SIDES},
                'bayesian_shared_curve': {side: curve['players'][i]['unrounded_estimate'] for i, side in enumerate(SIDES)}}
        variations = {}
        for module in modules:
            if hasattr(module, 'prepare') and hasattr(module, 'predict_prepared'):
                prepared = module.prepare(evidence, actual)
                predict = lambda ratings: module.predict_prepared(prepared, ratings)
            else:
                predict = lambda ratings: module.predict(evidence, ratings)
            result = predict(actual)
            if set(base).intersection(result):
                raise ValueError('Duplicate method names.')
            base.update(result)
            if sensitivity:
                for who in (*SIDES, 'both'):
                    for delta in SENSITIVITY_OFFSETS:
                        perturbed = {side: value+(delta if who in (side, 'both') else 0.) for side, value in actual.items()}
                        variant = predict(perturbed)
                        for method in result:
                            for side in SIDES:
                                if who not in (side, 'both'):
                                    continue
                                category = 'own_rating_change' if who == side else 'both_ratings_change'
                                key = (method, side, category)
                                if result[method][side] is not None and variant[method][side] is not None:
                                    variations[key] = max(variations.get(key, 0.), abs(variant[method][side]-result[method][side]))
        for method, pair in base.items():
            for side in SIDES:
                predicted.append({'game': case['name'], 'number': case['number'], 'side': side, 'method': method,
                                  'actual': actual[side], 'accuracy': accuracies[side], 'estimate': pair[side],
                                  'edge': crossing[side]['edge'], 'intersection_status': crossing[side]['intersection_status'],
                                  **{category: variations.get((method, side, category), 0. if sensitivity else None)
                                     for category in ('own_rating_change', 'both_ratings_change')}})
        print(f'{case["name"]}: predictions finished without reference labels.', flush=True)
    # Persist predictions before the first reference-header read in this runner.
    dump(output/'predictions.json', predicted)
    dump(output/'contexts.json', contexts)
    references = {case['name']: {side: int(case['game'].headers[side+'EloEstimate']) for side in SIDES} for case in cases}
    rows = [{**row, 'reference': references[row['game']][row['side']]} for row in predicted]
    methods = list(dict.fromkeys(row['method'] for row in rows))
    summary = {method: metrics([row for row in rows if row['method'] == method]) for method in methods}
    for path, digest in protected.items():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Protected input changed: {path}')
    for module in modules:
        if sha256(Path(module.__file__).read_bytes()).hexdigest() != declarations[module.__name__]['sha256']:
            raise RuntimeError(f'Candidate changed during evaluation: {module.__name__}')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'games': len(cases), 'design': declarations,
              'summary': summary, 'players': rows, 'runtime_seconds': perf_counter()-started,
              'evidence_audits': {case['name']: case['audit'] for case in cases},
              'input_sha256': {str(path): digest for path, digest in protected.items()},
              'protected_files_unchanged': len(protected), 'sensitivity_evaluated': sensitivity,
              'sensitivity_offsets': list(SENSITIVITY_OFFSETS) if sensitivity else [],
              'sensitivity_scope': 'Largest observed change on the declared perturbation grid, not a proof of a continuous-interval maximum.'}
    dump(output/'comparison.json', result)
    with (output/'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({method: {k: v for k, v in values.items() if k != 'pairs'} for method, values in summary.items()}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modules', nargs='+', default=list(DEFAULT_MODULES))
    parser.add_argument('--output', type=Path, default=ROOT/'tests/analysis/output/intuitive-curves')
    parser.add_argument('--skip-sensitivity', action='store_true')
    args = parser.parse_args()
    run(args.modules, args.output, sensitivity=not args.skip_sensitivity)
