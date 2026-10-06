"""Compare predeclared simple rating rules without changing production outputs."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from analysis.cache import write_json
from analysis.settings import CONFIG
from tests.analysis.compare_rating_methods import load_cases

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/simple-rating-restart'
FAMILIES = ('curve_account', 'accuracy_scale', 'pair_resolution', 'pair_pooling', 'arithmetic_likelihood', 'bounded_accuracy')
SIDES = ('White', 'Black')
EDGES = {('game1', 'White'), ('game4', 'Black'), ('game12', 'Black'), ('game15', 'White'), ('game15', 'Black')}
TIE_ELO = 50.


def rounded(value):
    return int(np.floor(value+.5))


def matches(reference, estimate):
    """Exact reference ties allow <50 fitted Elo; any other reference retains sign."""
    return abs(estimate) < TIE_ELO if reference == 0 else np.sign(estimate) == np.sign(reference)


def classify(gap):
    return 0 if abs(gap) < TIE_ELO else int(np.sign(gap))


def score(rows):
    ranking = []
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        errors = np.array([abs(r['estimate']-r['reference']) for r in selected])
        pairs = {}
        for row in selected:
            pairs.setdefault(row['game'], {})[row['side']] = row
        checks, strict, failures = [], [], []
        for game, pair in pairs.items():
            reference = pair['White']['reference']-pair['Black']['reference']
            estimate = pair['White']['estimate']-pair['Black']['estimate']
            ok = bool(matches(reference, estimate))
            checks.append(ok)
            strict.append(bool(np.sign(reference) == classify(estimate)))
            if not ok:
                failures.append({'game': game, 'reference_gap': reference, 'estimated_gap': estimate})
        ranking.append({'method': method, 'mae': float(errors.mean()), 'maximum_error': float(errors.max()),
                        'edge_mae': float(np.mean([abs(r['estimate']-r['reference']) for r in selected if r['edge']])),
                        'new_mae': float(np.mean([abs(r['estimate']-r['reference']) for r in selected if r['number'] >= 16])),
                        'ordering_matches': sum(checks), 'games': len(checks), 'strict_three_way_matches': sum(strict),
                        'failures': failures, 'actual_sensitivity': max(r['own_rating_sensitivity'] for r in selected)})
    return sorted(ranking, key=lambda r: (-r['ordering_matches'], r['mae']))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(path, result):
    selected = ('production_ensemble', 'production_bayesian', 'simple_posterior_mean_account10',
                'arithmetic_predictive_common_account', 'arithmetic_coverage_common_account',
                'simple_target_beta_common_account10', 'pair_pool_two_sigma')
    labels = ('Current ensemble', 'Original Bayesian', 'Simple Gaussian',
              'Arithmetic-only mixture', 'Arithmetic coverage', 'Simple bounded Beta', 'Pair uncertainty control')
    available = {r['method'] for r in result['ranking']}
    choices = [(name, label) for name, label in zip(selected, labels, strict=True) if name in available]
    lines = ['# Simple rating-method comparison', '', result['scope'], '',
             result['ordering_rule'], '',
             '| Method | All-player MAE | Five-edge MAE | Maximum error | Direction/tie matches | Strict three-way matches |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, label in choices:
        r = next(r for r in result['ranking'] if r['method'] == name)
        lines.append(f"| {label} | {r['mae']:.2f} | {r['edge_mae']:.2f} | {r['maximum_error']:.0f} | {r['ordering_matches']}/19 | {r['strict_three_way_matches']}/19 |")
    lines.extend(['', 'The ordering check alone is insufficient: a rule can compress genuinely different players into almost equal numbers while retaining their mathematical sign. Passing these pairs also does not guarantee monotonicity elsewhere. The arithmetic likelihood mixture can decrease as accuracy increases; the arithmetic coverage method explicitly constructs a nondecreasing mapping. See arithmetic-predictive-monotonicity.json for the separate frozen-context audit.', '',
                  '| Game | Commercial W / B | '+' | '.join(label+' W / B' for _, label in choices)+' |',
                  '| --- | --- | '+' | '.join('---' for _ in choices)+' |'])
    for number in range(19):
        rows = [r for r in result['players'] if r['number'] == number]
        reference = {r['side']: r['reference'] for r in rows}
        values = []
        for name, _ in choices:
            pair = {r['side']: r['estimate'] for r in rows if r['method'] == name}
            values.append(f"{pair['White']} / {pair['Black']}")
        lines.append(f"| game{number} | {reference['White']} / {reference['Black']} | "+' | '.join(values)+' |')
    lines.extend(['', f"Protected production/evidence files unchanged: {result['protected_files']}. All experiment predictions used saved evidence; no engine analysis or coaching calls were run.", ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(families=FAMILIES, output=OUTPUT):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'tests/analysis/output'):
        raise ValueError('Experiment output must remain under tests/analysis/output.')
    cases = load_cases(ROOT/'games', range(19), Path(CONFIG['ANALYSIS']['CACHE_DIR']))
    archived = json.loads((ROOT/'tests/analysis/output/rating-methods-games0-18/comparison.json').read_text(encoding='utf-8'))
    modules = [import_module('tests.analysis.simple_'+name) for name in families]
    paths = {p for case in cases for p in case['inputs']}
    paths.update((ROOT/'analysis').rglob('*.py'))
    paths.add(ROOT/'config.yaml')
    paths.add(ROOT/'analysis/player_rating/data/maia_accuracy_calibration.json')
    paths.update((ROOT/'games/output').rglob('*.svg'))
    paths.update((ROOT/'games/output').rglob('fit.json'))
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}
    source_hashes = {m.__name__: sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules}
    descriptions = {name: description for module in modules for name, description in module.describe().items()}
    raw, started = [], perf_counter()
    for case in cases:
        evidence = case['evidence']
        actual = {side: evidence[side].get('actual_rating') for side in SIDES}
        predictions, sensitivities = {}, {}
        for module in modules:
            result = module.predict(evidence, actual)
            if set(result) & set(predictions):
                raise ValueError('Duplicate experiment method names.')
            predictions.update(result)
            sensitivities.update({name: dict.fromkeys(SIDES, 0.) for name in result})
            for side in SIDES:
                for delta in (-200, 200):
                    altered = dict(actual, **{side: actual[side]+delta})
                    changed = module.predict(evidence, altered)
                    for name in result:
                        sensitivities[name][side] = max(sensitivities[name][side], abs(changed[name][side]-result[name][side]))
        # Existing frozen results are controls, never inputs to a new predictor.
        old = json.loads((ROOT/'tests/analysis/output/rating-methods-games0-18'/case['name']/'bayesian_shared_curve/fit.json').read_text(encoding='utf-8'))
        predictions['production_ensemble'] = {row['side']: row['unrounded_estimate'] for row in archived['players']
                                             if row['game'] == case['name'] and row['method_id'] == 'uncertainty_ensemble'}
        predictions['production_bayesian'] = {side: old['players'][side]['unrounded_estimate'] for side in SIDES}
        sensitivities['production_ensemble'] = dict.fromkeys(SIDES, 20.)
        sensitivities['production_bayesian'] = dict.fromkeys(SIDES, 0.)
        raw.append((case, predictions, sensitivities))
        print(f'{case["name"]}: {len(predictions)} fixed methods evaluated.', flush=True)
    # Labels become scoring data only after all predictions are complete.
    rows = []
    for case, predictions, sensitivities in raw:
        for name, pair in predictions.items():
            for side in SIDES:
                point = float(pair[side])
                if not np.isfinite(point):
                    raise ValueError('A method produced a nonfinite point.')
                rows.append({'game': case['name'], 'number': case['number'], 'side': side, 'method': name,
                             'estimate': rounded(point), 'unrounded_estimate': point,
                             'reference': int(case['game'].headers[side+'EloEstimate']),
                             'actual': case['evidence'][side]['actual_rating'],
                             'edge': (case['name'], side) in EDGES,
                             'own_rating_sensitivity': sensitivities[name][side]})
    after = {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}
    if before != after:
        raise RuntimeError('Production source or game evidence changed during the experiment.')
    ranking = score(rows)
    result = {'ranking': ranking, 'players': rows, 'descriptions': descriptions, 'source_hashes': source_hashes,
              'seconds': perf_counter()-started, 'protected_files': len(before),
              'scope': 'Exploratory comparison of fixed formulas; no parameter fitting to commercial labels. Choosing a method after these comparisons is not independent validation. No engines or coaching models run.',
              'ordering_rule': 'Only exactly equal commercial ratings count as a reference tie; our fitted difference must then be <50 Elo. Every nonzero commercial difference, including game8, requires the same fitted sign. The additional strict_three_way diagnostic treats our gaps <50 as ties and is not an acceptance requirement.',
              'sensitivity_scope': 'Own supplied rating -200/+200 with all chess measurements fixed. Production ensemble entry is its proven <=20 bound; other new methods are measured.'}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'comparison.json', result)
    write_json(output/'source-hashes.json', {'before': before, 'after': after})
    write_csv(output/'players.csv', rows)
    write_csv(output/'ranking.csv', [{k:v for k,v in row.items() if k != 'failures'} for row in ranking])
    write_comparison(output/'comparison.md', result)
    print(json.dumps(ranking, indent=2))
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--families', nargs='+', default=FAMILIES, choices=FAMILIES)
    cli.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = cli.parse_args()
    run(args.families, args.output_dir)
