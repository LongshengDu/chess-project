"""Compare universal rating methods without exposing references to inference.

Every candidate uses one formula for all players. The original shared-curve
ordering is a separately recorded acceptance requirement, not a fitted target.
The five original curve-edge players form a fixed evaluation subset only.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from analysis.cache import write_json
from tests.analysis.experiment_edge_rating import ROOT, load_cases
from tests.analysis.experiment_shared_curve_lichess import _digest, _output_path
from tests.analysis.experiment_shared_curve_sweep import _write_csv
from tests.analysis.edge_quality_likelihood import is_edge
from tests.analysis.universal_competitiveness import annotate_probabilities


FAMILIES = ('posterior_decision', 'pair_quality', 'curve_measurement',
            'predictive_distribution', 'curve_uncertainty', 'accuracy_likelihood',
            'accuracy_cdf', 'move_quality', 'noise_sensitivity',
            'bootstrap_calibration', 'hurdle_quality', 'competitiveness',
            'quality_consensus', 'quantile_transport', 'native_coverage',
            'competitive_sensitivity', 'joint_accuracy', 'coverage_consensus',
            'competitive_marginal', 'empirical_prior', 'jackknife_consensus',
            'joint_copula', 'consensus_sensitivity', 'account_sensitivity',
            'pair_contrast', 'variance_uncertainty', 'outcome_quality',
            'side_uncertainty', 'uncertainty_consensus', 'robust_consensus',
            'context_similarity')
OUTPUT = ROOT/'tests/analysis/output/universal-rating-methods'


def prepare_cases(games_dir):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    cases, sources = load_cases(games_dir)
    for case in cases:
        path = Path(games_dir)/'output'/f'{case["game"]}-full'/'analysis.json'
        moves = json.loads(path.read_text(encoding='utf-8'))['moves']
        data = case['input']
        data['evidence'] = annotate_probabilities(data['evidence'], moves)
    sources.extend(Path(__file__).parent.glob('universal_*.py'))
    sources.extend(Path(__file__).parent.glob('experiment_*.py'))
    sources.append(Path(__file__).with_name('compare_shared_curve_games.py'))
    sources.append(Path(__file__))
    return cases, sources


def score(rows):
    ranking = []
    for method in sorted({row['method'] for row in rows}):
        selected = [row for row in rows if row['method'] == method]
        errors = np.asarray([row['estimate']-row['reference'] for row in selected])
        rounded_errors = np.asarray([np.floor(row['estimate']+.5)-row['reference'] for row in selected])
        edge_errors = [abs(row['estimate']-row['reference']) for row in selected if row['edge']]
        rounded_edge_errors = [abs(np.floor(row['estimate']+.5)-row['reference'])
                               for row in selected if row['edge']]
        ordering = sum(np.sign(selected[i]['estimate']-selected[i+1]['estimate']) ==
                       np.sign(selected[i]['original_estimate']-selected[i+1]['original_estimate'])
                       for i in range(0, len(selected), 2))
        entry = {'method': method, 'mae': float(np.abs(errors).mean()),
                 'edge_mae': float(np.mean(edge_errors)), 'maximum_error': float(np.abs(errors).max()),
                 'rounded_mae': float(np.abs(rounded_errors).mean()),
                 'rounded_edge_mae': float(np.mean(rounded_edge_errors)),
                 'rounded_maximum_error': float(np.abs(rounded_errors).max()),
                 'ordering_matches': int(ordering), 'games': len(selected)//2}
        entry['meets_required_targets'] = bool(entry['mae'] < 100 and entry['edge_mae'] < 100
                                               and entry['ordering_matches'] == entry['games'])
        entry['meets_maximum_aim'] = bool(entry['maximum_error'] < 200)
        entry['meets_rounded_error_targets'] = bool(entry['rounded_mae'] < 100
                                                    and entry['rounded_edge_mae'] < 100)
        ranking.append(entry)
    return sorted(ranking, key=lambda row: (not row['meets_required_targets'], row['mae']))


def run(games_dir=ROOT/'games', output=OUTPUT, families=FAMILIES):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    started = perf_counter()
    output = _output_path(output); output.mkdir(parents=True, exist_ok=True)
    cases, sources = prepare_cases(games_dir)
    modules = [importlib.import_module('tests.analysis.universal_'+name) for name in families]
    before = _digest(sources)
    predictions = []
    for case in cases:
        data = case['input']
        calibration = [other['input'] for other in cases if other is not case]
        values = {'current': {side: data['fit']['players'][side]['estimate'] for side in ('White', 'Black')}}
        for module in modules:
            returned = module.predict(**data, calibration_cases=calibration)
            overlap = values.keys() & returned.keys()
            if overlap:
                raise ValueError(f'Duplicate candidate names: {overlap}')
            values.update(returned)
        predictions.append(values)
        print(f'{case["game"]}: {len(values)} universal variants', flush=True)
    # Reference scores are consulted only after every inference has finished.
    rows = []
    for case, values in zip(cases, predictions, strict=True):
        for name, pair in values.items():
            for side in ('White', 'Black'):
                estimate = float(pair[side])
                if not np.isfinite(estimate) or not 0 <= estimate <= 3200:
                    raise ValueError(f'{name} produced an invalid estimate.')
                rows.append({'game': case['game'], 'side': side, 'method': name, 'estimate': estimate,
                             'reference': case['references'][side], 'actual': case['input']['ratings'][side],
                             'original_estimate': case['input']['fit']['players'][side]['estimate'],
                             'edge': bool(is_edge(case['input']['fit'], side))})
    after = _digest(sources)
    if before != after:
        raise AssertionError('Protected source/evidence changed during evaluation.')
    ranking = score(rows)
    result = {'ranking': ranking, 'players': rows, 'families': list(families),
              'elapsed_seconds': perf_counter()-started,
              'scope': {'games': len(cases), 'players': 2*len(cases),
                        'calibration': 'Target game excluded; no commercial reference in inference.',
                        'production_unchanged': True, 'protected_files': len(before)}}
    write_json(output/'comparison.json', result)
    write_json(output/'source-hashes.json', {'before': before, 'after': after})
    _write_csv(output/'ranking.csv', ranking)
    _write_csv(output/'players.csv', rows)
    print(json.dumps(ranking, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games-dir', type=Path, default=ROOT/'games')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--families', nargs='+', default=FAMILIES)
    options = parser.parse_args()
    run(options.games_dir, options.output_dir, options.families)
