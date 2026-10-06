"""Numerical control: a monotone decision map for the saved predictive mixture.

This keeps the declared equal-prior Gaussian/Beta likelihood and common 5%
account anchor. Only its whole accuracy-to-posterior-mean map is projected onto
nondecreasing functions, before either played accuracy is read. References are
not prediction inputs. The projected point is a decision, not a posterior mean.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import isotonic_regression

from analysis.player_rating.arithmetic_coverage import _account_anchor
from analysis.player_rating.calibration import load_calibration
from analysis.player_rating.uncertainty_likelihood import (
    ACCURACY_GRID_STEP, beta_accuracy_mass, gaussian_accuracy_mass, posterior_mean,
)
from analysis.player_rating.uncertainty_measurement import ARGS, measure


METHOD = 'monotone_predictive_mean_common_account'
SIDES = ('White', 'Black')


def describe():
    return {METHOD: 'Equal-prior arithmetic Gaussian/Beta predictive mean, projected to a nondecreasing accuracy map, with the existing 5% common account anchor.'}


@lru_cache(maxsize=32)
def mapping(target_mean, target_variance, population_mean, population_variance):
    accuracy = np.linspace(0., 100., int(round(100/ACCURACY_GRID_STEP))+1)
    likelihood = .5*gaussian_accuracy_mass(accuracy[:, None], target_mean, target_variance)
    likelihood += .5*beta_accuracy_mass(accuracy[:, None], population_mean, population_variance)
    raw, valid = posterior_mean(likelihood, ARGS.grid)
    if not np.all(valid):
        raise ValueError('The fixed predictive mixture has no finite posterior mass.')
    projected = isotonic_regression(raw, increasing=True).x
    if np.any(np.diff(projected) < -1e-9):
        raise AssertionError('The projected decision map must be nondecreasing.')
    for values in (accuracy, raw, projected):
        values.setflags(write=False)
    return {'accuracy': accuracy, 'raw': raw, 'projected': projected,
            'maximum_projection_change': float(np.max(abs(projected-raw))),
            'decreasing_steps': int(np.sum(np.diff(raw) < -1e-7))}


def predict(evidence, actual_ratings):
    measurement = measure(evidence)
    curve = measurement['curve']
    if not curve['identifiable']:
        return {METHOD: dict.fromkeys(SIDES)}
    corpus = load_calibration().for_evidence(evidence)
    population, population_variance = corpus.population(ARGS.grid, 'arithmetic')
    decision = mapping(tuple(curve['shared_accuracy']), float(curve['likelihood']['accuracy_variance']),
                       tuple(population), tuple(population_variance))
    anchor, _ = _account_anchor(actual_ratings)
    points = {}
    for side in SIDES:
        moment = measurement['sides'][side]
        if moment is None:
            points[side] = None
            continue
        point = float(np.interp(moment['accuracy'], decision['accuracy'], decision['projected']))
        points[side] = point if anchor is None else .95*point+.05*anchor
    return {METHOD: points}


def run():
    """Evaluate the fixed numerical control offline, preserving normal outputs."""
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from hashlib import sha256
    import json
    from pathlib import Path
    from time import perf_counter

    from analysis.settings import CONFIG
    from tests.analysis.compare_joint_rating_directions import score
    from tests.analysis.compare_rating_methods import load_cases

    root = Path(__file__).resolve().parents[2]
    cases = load_cases(root/'games', range(19), Path(CONFIG['ANALYSIS']['CACHE_DIR']))
    protected = {path for case in cases for path in case['inputs']}
    protected.update((root/'analysis').rglob('*.py'))
    protected.update((root/'games/output').rglob('*.svg'))
    protected.update((root/'games/output').rglob('fit.json'))
    protected.update((root/'config.yaml', root/'analysis/player_rating/data/maia_accuracy_calibration.json'))
    before = {str(path): sha256(path.read_bytes()).hexdigest() for path in protected}
    predictions, checks, started = [], [], perf_counter()
    for case in cases:
        evidence = case['evidence']
        actual = {side: evidence[side].get('actual_rating') for side in SIDES}
        points = predict(evidence, actual)[METHOD]
        for side in SIDES:
            altered = {**actual, side: actual[side]+200}
            adjusted = predict(evidence, altered)[METHOD]
            assert all(abs(adjusted[key]-points[key]-5) < 1e-7 for key in SIDES)
        # Equal accuracies must remain equal; swapping all sides must only swap points.
        swapped = {**evidence, 'White': evidence['Black'], 'Black': evidence['White']}
        reversed_points = predict(swapped, dict(zip(SIDES, reversed(list(actual.values())))))[METHOD]
        assert abs(reversed_points['White']-points['Black']) < 1e-7
        assert abs(reversed_points['Black']-points['White']) < 1e-7
        predictions.append((case, actual, points))
        checks.append({'game': case['name'], 'swapping_players': True,
                       'one_account_plus200_shifts_both_by5': True})
        print(case['name'], {side: round(value, 2) for side, value in points.items()}, flush=True)
    # Reference labels are accessed only after all predictions and invariants.
    rows = [{'game': case['name'], 'side': side, 'method': METHOD,
             'unrounded_estimate': points[side], 'actual': actual[side],
             'reference': int(case['game'].headers[side+'EloEstimate'])}
            for case, actual, points in predictions for side in SIDES]
    after = {path: sha256(Path(path).read_bytes()).hexdigest() for path in before}
    if before != after:
        raise RuntimeError('A protected production or game input changed.')
    result = {'method': METHOD, 'description': describe()[METHOD], 'players': rows,
              'rounded': score(rows, rounded=True), 'unrounded': score(rows, rounded=False),
              'invariants': checks, 'elapsed_seconds': perf_counter()-started,
              'protected_inputs_unchanged': len(before), 'input_sha256': before,
              'parameters_fit_to_references': False}
    output = root/'tests/analysis/output/joint-rating-directions/monotone-control.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['rounded'], indent=2))
    return result


if __name__ == '__main__':
    run()
