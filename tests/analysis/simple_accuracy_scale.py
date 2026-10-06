"""Eight fixed, simple arithmetic-accuracy scales; research only.

Declared before evaluation: invert the local, frozen-population, or equally
pooled shared curve; use identity/log-loss/log-odds accuracy with a model-only
population secant slope; and test local/population log-odds contrasts using the
conventional Elo logistic coefficient 400/log(10). The latter is a diagnostic
analogy, not a claim that move accuracy equals expected game score.

All methods use complete Maia policies and ordinary arithmetic played accuracy.
There are no likelihood mixtures, competitive-position weights or fitted label
coefficients. A fixed 5% contribution from the two players' mean account rating
is a common contextual anchor, preserving the shared accuracy ordering. Changing
one supplied rating by 200 changes each output by at most 5 Elo; changing both
by 200 changes each by at most 10 Elo. Account ratings do not affect the curves.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analysis.player_rating.bayesian_shared_curve import SharedCurve, side_moments
from analysis.player_rating.calibration import load_calibration

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/simple-rating-restart/global'
SIDES = ('White', 'Black')
RATING_GRID = np.arange(200., 3001., 5.)
REFERENCE_RATING = 1600.
ACCOUNT_WEIGHT = .05
ACCURACY_ENDPOINT_FLOOR = .005  # Half the existing 0.01-point observation bin.
ELO_LOG_ODDS_SCALE = 400./np.log(10.)
METHODS = (
    'simple_global_local_inverse',
    'simple_global_population_inverse',
    'simple_global_equal_curve_inverse',
    'simple_global_arithmetic_secant',
    'simple_global_logloss_secant',
    'simple_global_logodds_secant',
    'simple_global_local_elo_odds',
    'simple_global_population_elo_odds',
)
EDGE_PLAYERS = {('game1', 'White'), ('game4', 'Black'), ('game15', 'White'),
                ('game12', 'Black'), ('game15', 'Black')}


def describe():
    """Describe the fixed experiment variants for the common comparison harness."""
    explanations = (
        'Invert the game-specific arithmetic shared curve.',
        'Invert the reference-free population arithmetic curve.',
        'Invert the equally pooled game-specific and population arithmetic curves.',
        'Center arithmetic accuracy on the local curve at 1600; use the population 600–2600 secant slope.',
        'Center negative log accuracy-loss on the local curve at 1600; use its population 600–2600 secant slope.',
        'Center accuracy log-odds on the local curve at 1600; use its population 600–2600 secant slope.',
        'Map local-centered accuracy log-odds with 400/log(10); diagnostic Elo analogy, not a calibrated accuracy law.',
        'Map population-centered accuracy log-odds with 400/log(10); diagnostic Elo analogy, not a calibrated accuracy law.',
    )
    suffix = ' Add a 5% common mean-account-rating contribution; preserve the shared arithmetic order.'
    return {method: explanation+suffix for method, explanation in zip(METHODS, explanations, strict=True)}


def _transform(values, name):
    values = np.asarray(values, dtype=float)
    if name == 'arithmetic':
        return values
    clipped = np.clip(values, ACCURACY_ENDPOINT_FLOOR, 100.-ACCURACY_ENDPOINT_FLOOR)
    if name == 'logloss':
        return -np.log(100.-clipped)
    if name == 'logodds':
        return np.log(clipped/(100.-clipped))
    raise ValueError('Unknown arithmetic-accuracy transform.')


def _inverse(accuracy, curve):
    """Invert a bounded monotone curve; a flat level maps to its rating midpoint."""
    distinct, starts, counts = np.unique(curve, return_index=True, return_counts=True)
    locations = np.array([RATING_GRID[start:start+count].mean() for start, count in zip(starts, counts)])
    return np.interp(accuracy, distinct, locations)


def raw_scales(accuracy, local_curve, population_curve):
    """Apply all declared maps to an accuracy scalar or array on a fixed context."""
    accuracy = np.asarray(accuracy, dtype=float)
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 100)):
        raise ValueError('Observed arithmetic accuracy must lie in [0, 100].')
    local = np.asarray(local_curve, dtype=float)
    population = np.asarray(population_curve, dtype=float)
    for curve in (local, population):
        if (curve.shape != RATING_GRID.shape or not np.isfinite(curve).all()
                or np.any(np.diff(curve) < -1e-9) or np.any((curve < 0) | (curve > 100))):
            raise ValueError('A bounded monotone shared accuracy curve is required.')
    results = {
        METHODS[0]: _inverse(accuracy, local),
        METHODS[1]: _inverse(accuracy, population),
        METHODS[2]: _inverse(accuracy, (local+population)/2),
    }
    local_anchor = float(np.interp(REFERENCE_RATING, RATING_GRID, local))
    population_anchor = float(np.interp(REFERENCE_RATING, RATING_GRID, population))
    population_endpoints = np.interp([600., 2600.], RATING_GRID, population)
    for name, method in zip(('arithmetic', 'logloss', 'logodds'), METHODS[3:6], strict=True):
        endpoints = _transform(population_endpoints, name)
        slope = (endpoints[1]-endpoints[0])/2000.
        if slope <= 1e-12:
            raise ValueError('Population accuracy does not identify a positive rating slope.')
        results[method] = REFERENCE_RATING+(_transform(accuracy, name)-_transform(local_anchor, name))/slope
    for method, anchor in zip(METHODS[6:], (local_anchor, population_anchor), strict=True):
        results[method] = REFERENCE_RATING+ELO_LOG_ODDS_SCALE*(_transform(accuracy, 'logodds')-_transform(anchor, 'logodds'))
    return {name: np.clip(value, RATING_GRID[0], RATING_GRID[-1]) for name, value in results.items()}


def predict(evidence, actual_ratings, *, calibration=None):
    """Pure label-free API returning eight method mappings of White/Black points."""
    from scipy.optimize import isotonic_regression
    moments = {side: side_moments(evidence[side]) for side in SIDES}
    if any(moment is None for moment in moments.values()):
        raise ValueError('Both players need a non-forced decision in this experiment.')
    expected = (moments['White']['mean']+moments['Black']['mean'])/2
    local = SharedCurve(np.clip(isotonic_regression(expected).x, 0., 100.))(RATING_GRID)
    corpus = (calibration or load_calibration()).for_evidence(evidence)
    population, _ = corpus.population(RATING_GRID, 'arithmetic')
    ratings = np.asarray([actual_ratings[side] for side in SIDES], dtype=float)
    if not np.isfinite(ratings).all():
        raise ValueError('Finite actual ratings are required by the common account anchor.')
    account_anchor = float(np.clip(ratings, 0., 3200.).mean())
    accuracy = np.array([moments[side]['accuracy'] for side in SIDES])
    outputs = raw_scales(accuracy, local, population)
    return {method: dict(zip(SIDES, map(float, np.clip((1-ACCOUNT_WEIGHT)*values+ACCOUNT_WEIGHT*account_anchor,
                                                     RATING_GRID[0], RATING_GRID[-1])), strict=True))
            for method, values in outputs.items()}


def _metrics(rows, pairs):
    results = []
    for method in METHODS:
        selected = [row for row in rows if row['method'] == method]
        errors = np.array([row['estimate']-row['reference'] for row in selected])
        rounded = np.array([int(np.floor(row['estimate']+.5))-row['reference'] for row in selected])
        paired = [pair for pair in pairs if pair['method'] == method]
        subset = lambda predicate: float(np.mean([abs(row['estimate']-row['reference']) for row in selected if predicate(row)]))
        results.append({'method': method, 'mae': float(np.mean(abs(errors))), 'rounded_mae': float(np.mean(abs(rounded))),
                        'edge_mae': subset(lambda row: row['edge']), 'new_mae': subset(lambda row: row['new']),
                        'maximum_error': float(np.max(abs(errors))), 'rounded_maximum_error': float(np.max(abs(rounded))),
                        'primary_order_matches': sum(pair['primary_match'] for pair in paired), 'games': len(paired),
                        'strict_three_way_matches': sum(pair['strict_three_way_match'] for pair in paired),
                        'decisive_sign_matches': sum(pair['sign_match'] for pair in paired if not pair['reference_tie']),
                        'decisive_games': sum(not pair['reference_tie'] for pair in paired),
                        'own_rating_max_change': max(row['own_rating_max_change'] for row in selected),
                        'opponent_rating_max_change': max(row['opponent_rating_max_change'] for row in selected),
                        'both_ratings_max_change': max(row['both_ratings_max_change'] for row in selected),
                        'reference_ties': [pair for pair in paired if pair['reference_tie']]})
    return sorted(results, key=lambda row: (row['mae'], row['method']))


def _protected(cases):
    paths = {ROOT/'config.yaml', ROOT/'analysis/player_rating/data/maia_accuracy_calibration.json'}
    for component in ('analysis', 'backend', 'coach', 'engine'):
        paths.update((ROOT/component).rglob('*.py'))
    for case in cases:
        paths.update(case['inputs'])
    for suffix in ('*.svg', '*.png'):
        paths.update((ROOT/'games/output').rglob(suffix))
    return {path: sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from analysis.settings import CONFIG
    from tests.analysis.compare_rating_methods import load_cases
    cases = load_cases(ROOT/'games', range(19), CONFIG['ANALYSIS']['CACHE_DIR'])
    protected = _protected(cases)
    calibration = load_calibration()
    predictions, sensitivity = [], []
    for case in cases:
        evidence = case['evidence']
        actual = {side: evidence[side]['actual_rating'] for side in SIDES}
        base = predict(evidence, actual, calibration=calibration)
        predictions.append(base)
        variants = {}
        for changed in ('White', 'Black', 'both'):
            variants[changed] = []
            for delta in (-200., 200.):
                shifted = {side: value+(delta if side == changed or changed == 'both' else 0.) for side, value in actual.items()}
                variants[changed].append(predict(evidence, shifted, calibration=calibration))
        sensitivity.append(variants)
    # Commercial labels are first consulted after all inference and perturbations.
    rows, pairs = [], []
    for case, prediction, variants in zip(cases, predictions, sensitivity, strict=True):
        references = {side: int(case['game'].headers[side+'EloEstimate']) for side in SIDES}
        for method, pair in prediction.items():
            reference_gap = references['White']-references['Black']
            gap = pair['White']-pair['Black']
            ref_tie, pred_tie = reference_gap == 0., abs(gap) < 50.
            sign_match = bool(np.sign(reference_gap) == np.sign(gap))
            pairs.append({'method': method, 'game': case['name'], 'white': pair['White'], 'black': pair['Black'],
                          'reference_gap': reference_gap, 'predicted_gap': gap, 'reference_tie': ref_tie,
                          'predicted_tie': pred_tie, 'sign_match': sign_match,
                          'primary_match': bool(pred_tie if ref_tie else sign_match),
                          'strict_three_way_match': bool(pred_tie if ref_tie else (not pred_tie and sign_match))})
            for side, estimate in pair.items():
                other = 'Black' if side == 'White' else 'White'
                change = lambda which: max(abs(result[method][side]-estimate) for result in variants[which])
                rows.append({'method': method, 'game': case['name'], 'side': side, 'estimate': estimate,
                             'reference': references[side], 'edge': (case['name'], side) in EDGE_PLAYERS,
                             'new': case['number'] >= 16, 'actual_rating': case['evidence'][side]['actual_rating'],
                             'own_rating_max_change': change(side), 'opponent_rating_max_change': change(other),
                             'both_ratings_max_change': change('both')})
    changed = [str(path) for path, digest in protected.items() if sha256(path.read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError('Protected inputs changed: '+', '.join(changed))
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'design': __doc__, 'methods': list(METHODS),
              'calibration_hash': calibration.content_hash, 'ranking': _metrics(rows, pairs), 'players': rows, 'pairs': pairs,
              'protected_files': len(protected), 'protected_files_changed': 0,
              'parameters': {'account_weight': ACCOUNT_WEIGHT, 'account_context': 'mean_of_both_actual_ratings',
                             'reference_rating': REFERENCE_RATING, 'rating_range': [200., 3000.],
                             'accuracy_endpoint_floor': ACCURACY_ENDPOINT_FLOOR,
                             'elo_log_odds_scale': ELO_LOG_ODDS_SCALE,
                             'measured_slope_interval': [600., 2600.], 'reference_tie_gap': 0.,
                             'prediction_tie_tolerance': 50., 'prediction_tie_tolerance_inclusive': False}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['ranking'], indent=2))


if __name__ == '__main__':
    main()
