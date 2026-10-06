"""Universal reference-free fits based on informative move qualities.

Fixed experimental assumptions, chosen before reference scoring:
1. Accuracy observations receive reliability weights signal/(signal+noise),
   where signal is between-rating expected-quality variance and noise is Maia's
   average conditional move-quality variance. Both colors share the curve.
2. Quality midranks have model expectation 1/2 regardless of quality spacing.
   Their centered sum supplies a standardized estimating-equation likelihood.
3. A move is good at accuracy >=95. Bernoulli Jensen--Shannon information across
   Maia ratings weights this categorical likelihood; 5% contamination models
   errors in Maia's probabilities. Weights never depend on what was played.

Each model caps effective independent moves at20 and has a separately reported
fixed5% account blend. All players use the same formula; there is no edge gate.
Calibration cases are unused: quality expectations come from the actual positions.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import isotonic_regression
from scipy.special import xlogy

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID, SharedCurve, curve_posterior, prior_weights


MAX_EFFECTIVE_MOVES = 20.
GOOD_ACCURACY = 95.
CONTAMINATION = .05
ACCOUNT_WEIGHT = .05
MEASURED_GRID = np.arange(GRID[0], GRID[-1] + 1., 5.)
BASE_METHODS = ('uq_information_curve_all', 'uq_midrank_all', 'uq_information_bernoulli_all')


def _record(record):
    """Accumulate reference-free quality statistics from every informative position."""
    means, variances, observed, weights = [], [], [], []
    rank_residual = np.zeros_like(GRID)
    rank_variance = np.zeros_like(GRID)
    rank_moves = 0
    binary_logs, binary_information = [], []
    for row in record['observations']:
        q = np.asarray(row['qualities']['position'], dtype=float)
        p = np.asarray(row['maia_probabilities'], dtype=float)
        index = row['played_index']
        if (q.ndim != 1 or not len(q) or p.shape != (len(GRID), len(q))
                or not np.isfinite(q).all() or np.any((q < 0) | (q > 100))
                or not np.isfinite(p).all() or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1.) or not 0 <= index < len(q)):
            raise ValueError('Complete Maia policies and bounded legal-move qualities are required.')
        if len(q) == 1 or np.ptp(q) < 1e-10:
            continue
        mean = p @ q
        variance = np.maximum(0., p @ (q*q) - mean*mean)
        signal = float(np.var(mean))
        reliability = signal / max(signal + float(np.mean(variance)), 1e-12)
        means.append(mean)
        variances.append(variance)
        observed.append(q[index])
        weights.append(reliability)

        _, inverse = np.unique(q, return_inverse=True)
        mass = np.stack([p @ (inverse == group) for group in range(inverse.max()+1)], axis=1)
        midrank = np.cumsum(mass, axis=1) - .5*mass
        rank_residual += midrank[:, inverse[index]] - .5
        rank_variance += np.sum(mass*(midrank-.5)**2, axis=1)
        rank_moves += 1

        good = p @ (q >= GOOD_ACCURACY)
        entropy = lambda probability: -(xlogy(probability, probability)
                                         + xlogy(1-probability, 1-probability))
        information = max(0., float(entropy(good.mean()) - entropy(good).mean()))
        contaminated = (1-CONTAMINATION)*good + CONTAMINATION*.5
        binary_logs.append(np.log(contaminated if q[index] >= GOOD_ACCURACY else 1-contaminated))
        binary_information.append(information)

    if not means:
        return None
    weights = np.asarray(weights)
    if weights.sum() > 1e-12:
        normalized = weights/weights.sum()
        effective = 1./np.sum(normalized**2)
        average_mean = normalized @ np.asarray(means)
        average_variance = normalized**2 @ np.asarray(variances)
        average_variance *= max(1., effective/MAX_EFFECTIVE_MOVES)
        actual = float(normalized @ observed)
    else:
        average_mean = np.mean(means, axis=0)
        average_variance = np.mean(variances, axis=0)/min(len(means), MAX_EFFECTIVE_MOVES)
        actual = float(np.mean(observed))
    rank_log = -.5*rank_residual**2/np.maximum(rank_variance, 1e-12)
    rank_log *= min(1., MAX_EFFECTIVE_MOVES/rank_moves)
    information = np.asarray(binary_information)
    if information.max() > 1e-12:
        information /= information.max()
        effective = information.sum()**2/np.sum(information**2)
        binary_log = information @ np.asarray(binary_logs)
        binary_log *= min(1., MAX_EFFECTIVE_MOVES/effective)
    else:
        binary_log = np.zeros_like(GRID)
    return {'mean': average_mean, 'variance': average_variance, 'observed': actual,
            'rank_logs': rank_log, 'binary_logs': binary_log}


def _measured_median(logs):
    fine_logs = np.interp(MEASURED_GRID, GRID, logs)
    density = np.exp(fine_logs-fine_logs.max())*prior_weights(MEASURED_GRID)
    cdf = cumulative_trapezoid(density, MEASURED_GRID, initial=0.)
    return float(np.interp(.5, cdf/cdf[-1], MEASURED_GRID))


def predict(evidence, fit, ratings, calibration_cases):
    """Return all-player performance estimates without commercial/game identifiers."""
    del calibration_cases
    records = {side: _record(evidence[side]) for side in ('White', 'Black')}
    available = [record for record in records.values() if record is not None]
    methods = BASE_METHODS + tuple(name.replace('_all', '_account_5pct_all') for name in BASE_METHODS)
    result = {name: {} for name in methods}
    if available:
        mean = np.mean([record['mean'] for record in available], axis=0)
        curve = SharedCurve(np.clip(isotonic_regression(mean).x, 0., 100.))(ARGS.grid)
        variance = float(np.mean([record['variance'].mean() for record in available]))
    for side, record in records.items():
        if record is None or fit['players'][side]['estimate'] is None:
            for estimates in result.values():
                estimates[side] = fit['players'][side]['estimate']
            continue
        values = (
            curve_posterior(record['observed'], curve, variance)['unrounded_estimate'],
            _measured_median(record['rank_logs']),
            _measured_median(record['binary_logs']),
        )
        account = float(ratings[side])
        if not np.isfinite(account) or not 0 <= account <= 3200:
            raise ValueError('Account Elo must be finite within0--3200.')
        for name, value in zip(BASE_METHODS, values, strict=True):
            result[name][side] = float(value)
            result[name.replace('_all', '_account_5pct_all')][side] = (
                (1-ACCOUNT_WEIGHT)*float(value) + ACCOUNT_WEIGHT*account)
    return result


def main():
    """Score only after all label-free predictions have been generated."""
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, rank
    from tests.analysis.edge_quality_likelihood import is_edge
    cases, _ = load_cases(ROOT/'games')
    predictions = []
    for case in cases:
        data = case['input']
        predictions.append(predict(**data, calibration_cases=[]))
    rows = []
    for case, prediction in zip(cases, predictions, strict=True):
        data = case['input']
        for method, pair in prediction.items():
            for side, value in pair.items():
                account = '_account_' in method
                rows.append({'game': case['game'], 'side': side, 'method': method,
                             'estimate': value, 'reference': case['references'][side],
                             'edge': bool(is_edge(data['fit'], side)),
                             'own_elo_max_change': 10. if account else 0.,
                             'own_elo_span': 20. if account else 0., 'opponent_elo_max_change': 0.})
    output = ROOT/'tests/analysis/output/universal-rating-methods/move-quality.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = rank(rows)
    output.write_text(json.dumps({'rankings': summary, 'rows': rows,
                                  'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == '__main__':
    main()
