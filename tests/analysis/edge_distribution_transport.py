"""Leave-game-out quality-distribution fallbacks, with no reference calibration.

Three fixed measurement models retain information arithmetic accuracy discards:
log quality, reciprocal quality, and a Dirichlet-multinomial quality histogram.
Calibration cases supply full Maia-predicted quality distributions only; their
played moves, account ratings, saved estimates and commercial labels are unused.

The first two models transport the target's transformed average onto the common
Maia population scale. Population uncertainty follows the law of total variance:
between-game mean variance plus expected within-position variance / move count.
The histogram model instead matches the entire observed severity distribution;
between-game distribution variance determines its Dirichlet concentration.

Fixed choices: one accuracy-point regularization for log/reciprocal transforms,
six interpretable quality bins, at most 20 effective moves for serial dependence,
and 5% account-rating shrinkage. Account perturbations of +/-200 therefore alter
estimates by at most 10 Elo. The support remains Maia's measured 600--2600 range.
No curve-shape or reference-specific correction is learned from the test games.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import gammaln

from analysis.player_rating.bayesian_shared_curve import prior_weights
from analysis.player_rating.parameters import RATINGS
from tests.analysis.edge_quality_likelihood import BUCKET_EDGES, is_edge


METHODS = ('transport_log_quality', 'transport_harmonic_quality', 'transport_dirichlet_quality')
GRID = np.asarray(RATINGS, dtype=float)
FINE_GRID = np.arange(GRID[0], GRID[-1] + 1, 5.)
ACCOUNT_WEIGHT = .05
MAX_EFFECTIVE_MOVES = 20
QUALITY_REGULARIZER = 1.


def _transforms(qualities):
    """Monotone 0--100 transforms emphasizing different parts of the error tail."""
    q = np.asarray(qualities, dtype=float)
    epsilon = QUALITY_REGULARIZER
    log_quality = 100 * np.log1p(q / epsilon) / np.log1p(100 / epsilon)
    reciprocal_quality = 100 * (1 - epsilon / (epsilon + q)) / (1 - epsilon / (epsilon + 100))
    return np.stack((log_quality, reciprocal_quality))


def _record_distribution(record, *, observe=False):
    """Compute conditional moments/histograms, ignoring played indices in calibration."""
    means, variances, histograms, observed = [], [], [], []
    counts = np.zeros(len(BUCKET_EDGES) - 1)
    for row in record['observations']:
        q = np.asarray(row['qualities']['position'], dtype=float)
        p = np.asarray(row['maia_probabilities'], dtype=float)
        if (q.ndim != 1 or not len(q) or p.shape != (len(GRID), len(q))
                or not np.isfinite(q).all() or np.any((q < 0) | (q > 100))
                or not np.isfinite(p).all() or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1.)):
            raise ValueError('Complete normalized Maia policies and bounded qualities are required.')
        if len(q) == 1:
            continue
        t = _transforms(q)
        mean = t @ p.T
        means.append(mean)
        variances.append(np.maximum(0., (t * t) @ p.T - mean * mean))
        buckets = np.searchsorted(BUCKET_EDGES, q, side='right') - 1
        histograms.append(np.stack([p @ (buckets == bucket)
                                    for bucket in range(len(counts))], axis=1))
        if observe:
            index = row['played_index']
            if not isinstance(index, int) or not 0 <= index < len(q):
                raise ValueError('The played index must identify one legal move.')
            observed.append(t[:, index])
            counts[buckets[index]] += 1
    if not means:
        return None
    return {'means': np.mean(means, axis=0), 'variances': np.mean(variances, axis=0),
            'histograms': np.mean(histograms, axis=0), 'moves': len(means),
            'observed': np.mean(observed, axis=0) if observe else None,
            'counts': counts if observe else None}


def _population(cases):
    """Give each game one vote, and each available side half of that game's vote."""
    means, variances, histograms = [], [], []
    for case in cases:
        sides = [_record_distribution(case['evidence'][side]) for side in ('White', 'Black')]
        sides = [side for side in sides if side is not None]
        if sides:
            means.append(np.mean([side['means'] for side in sides], axis=0))
            variances.append(np.mean([side['variances'] for side in sides], axis=0))
            histograms.append(np.mean([side['histograms'] for side in sides], axis=0))
    if len(means) < 2:
        raise ValueError('At least two independent calibration games are required.')
    histogram_mean = np.mean(histograms, axis=0)
    histogram_variance = np.var(histograms, axis=0, ddof=1).sum(axis=1)
    numerator = (histogram_mean * (1 - histogram_mean)).sum(axis=1)
    concentration = np.clip(numerator / np.maximum(histogram_variance, 1e-12) - 1, 1., 1e6)
    return {'mean': np.mean(means, axis=0), 'between': np.var(means, axis=0, ddof=1),
            'within': np.mean(variances, axis=0), 'histogram': histogram_mean,
            'concentration': concentration}


def _median(log_likelihood):
    logs = np.interp(FINE_GRID, GRID, log_likelihood)
    density = np.exp(logs - logs.max()) * prior_weights(FINE_GRID)
    cdf = cumulative_trapezoid(density, FINE_GRID, initial=0.)
    return float(np.interp(.5, cdf / cdf[-1], FINE_GRID))


def predict(evidence, fit, ratings, calibration_cases, *, edge_only=True):
    """Predict edge ratings using a leave-game-out corpus of Maia quality policies."""
    results = {name: {} for name in METHODS}
    population = _population(calibration_cases)
    for side in ('White', 'Black'):
        baseline = fit['players'][side]['estimate']
        for result in results.values():
            result[side] = baseline
        if baseline is None or (edge_only and not is_edge(fit, side)):
            continue
        target = _record_distribution(evidence[side], observe=True)
        if target is None:
            continue
        account = float(ratings[side])
        if not math.isfinite(account):
            raise ValueError('Actual account ratings must be finite.')
        n = min(target['moves'], MAX_EFFECTIVE_MOVES)
        variance = np.maximum(population['between'] + population['within'] / n, 1e-12)
        logs = -.5 * ((target['observed'][:, None] - population['mean']) ** 2 / variance
                      + np.log(variance))
        concentration = population['concentration']
        alpha = np.maximum(population['histogram'], 1e-12)
        alpha = alpha / alpha.sum(axis=1, keepdims=True) * concentration[:, None]
        counts = target['counts'] * n / target['moves']
        histogram_logs = (gammaln(concentration) - gammaln(concentration + n)
                          + (gammaln(alpha + counts[None, :]) - gammaln(alpha)).sum(axis=1))
        for name, likelihood in zip(METHODS, [logs[0], logs[1], histogram_logs], strict=True):
            results[name][side] = (1 - ACCOUNT_WEIGHT) * _median(likelihood) + ACCOUNT_WEIGHT * account
    return results
