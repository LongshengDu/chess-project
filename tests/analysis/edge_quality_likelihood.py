"""Reference-free, per-move quality likelihoods for shared-curve edge cases.

These are experiments, not production fitters. Constants are selected for an
interpretable measurement model before reference evaluation:

* Accuracy buckets separate severe errors, moderate errors and near-perfect play.
* A five-point Gaussian kernel permits evaluation/model discrepancies.
* A Bernoulli model asks only whether move accuracy is at least 95.
* Five percent contamination prevents a tiny Maia probability from dominating.
* At most 20 effective observations limit confidence from correlated game moves.

The account-rating variants apply a fixed five-percent shrinkage to the estimate.
Consequently, changing supplied Elo by 200 changes the output by at most 10 Elo.
This is an explicit robustness choice, not a calibration to the commercial data.
All methods retain the existing estimate exactly for observations intersecting
the measured shared curve. Edge likelihoods are restricted to Maia's measured
600--2600 range; they do not invent quality distributions beyond that range.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import ndtr

from analysis.player_rating.bayesian_shared_curve import prior_weights
from analysis.player_rating.parameters import RATINGS


BUCKET_EDGES = np.array([0., 50., 70., 85., 95., 99.5, 100.000001])
QUALITY_KERNEL_SIGMA = 5.
GOOD_MOVE_ACCURACY = 95.
MODEL_CONTAMINATION = .05
MAX_EFFECTIVE_MOVES = 20
ACCOUNT_WEIGHT = .05
GRID = np.asarray(RATINGS, dtype=float)
FINE_GRID = np.arange(GRID[0], GRID[-1] + 1, 5.)
BASE_METHODS = ('quality_bucket', 'quality_kernel', 'quality_bernoulli')
METHODS = BASE_METHODS + tuple(name + '_account' for name in BASE_METHODS)


def is_edge(fit, side):
    """Classify against measured knots, never the synthetic exponential tails."""
    accuracy = fit['players'][side]['average_accuracy']
    knots = np.asarray(fit['diagnostics']['curve']['monotone_expected_accuracy'])
    return accuracy is not None and (accuracy < knots[0] or accuracy > knots[-1])


def _quality_logs(record):
    """Return independent categorical/kernel/binary composite log likelihoods."""
    logs = {name: np.zeros_like(GRID) for name in BASE_METHODS}
    count = 0
    for row in record['observations']:
        q = np.asarray(row['qualities']['position'], dtype=float)
        policy = np.asarray(row['maia_probabilities'], dtype=float)
        index = row['played_index']
        if (q.ndim != 1 or not len(q) or policy.shape != (len(GRID), len(q))
                or not 0 <= index < len(q) or not np.isfinite(q).all()
                or np.any((q < 0) | (q > 100)) or not np.isfinite(policy).all()
                or np.any(policy < 0) or not np.allclose(policy.sum(axis=1), 1.)):
            raise ValueError('Complete normalized Maia policies and legal move qualities are required.')
        if len(q) == 1:
            continue
        count += 1
        observed = q[index]
        buckets = np.searchsorted(BUCKET_EDGES, q, side='right') - 1
        same_bucket = buckets == buckets[index]
        bucket_mass = policy @ same_bucket
        bucket_mass = ((1 - MODEL_CONTAMINATION) * bucket_mass
                       + MODEL_CONTAMINATION / (len(BUCKET_EDGES) - 1))

        kernel = np.exp(-.5 * ((q - observed) / QUALITY_KERNEL_SIGMA) ** 2)
        # Expected kernel under an uninformative uniform quality distribution.
        uniform_kernel = (QUALITY_KERNEL_SIGMA * math.sqrt(2 * math.pi) / 100
                          * (ndtr((100 - observed) / QUALITY_KERNEL_SIGMA)
                             - ndtr(-observed / QUALITY_KERNEL_SIGMA)))
        kernel_mass = ((1 - MODEL_CONTAMINATION) * (policy @ kernel)
                       + MODEL_CONTAMINATION * uniform_kernel)

        good = q >= GOOD_MOVE_ACCURACY
        binary_mass = policy @ (good if good[index] else ~good)
        binary_mass = (1 - MODEL_CONTAMINATION) * binary_mass + MODEL_CONTAMINATION / 2
        for name, mass in zip(BASE_METHODS, (bucket_mass, kernel_mass, binary_mass), strict=True):
            logs[name] += np.log(np.maximum(mass, 1e-300))
    scale = min(1., MAX_EFFECTIVE_MOVES / count) if count else 1.
    return {name: values * scale for name, values in logs.items()}, count


def _median(log_likelihood):
    """Integrate on native Maia support with the unchanged production prior."""
    logs = np.interp(FINE_GRID, GRID, log_likelihood)
    density = np.exp(logs - logs.max()) * prior_weights(FINE_GRID)
    cdf = cumulative_trapezoid(density, FINE_GRID, initial=0.)
    cdf /= cdf[-1]
    return float(np.interp(.5, cdf, FINE_GRID))


def predict(evidence, fit, ratings):
    """Return six edge-only predictions; no game identity or references enter."""
    predictions = {name: {} for name in METHODS}
    for side in ('White', 'Black'):
        existing = fit['players'][side]['estimate']
        for result in predictions.values():
            result[side] = existing
        if existing is None or not is_edge(fit, side):
            continue
        logs, count = _quality_logs(evidence[side])
        if not count:
            continue
        account = float(ratings[side])
        if not math.isfinite(account):
            raise ValueError('Account ratings must be finite.')
        for name, values in logs.items():
            estimate = _median(values)
            predictions[name][side] = estimate
            predictions[name + '_account'][side] = (
                (1 - ACCOUNT_WEIGHT) * estimate + ACCOUNT_WEIGHT * account)
    return predictions
