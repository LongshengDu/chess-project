"""Shape-constrained projection of predictive-mixture rating estimates.

For a fixed game and calibration context, evaluate the proper mixture posterior
median at arithmetic accuracies 0--100 in steps of0.1. Project that entire mapping
onto nondecreasing sequences by equal-weight least squares, then interpolate at
the observed accuracy. The fixed grid represents a resolution of one tenth of an
accuracy point; it is not fitted to reference estimates or observed target values.

The projected point is a shape-constrained estimate, NOT the original Bayesian
posterior median. It does not imply that projected posterior densities or credible
intervals exist. Calibration is Maia-only and must exclude the target game.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import isotonic_regression

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior


ACCURACY_STEP = .1
ACCOUNT_WEIGHT = .05
METHODS = ('monotone_predictive_mixture_all', 'monotone_predictive_mixture_account_5pct_all')


@lru_cache(maxsize=64)
def _cached_mapping(target_mean, target_variance, population_mean, population_variance, rating_grid):
    """Cache by numerical model inputs only; accounts and references cannot enter."""
    grid = np.asarray(rating_grid, dtype=float)
    accuracies = np.linspace(0., 100., int(round(100./ACCURACY_STEP))+1)
    result = posterior(accuracies[:, None], np.asarray(target_mean), target_variance,
                       np.asarray(population_mean), np.asarray(population_variance), grid)
    raw = np.asarray(result['median'], dtype=float)
    if raw.shape != accuracies.shape or not np.isfinite(raw).all():
        raise ValueError('The whole accuracy domain must produce finite mixture medians.')
    projected = isotonic_regression(raw, increasing=True).x
    for values in (accuracies, raw, projected):
        values.setflags(write=False)
    return accuracies, raw, projected


def accuracy_mapping(target_mean, target_variance, population_mean, population_variance, rating_grid):
    """Return read-only accuracy/raw/projected arrays, cached by numeric context."""
    inputs = [np.asarray(values, dtype=float) for values in
              (target_mean, population_mean, population_variance, rating_grid)]
    if (any(values.ndim != 1 or not len(values) or not np.isfinite(values).all() for values in inputs)
            or any(values.shape != inputs[-1].shape for values in inputs[:-1])
            or not np.isfinite(target_variance) or target_variance < 0):
        raise ValueError('Finite same-length curve/variance/grid arrays are required.')
    target_mean, population_mean, population_variance, grid = inputs
    if np.any(np.diff(grid) <= 0):
        raise ValueError('The rating integration grid must be strictly increasing.')
    return _cached_mapping(tuple(target_mean), float(target_variance), tuple(population_mean),
                            tuple(population_variance), tuple(grid))


def interpolate(mapping, accuracy):
    """Linear interpolation of the nondecreasing map, within its full domain."""
    accuracy = np.asarray(accuracy, dtype=float)
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 100)):
        raise ValueError('Observed arithmetic accuracy must lie in[0,100].')
    grid, _, projected = mapping
    return np.interp(accuracy, grid, projected)


def predict(evidence, fit, ratings, calibration_cases):
    """Return two full-player projected estimates; no reference or hard edge gate."""
    del evidence
    grid = ARGS.grid
    _, calibration_variances, pooled, between = _population(calibration_cases, grid)
    population_variance = between+calibration_variances.mean()
    diagnostic = fit['diagnostics']['curve']
    if not np.array_equal(np.asarray(diagnostic['fine_ratings']), grid):
        raise ValueError('The target fit must use the current production rating grid.')
    mapping = accuracy_mapping(diagnostic['shared_accuracy'], diagnostic['likelihood']['accuracy_variance'],
                               pooled, population_variance, grid)
    outputs = {name: {} for name in METHODS}
    for side in ('White', 'Black'):
        player = fit['players'][side]
        observed = player['average_accuracy']
        if observed is None or player['estimate'] is None:
            for values in outputs.values():
                values[side] = player['estimate']
            continue
        account = ratings.get(side)
        if account is None or not np.isfinite(account) or not 0 <= account <= 3200:
            raise ValueError('Account Elo must be finite and lie in[0,3200].')
        estimate = float(interpolate(mapping, observed))
        outputs['monotone_predictive_mixture_all'][side] = estimate
        outputs['monotone_predictive_mixture_account_5pct_all'][side] = (
            (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*account)
    return outputs
