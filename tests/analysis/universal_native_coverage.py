"""Universal native-curve coverage weighting with shape-constrained point decisions.

For every observed arithmetic accuracy A, local reliability is the Gaussian mass
Phi((C2600-A)/sigma)-Phi((C600-A)/sigma). This treats intersection as uncertain,
with one continuous formula for all players; there is no inside/outside branch.
The weight blends local Gaussian and population Beta posterior point estimates.

Mean and median decisions are evaluated on the same fixed0--100,step0.1 grid.
Any decreasing segments are projected by equal-weight isotonic regression before
interpolation. These blended/projected points are decision rules, not the median
or mean of a newly claimed Bayesian posterior. No commercial labels set weights.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import isotonic_regression
from scipy.special import ndtr

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_density
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass


ACCURACY_STEP = .1
ACCOUNT_WEIGHT = .05
BASE_METHODS = ('native_coverage_mean_all', 'native_coverage_median_all')
METHODS = tuple(name for base in BASE_METHODS for name in (base, base[:-4]+'_account_5pct_all'))


def native_reliability(accuracy, low, high, variance):
    """Gaussian probability that underlying accuracy lies on the measured curve."""
    if not np.isfinite([low, high, variance]).all() or not 0 <= low <= high <= 100 or variance < 0:
        raise ValueError('Valid measured accuracy bounds and nonnegative variance are required.')
    sigma = np.sqrt(max(float(variance), 1e-9))
    accuracy = np.asarray(accuracy, dtype=float)
    return np.clip(ndtr((high-accuracy)/sigma)-ndtr((low-accuracy)/sigma), 0., 1.)


def _points(mass, grid):
    weights = np.asarray(mass, dtype=float)*prior_density(grid)
    total = trapezoid(weights, grid, axis=-1)
    valid = np.isfinite(total) & (total > 0)
    density = np.divide(weights, total[:, None], out=np.zeros_like(weights), where=valid[:, None])
    cdf = cumulative_trapezoid(density, grid, axis=-1, initial=0.)
    return {'mean': trapezoid(density*grid, grid, axis=-1),
            'median': np.array([np.interp(.5, row, grid) for row in cdf]), 'valid': valid}


@lru_cache(maxsize=64)
def _cached_mapping(target_mean, target_variance, population_mean, population_variance, native_bounds, rating_grid):
    grid = np.asarray(rating_grid)
    target_mean, population_mean, population_variance = map(np.asarray, (target_mean, population_mean, population_variance))
    accuracy = np.linspace(0., 100., int(round(100./ACCURACY_STEP))+1)
    reliability = native_reliability(accuracy, *native_bounds, target_variance)
    local = _points(gaussian_accuracy_mass(accuracy[:, None], target_mean, target_variance), grid)
    population = _points(beta_accuracy_mass(accuracy[:, None], population_mean, population_variance), grid)
    if not np.all(population['valid']) or np.any((~local['valid']) & (reliability > 1e-12)):
        raise ValueError('An active likelihood component has no finite posterior mass.')
    mapping = {'accuracy_grid': accuracy, 'reliability': reliability}
    for point in ('mean', 'median'):
        # At Gaussian numerical underflow, its reliability is below1e-12, so its
        # omitted influence is bounded by3.2e-9Elo on the declared rating domain.
        raw = np.where(local['valid'], reliability*local[point]+(1-reliability)*population[point], population[point])
        projected = isotonic_regression(raw, increasing=True).x
        mapping[point] = projected
        mapping[point+'_raw'] = raw
        mapping[point+'_decreasing_steps'] = int(np.sum(np.diff(raw) < -1e-7))
        mapping[point+'_maximum_correction'] = float(np.max(abs(projected-raw)))
    for value in mapping.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return mapping


def accuracy_mapping(target_mean, target_variance, population_mean, population_variance, native_bounds, rating_grid):
    return _cached_mapping(tuple(target_mean), float(target_variance), tuple(population_mean),
                            tuple(population_variance), tuple(native_bounds), tuple(rating_grid))


def predict(evidence, fit, ratings, calibration_cases):
    """Apply native-coverage reliability and identical point rules to all players."""
    del evidence
    grid = ARGS.grid
    _, variances, pooled, between = _population(calibration_cases, grid)
    diagnostic = fit['diagnostics']['curve']
    if not np.array_equal(np.asarray(diagnostic['fine_ratings']), grid):
        raise ValueError('The target fit must use the current production rating grid.')
    knots = diagnostic['monotone_expected_accuracy']
    mapping = accuracy_mapping(diagnostic['shared_accuracy'], diagnostic['likelihood']['accuracy_variance'],
                               pooled, between+variances.mean(), [knots[0], knots[-1]], grid)
    output = {name: {} for name in METHODS}
    for side in ('White', 'Black'):
        player = fit['players'][side]
        if player['average_accuracy'] is None or player['estimate'] is None:
            for values in output.values():
                values[side] = player['estimate']
            continue
        account = ratings.get(side)
        if account is None or not np.isfinite(account) or not 0 <= account <= 3200:
            raise ValueError('Account ratings must be finite and lie in[0,3200].')
        for point, name in zip(('mean', 'median'), BASE_METHODS, strict=True):
            value = float(np.interp(player['average_accuracy'], mapping['accuracy_grid'], mapping[point]))
            output[name][side] = value
            output[name[:-4]+'_account_5pct_all'][side] = (1-ACCOUNT_WEIGHT)*value+ACCOUNT_WEIGHT*account
    return output
