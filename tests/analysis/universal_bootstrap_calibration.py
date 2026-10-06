"""Parametric-bootstrap response calibration of the proper-mixture posterior mean.

For each known synthetic rating r, integrate B(r)=E[T(A)|r] under the unchanged
Gaussian/Beta mixture measurement distribution. T is the original posterior mean.
This estimates prior-induced shrinkage using Maia rating labels, never commercial
references. The fixed0.25-point accuracy grid uses exact cell probabilities with
endpoint halfcells; it is a deterministic quadrature, not simulated random noise.

Two standard corrections are tested: invert the monotone estimator-response map,
and one-step bootstrap bias correction2T-B(T). Both are constrained to the existing
0--3200 rating domain. The resulting estimates are bias-corrected decisions, not
the original Bayesian posterior means or newly calibrated credible intervals.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.special import betainc, betaincc, ndtr

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_bounded_accuracy import _stable_mass, beta_parameters
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior


ACCURACY_STEP = .25
ACCOUNT_WEIGHT = .05
BASE_METHODS = ('bootstrap_inverse_mean_all', 'bootstrap_onestep_mean_all')
METHODS = tuple(name for base in BASE_METHODS for name in (base, base[:-4]+'_account_5pct_all'))


def measurement_cells(accuracy_grid, target_mean, target_variance, population_mean, population_variance):
    """Exact probabilities of adjacent accuracy cells, one distribution per rating."""
    accuracy_grid = np.asarray(accuracy_grid, dtype=float)
    if (accuracy_grid.ndim != 1 or len(accuracy_grid) < 2 or accuracy_grid[0] != 0
            or accuracy_grid[-1] != 100 or np.any(np.diff(accuracy_grid) <= 0)):
        raise ValueError('Accuracy quadrature must span0--100 in increasing order.')
    boundaries = np.r_[0., (accuracy_grid[:-1]+accuracy_grid[1:])/2, 100.]
    lower, upper = boundaries[:-1, None], boundaries[1:, None]
    target_mean = np.asarray(target_mean)
    sigma = np.sqrt(max(float(target_variance), 1e-9))
    lower_z, upper_z = (lower-target_mean)/sigma, (upper-target_mean)/sigma
    normal_mass = _stable_mass(ndtr(lower_z), ndtr(upper_z), ndtr(-lower_z), ndtr(-upper_z))
    normalizer = ndtr((100.-target_mean)/sigma)-ndtr(-target_mean/sigma)
    normal_mass /= normalizer
    alpha, beta = beta_parameters(population_mean, population_variance)
    beta_mass = _stable_mass(betainc(alpha, beta, lower/100.), betainc(alpha, beta, upper/100.),
                             betaincc(alpha, beta, lower/100.), betaincc(alpha, beta, upper/100.))
    mass = .5*(normal_mass+beta_mass)
    total = mass.sum(axis=0)
    if not np.isfinite(mass).all() or np.any(total <= 0):
        raise ValueError('The synthetic accuracy distribution must have finite positive mass.')
    return mass/total


@lru_cache(maxsize=64)
def _cached_response(target_mean, target_variance, population_mean, population_variance, rating_grid):
    grid = np.asarray(rating_grid, dtype=float)
    target_mean, population_mean, population_variance = map(np.asarray, (target_mean, population_mean, population_variance))
    accuracy = np.linspace(0., 100., int(round(100./ACCURACY_STEP))+1)
    decisions = posterior(accuracy[:, None], target_mean, target_variance,
                          population_mean, population_variance, grid)['mean']
    cells = measurement_cells(accuracy, target_mean, target_variance, population_mean, population_variance)
    raw_response = cells.T @ decisions
    response = isotonic_regression(raw_response, increasing=True).x
    for values in (accuracy, decisions, raw_response, response):
        values.setflags(write=False)
    return {'accuracy_grid': accuracy, 'decisions': decisions, 'raw_response': raw_response,
            'response': response, 'maximum_shape_correction': float(np.max(abs(response-raw_response)))}


def response_curve(target_mean, target_variance, population_mean, population_variance, rating_grid):
    return _cached_response(tuple(target_mean), float(target_variance), tuple(population_mean),
                             tuple(population_variance), tuple(rating_grid))


def correct_estimate(estimate, grid, response):
    """Invert B using plateau midpoints and apply the standard one-step correction."""
    grid, response = np.asarray(grid, dtype=float), np.asarray(response, dtype=float)
    if (response.shape != grid.shape or not np.isfinite(response).all()
            or not np.isfinite(estimate) or np.any(np.diff(grid) <= 0) or np.any(np.diff(response) < 0)):
        raise ValueError('A finite nondecreasing response on an increasing rating grid is required.')
    levels, indices = np.unique(response, return_inverse=True)
    rating_centers = np.bincount(indices, weights=grid)/np.bincount(indices)
    inverse = float(np.interp(estimate, levels, rating_centers, left=grid[0], right=grid[-1]))
    onestep = float(np.clip(2*estimate-np.interp(estimate, grid, response), grid[0], grid[-1]))
    return inverse, onestep


def predict(evidence, fit, ratings, calibration_cases):
    """Apply the same synthetic calibration to both players, with no edge switch."""
    del evidence
    grid = ARGS.grid
    _, conditional_variances, pooled, between = _population(calibration_cases, grid)
    population_variance = between+conditional_variances.mean()
    diagnostic = fit['diagnostics']['curve']
    target_mean = np.asarray(diagnostic['shared_accuracy'])
    target_variance = diagnostic['likelihood']['accuracy_variance']
    if not np.array_equal(np.asarray(diagnostic['fine_ratings']), grid):
        raise ValueError('The target fit must use the production rating grid.')
    response = response_curve(target_mean, target_variance, pooled, population_variance, grid)['response']
    outputs = {name: {} for name in METHODS}
    for side in ('White', 'Black'):
        player = fit['players'][side]
        if player['average_accuracy'] is None or player['estimate'] is None:
            for values in outputs.values():
                values[side] = player['estimate']
            continue
        account = ratings.get(side)
        if account is None or not np.isfinite(account) or not 0 <= account <= 3200:
            raise ValueError('Account ratings must be finite and lie in[0,3200].')
        original = float(posterior(player['average_accuracy'], target_mean, target_variance,
                                   pooled, population_variance, grid)['mean'])
        for name, estimate in zip(BASE_METHODS, correct_estimate(original, grid, response), strict=True):
            outputs[name][side] = estimate
            outputs[name[:-4]+'_account_5pct_all'][side] = (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*account
    return outputs
