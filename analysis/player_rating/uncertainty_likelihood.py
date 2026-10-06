"""Bounded accuracy likelihoods and shape-constrained coverage decisions."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import isotonic_regression
from scipy.special import betainc, betaincc, ndtr

from analysis.player_rating.bayesian_shared_curve import prior_density
from analysis.player_rating.uncertainty_measurement import ARGS


ACCURACY_BIN_WIDTH = .01
ACCURACY_GRID_STEP = .1
PARAMETER_EPSILON = 1e-9


def _moments(mean, variance):
    mean, variance = np.broadcast_arrays(np.asarray(mean, dtype=float), np.asarray(variance, dtype=float))
    if (not np.isfinite(mean).all() or np.any((mean < 0) | (mean > 100))
            or not np.isfinite(variance).all() or np.any(variance < 0)):
        raise ValueError('Accuracy means must be in [0, 100] and variances finite and nonnegative.')
    return mean, variance


def _interval(accuracy):
    accuracy = np.asarray(accuracy, dtype=float)
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 100)):
        raise ValueError('Observed accuracy must be finite and lie in [0, 100].')
    return np.maximum(0., accuracy-ACCURACY_BIN_WIDTH/2), np.minimum(100., accuracy+ACCURACY_BIN_WIDTH/2)


def _stable_mass(lower, upper, lower_survival, upper_survival):
    return np.maximum(np.where(lower < .5, upper-lower, lower_survival-upper_survival), 0.)


def beta_accuracy_mass(accuracy, mean, variance):
    """Moment-matched Beta probability in a fixed 0.01-point accuracy bin.

    Variance is restricted strictly below the Bernoulli bound, and endpoint
    means have a 1e-9 numerical floor. These are explicit feasibility safeguards.
    """
    mean, variance = _moments(mean, variance)
    probability = np.clip(mean/100., PARAMETER_EPSILON, 1-PARAMETER_EPSILON)
    maximum = probability*(1-probability)
    unit_variance = np.clip(variance/10000., maximum*PARAMETER_EPSILON,
                            maximum*(1-PARAMETER_EPSILON))
    concentration = maximum/unit_variance-1
    alpha, beta = probability*concentration, (1-probability)*concentration
    lower, upper = _interval(accuracy)
    lower, upper = lower/100., upper/100.
    return _stable_mass(betainc(alpha, beta, lower), betainc(alpha, beta, upper),
                        betaincc(alpha, beta, lower), betaincc(alpha, beta, upper))


def gaussian_accuracy_mass(accuracy, mean, variance):
    """Fixed-bin Gaussian probability after truncation and normalization to 0–100."""
    mean, variance = _moments(mean, variance)
    lower, upper = _interval(accuracy)
    sigma = np.sqrt(np.maximum(variance, 1e-9))
    lower_z, upper_z = (lower-mean)/sigma, (upper-mean)/sigma
    mass = _stable_mass(ndtr(lower_z), ndtr(upper_z), ndtr(-lower_z), ndtr(-upper_z))
    low, high = -mean/sigma, (100.-mean)/sigma
    normalizer = _stable_mass(ndtr(low), ndtr(high), ndtr(-low), ndtr(-high))
    if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0):
        raise ValueError('The truncated Gaussian likelihood has no finite mass.')
    return np.clip(mass/normalizer, 0., 1.)


def posterior_mean(mass, grid):
    """Integrate likelihood times the common prior; flag underflow explicitly."""
    weights = np.asarray(mass, dtype=float)*prior_density(grid, args=ARGS)
    total = trapezoid(weights, grid, axis=-1)
    valid = np.isfinite(total) & (total > 0)
    density = np.divide(weights, np.expand_dims(total, -1), out=np.zeros_like(weights),
                        where=np.expand_dims(valid, -1))
    return trapezoid(density*grid, grid, axis=-1), valid


def predictive_mean(accuracy, target_mean, target_variance, population_mean, population_variance,
                    target_model_prior=.5):
    """Mix proper Gaussian/Beta likelihoods before taking the posterior mean."""
    grid = ARGS.grid
    target = target_model_prior*gaussian_accuracy_mass(accuracy, target_mean, target_variance)
    population = (1-target_model_prior)*beta_accuracy_mass(accuracy, population_mean, population_variance)
    point, valid = posterior_mean(target+population, grid)
    if not np.all(valid):
        raise ValueError('The predictive mixture has no finite posterior mass.')
    total = trapezoid((target+population)*prior_density(grid, args=ARGS), grid, axis=-1)
    probability = trapezoid(target*prior_density(grid, args=ARGS), grid, axis=-1)/total
    return {'mean': float(point), 'target_model_probability': float(probability)}


def native_reliability(accuracy, low, high, variance):
    """Probability that uncertain observed accuracy lies within the measured curve."""
    if not np.isfinite([low, high, variance]).all() or not 0 <= low <= high <= 100 or variance < 0:
        raise ValueError('Valid measured accuracy bounds and nonnegative variance are required.')
    sigma = np.sqrt(max(float(variance), 1e-9))
    return np.clip(ndtr((high-np.asarray(accuracy))/sigma)-ndtr((low-np.asarray(accuracy))/sigma), 0., 1.)


@lru_cache(maxsize=64)
def _population_mapping(mean, variance):
    accuracy = np.linspace(0., 100., int(round(100/ACCURACY_GRID_STEP))+1)
    points, valid = posterior_mean(beta_accuracy_mass(accuracy[:, None], mean, variance), ARGS.grid)
    if not np.all(valid):
        raise ValueError('The population likelihood has no finite posterior mass.')
    points.setflags(write=False)
    return points


@lru_cache(maxsize=128)
def _coverage_mapping(target_mean, target_variance, population_mean, population_variance, native_bounds):
    accuracy = np.linspace(0., 100., int(round(100/ACCURACY_GRID_STEP))+1)
    reliability = native_reliability(accuracy, *native_bounds, target_variance)
    local, valid = posterior_mean(gaussian_accuracy_mass(accuracy[:, None], target_mean, target_variance), ARGS.grid)
    population = _population_mapping(population_mean, population_variance)
    if np.any((~valid) & (reliability > 1e-12)):
        raise ValueError('An active local likelihood has no finite posterior mass.')
    # At local underflow, the omitted influence is bounded by 3.2e-9 Elo.
    raw = np.where(valid, reliability*local+(1-reliability)*population, population)
    projected = isotonic_regression(raw, increasing=True).x
    for values in (accuracy, projected):
        values.setflags(write=False)
    return {'accuracy_grid': accuracy, 'mean': projected,
            'maximum_projection_change': float(np.max(abs(projected-raw))),
            'decreasing_steps': int(np.sum(np.diff(raw) < -1e-7))}


def coverage_mean(accuracy, target_mean, target_variance, population_mean, population_variance, native_bounds):
    """Interpolate the nondecreasing coverage decision map, without edge switches."""
    mapping = _coverage_mapping(tuple(target_mean), float(target_variance), tuple(population_mean),
                                tuple(population_variance), tuple(native_bounds))
    return {'mean': float(np.interp(accuracy, mapping['accuracy_grid'], mapping['mean'])),
            'native_reliability': float(native_reliability(accuracy, *native_bounds, target_variance)),
            'maximum_projection_change': mapping['maximum_projection_change'],
            'decreasing_steps': mapping['decreasing_steps']}
