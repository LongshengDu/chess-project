"""Continuous equal-prior model mixture for game strength and global quality.

Model1 is the target game's Gaussian accuracy measurement with its saved sigma;
model2 is the leave-game-out population's moment-matched Beta accuracy model.
Both are normalized measurement distributions on [0,100], observed through the
same fixed0.01-point bin. Equal model priors mix likelihoods before conditioning,
so each model's posterior weight is determined by its integrated evidence.

The point decision is the posterior median (absolute-error loss), with a separately
labelled fixed5% account blend. All observations use the same model; there is no
curve-edge switch. This does not by itself guarantee monotonicity when competing
contexts/variance curves produce different model-specific likelihood orderings.
No commercial-reference labels or fitted correction coefficients are used.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import ndtr

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_density
from tests.analysis.edge_bounded_accuracy import _interval, _stable_mass, beta_accuracy_mass
from tests.analysis.edge_global_quality import _population


MODEL_PRIOR = .5
ACCOUNT_WEIGHT = .05
METHODS = ('predictive_mixture_all', 'predictive_mixture_account_5pct_all')


def gaussian_accuracy_mass(accuracy, mean, variance):
    """Fixed-bin normal probabilities after normalizing accuracy to[0,100]."""
    mean, variance = np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
    if (not np.isfinite(mean).all() or np.any((mean < 0) | (mean > 100))
            or not np.isfinite(variance).all() or np.any(variance < 0)):
        raise ValueError('Finite bounded means and nonnegative variances are required.')
    lower, upper = _interval(accuracy)
    sigma = np.sqrt(np.maximum(variance, 1e-9))
    lower_z, upper_z = (lower-mean)/sigma, (upper-mean)/sigma
    mass = _stable_mass(ndtr(lower_z), ndtr(upper_z), ndtr(-lower_z), ndtr(-upper_z))
    support_lower, support_upper = -mean/sigma, (100.-mean)/sigma
    normalizer = _stable_mass(ndtr(support_lower), ndtr(support_upper),
                              ndtr(-support_lower), ndtr(-support_upper))
    if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0):
        raise ValueError('The truncated Gaussian likelihood has no finite mass.')
    return np.clip(mass/normalizer, 0., 1.)


def posterior(accuracy, target_mean, target_variance, population_mean, population_variance, grid):
    """Integrate the likelihood mixture; arrays of observations may broadcast.

The final array axis is rating. With accuracy shape(N,1), the result contains N
    posteriors and medians. The component evidence weight and mean are returned for auditing.
    """
    grid = np.asarray(grid, dtype=float)
    target_mass = gaussian_accuracy_mass(accuracy, target_mean, target_variance)
    population_mass = beta_accuracy_mass(accuracy, population_mean, population_variance)
    weighted_target = MODEL_PRIOR*target_mass*prior_density(grid)
    weighted_population = (1-MODEL_PRIOR)*population_mass*prior_density(grid)
    weights = weighted_target+weighted_population
    mass = trapezoid(weights, grid, axis=-1)
    if not np.isfinite(weights).all() or np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError('The predictive mixture has no finite posterior mass.')
    density = weights/np.expand_dims(mass, axis=-1)
    mean = trapezoid(density*grid, grid, axis=-1)
    cdf = cumulative_trapezoid(density, grid, axis=-1, initial=0.)
    median = np.array([np.interp(.5, row, grid) for row in cdf.reshape(-1, len(grid))])
    median = median.reshape(density.shape[:-1])
    return {'density': density, 'mean': mean, 'median': median,
            'target_model_probability': trapezoid(weighted_target, grid, axis=-1)/mass}


def predict(evidence, fit, ratings, calibration_cases):
    """Return two full-player predictions, preserving only missing observations."""
    del evidence
    grid = ARGS.grid
    _, calibration_variances, pooled, between = _population(calibration_cases, grid)
    population_variance = between+calibration_variances.mean()
    diagnostic = fit['diagnostics']['curve']
    target_mean = np.asarray(diagnostic['shared_accuracy'], dtype=float)
    if not np.array_equal(np.asarray(diagnostic['fine_ratings']), grid):
        raise ValueError('The target fit must use the current production rating grid.')
    target_variance = float(diagnostic['likelihood']['accuracy_variance'])
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
        result = posterior(observed, target_mean, target_variance, pooled, population_variance, grid)
        estimate = float(result['median'])
        outputs['predictive_mixture_all'][side] = estimate
        outputs['predictive_mixture_account_5pct_all'][side] = (
            (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*account)
    return outputs
