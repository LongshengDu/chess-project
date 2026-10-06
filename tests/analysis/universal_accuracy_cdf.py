"""Probability-matching accuracy inversion on the rating coordinate.

For a monotone measurement CDF F(accuracy|rating), its negative rating derivative
defines a confidence-density analogue. This includes how quickly the measurement
distribution changes with rating, unlike a density likelihood with uniform Elo
prior. The original support taper is retained. This is a confidence inversion,
NOT the original Bayesian posterior and not a calibrated frequentist interval.

CDFs use the midpoint of the same0.01-point observation bin used by bounded
likelihood experiments. Minor stochastic-order violations are projected onto
nonincreasing CDFs before taking the numerical derivative. Point estimates are
confidence-density means; account variants add the fixed5% account contribution.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import isotonic_regression
from scipy.special import betainc, ndtr

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_weights
from tests.analysis.edge_bounded_accuracy import _interval, beta_parameters
from tests.analysis.edge_global_quality import _population
from tests.analysis.universal_accuracy_likelihood import _numeric_fit


ACCOUNT_WEIGHT = .05
BASE_METHODS = ('accuracy_cdf_target_all', 'accuracy_cdf_population_all', 'accuracy_cdf_mixture_all')
METHODS = tuple(name for base in BASE_METHODS for name in (base, base[:-4]+'_account_5pct_all'))


def normal_mid_cdf(accuracy, mean, variance):
    """Mid-bin CDF of a Gaussian normalized on accuracy0--100."""
    lower, upper = _interval(accuracy)
    mean = np.asarray(mean, dtype=float)
    sigma = np.sqrt(max(float(variance), 1e-9))
    support_lower = ndtr(-mean/sigma)
    normalizer = ndtr((100.-mean)/sigma)-support_lower
    if np.any(normalizer <= 0):
        raise ValueError('The truncated Gaussian has no finite mass.')
    return np.clip((.5*(ndtr((lower-mean)/sigma)+ndtr((upper-mean)/sigma))-support_lower)/normalizer, 0., 1.)


def beta_mid_cdf(accuracy, mean, variance):
    lower, upper = _interval(accuracy)
    alpha, beta = beta_parameters(mean, variance)
    return .5*(betainc(alpha, beta, lower/100.)+betainc(alpha, beta, upper/100.))


def invert_cdf(values, grid):
    """Normalize the rating derivative of the decreasing measurement CDF."""
    values, grid = np.asarray(values, dtype=float), np.asarray(grid, dtype=float)
    if (values.shape != grid.shape or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1)) or np.any(np.diff(grid) <= 0)):
        raise ValueError('A finite bounded CDF on an increasing rating grid is required.')
    monotone = isotonic_regression(values, increasing=False).x
    derivative = np.maximum(0., -np.gradient(monotone, grid))
    weights = derivative*prior_weights(grid)
    mass = trapezoid(weights, grid)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('The confidence inversion has no identifiable rating mass.')
    density = weights/mass
    return {'estimate': float(trapezoid(grid*density, grid)), 'density': density,
            'maximum_order_correction': float(np.max(abs(monotone-values)))}


@lru_cache(maxsize=64)
def _cached_fit(target_json, calibration_json):
    import json
    target = json.loads(target_json)
    calibration = [{'fit': json.loads(value)} for value in calibration_json]
    _, variances, pooled, between = _population(calibration, ARGS.grid)
    own = target['diagnostics']['curve']
    predictions = {method: {} for method in BASE_METHODS}
    for side in ('White', 'Black'):
        player = target['players'][side]
        accuracy = player['average_accuracy']
        if accuracy is None or player['estimate'] is None:
            for values in predictions.values():
                values[side] = player['estimate']
            continue
        local_cdf = normal_mid_cdf(accuracy, own['shared_accuracy'], own['likelihood']['accuracy_variance'])
        population_cdf = beta_mid_cdf(accuracy, pooled, between+variances.mean())
        for name, cdf in zip(BASE_METHODS, (local_cdf, population_cdf, .5*(local_cdf+population_cdf)), strict=True):
            predictions[name][side] = invert_cdf(cdf, ARGS.grid)['estimate']
    return predictions


def predict(evidence, fit, ratings, calibration_cases):
    """Use one fixed inverse-CDF rule for every observed player in every game."""
    import json
    del evidence
    if not calibration_cases:
        raise ValueError('Independent calibration cases are required.')
    target = _numeric_fit(fit)
    target['players'] = {side: {key: fit['players'][side][key] for key in ('average_accuracy', 'estimate')}
                         for side in ('White', 'Black')}
    predictions = _cached_fit(json.dumps(target, sort_keys=True),
                              tuple(json.dumps(_numeric_fit(case['fit']), sort_keys=True) for case in calibration_cases))
    output = {}
    for name, pair in predictions.items():
        output[name] = dict(pair)
        account = {}
        for side, estimate in pair.items():
            actual = ratings.get(side)
            if estimate is None:
                account[side] = None
            elif actual is None or not np.isfinite(actual) or not 0 <= actual <= 3200:
                raise ValueError('Account Elo must be finite and lie in[0,3200].')
            else:
                account[side] = (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*actual
        output[name[:-4]+'_account_5pct_all'] = account
    return output
