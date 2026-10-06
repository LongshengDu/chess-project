"""Reference-free population accuracy likelihoods with bounded/skewed support.

Beta accuracy and Gamma loss use the same leave-game-out Maia population moments:
population mean curve, between-game variance, and mean conditional game variance.
The target game's conditional sigma and commercial estimates are never consulted.
Beta matches those moments when feasible. Gamma matches loss moments before its
explicit truncation to [0, 100]; truncation can change the resulting mean/variance.

Every observation is scored as a fixed 0.01-accuracy-point interval, rather than
a density at an endpoint. This avoids infinite densities for exactly perfect
accuracy and defines a proper discrete measurement model. The bin width and five
percent account blend are predeclared assumptions, not reference-fitted values.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betainc, betaincc, gammainc, gammaincc

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID, SharedCurve
from tests.analysis.edge_global_quality import _population, _posterior_median


ACCURACY_BIN_WIDTH = .01
PARAMETER_EPSILON = 1e-9
ACCOUNT_WEIGHT = .05
METHODS = (
    'bounded_beta',
    'bounded_beta_account_5pct',
    'bounded_gamma_loss',
    'bounded_gamma_loss_account_5pct',
)


def _validate_moments(mean, variance):
    mean, variance = np.broadcast_arrays(np.asarray(mean, dtype=float), np.asarray(variance, dtype=float))
    if (not np.isfinite(mean).all() or np.any((mean < 0) | (mean > 100))
            or not np.isfinite(variance).all() or np.any(variance < 0)):
        raise ValueError('Finite accuracy means in [0, 100] and nonnegative variances are required.')
    return mean, variance


def beta_parameters(mean, variance):
    """Moment-match Beta; cap variance below p(1-p) and keep both shapes positive.

Variance exceeding the Bernoulli bound cannot describe a bounded [0,1] variable.
It is capped at (1-epsilon)*p(1-p), rather than silently producing negative shapes.
The small mean/variance floors also define finite limiting endpoint calculations.
    """
    mean, variance = _validate_moments(mean, variance)
    probability = np.clip(mean/100., PARAMETER_EPSILON, 1-PARAMETER_EPSILON)
    maximum = probability*(1-probability)
    unit_variance = np.clip(variance/10000., maximum*PARAMETER_EPSILON,
                            maximum*(1-PARAMETER_EPSILON))
    concentration = maximum/unit_variance-1
    return probability*concentration, (1-probability)*concentration


def gamma_parameters(mean, variance):
    """Match mean loss=100-accuracy and loss variance before truncation at100."""
    mean, variance = _validate_moments(mean, variance)
    loss = np.maximum(100.-mean, PARAMETER_EPSILON)
    variance = np.maximum(variance, PARAMETER_EPSILON)
    return loss*loss/variance, variance/loss


def _interval(accuracy):
    accuracy = np.asarray(accuracy, dtype=float)
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 100)):
        raise ValueError('Observed accuracy must be finite and lie in [0, 100].')
    return np.maximum(0., accuracy-ACCURACY_BIN_WIDTH/2), np.minimum(100., accuracy+ACCURACY_BIN_WIDTH/2)


def _stable_mass(lower_cdf, upper_cdf, lower_survival, upper_survival):
    """Use survival differences in the upper tail to avoid cancellation near1."""
    return np.maximum(np.where(lower_cdf < .5, upper_cdf-lower_cdf,
                               lower_survival-upper_survival), 0.)


def beta_accuracy_mass(accuracy, mean, variance):
    """Probability that bounded Beta accuracy falls in the observation bin."""
    lower, upper = _interval(accuracy)
    alpha, beta = beta_parameters(mean, variance)
    lower, upper = lower/100., upper/100.
    return _stable_mass(betainc(alpha, beta, lower), betainc(alpha, beta, upper),
                        betaincc(alpha, beta, lower), betaincc(alpha, beta, upper))


def gamma_accuracy_mass(accuracy, mean, variance):
    """Probability under Gamma loss, normalized after restricting loss to0--100."""
    accuracy_lower, accuracy_upper = _interval(accuracy)
    shape, scale = gamma_parameters(mean, variance)
    lower, upper = (100.-accuracy_upper)/scale, (100.-accuracy_lower)/scale
    normalizer = gammainc(shape, 100./scale)
    mass = _stable_mass(gammainc(shape, lower), gammainc(shape, upper),
                        gammaincc(shape, lower), gammaincc(shape, upper))
    if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0):
        raise ValueError('The truncated Gamma likelihood has no finite mass.')
    return np.clip(mass/normalizer, 0., 1.)


def predict(evidence, fit, ratings, calibration_cases, *, edge_only=True):
    """Return four fixed bounded population candidates; preserve intersections.

The caller must exclude the target game from calibration_cases. Empty calibration
is an error. Missing observations preserve the production missing estimate, and
all nonmissing account ratings used by the fixed blends must lie in [0, 3200].
    """
    del evidence
    grid = ARGS.grid
    _, calibration_variances, pooled, between = _population(calibration_cases, grid)
    variance = between+calibration_variances.mean()
    measured = SharedCurve(fit['diagnostics']['curve']['monotone_expected_accuracy'])(GRID[[0, -1]])
    outputs = {name: {} for name in METHODS}
    for side in ('White', 'Black'):
        player = fit['players'][side]
        observed, baseline = player['average_accuracy'], player['estimate']
        if observed is None or baseline is None or (edge_only and measured[0] <= observed <= measured[1]):
            for values in outputs.values():
                values[side] = baseline
            continue
        account = ratings.get(side)
        if account is None or not np.isfinite(account) or not 0 <= account <= 3200:
            raise ValueError('Account Elo must be finite and lie in [0, 3200].')
        for name, likelihood in (('bounded_beta', beta_accuracy_mass),
                                 ('bounded_gamma_loss', gamma_accuracy_mass)):
            mass = likelihood(observed, pooled, variance)
            logs = np.log(np.maximum(mass, 1e-300))
            estimate = _posterior_median(logs, grid)
            outputs[name][side] = estimate
            outputs[name+'_account_5pct'][side] = (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*account
    return outputs
