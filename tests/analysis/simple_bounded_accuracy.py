"""One fixed-concentration target-Beta accuracy model, with no model mixture.

The target shared curve supplies the Beta mean. One concentration per game
matches the existing shared conditional variance at the native curve's mean.
An identical prior and concentration for both sides preserve accuracy ordering.
Only the final ten-percent common account anchor uses supplied actual ratings.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import betaln, betainc, betaincc, logsumexp

from analysis.player_rating.bayesian_shared_curve import Args, fit_pair, prior_density


ARGS = Args()
METHOD = 'simple_target_beta_common_account10'
ACCOUNT_WEIGHT = .10
ACCURACY_BIN_WIDTH = .01
EPSILON = 1e-9
SIDES = ('White', 'Black')
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(32)


def describe():
    return {'method': METHOD, 'account_weight': ACCOUNT_WEIGHT, 'accuracy_bin_width': ACCURACY_BIN_WIDTH,
            'inputs': 'Target shared arithmetic accuracy curve, arithmetic observed accuracy, shared conditional Maia variance and supplied actual ratings.',
            'likelihood': 'Beta with mean C(r)/100 and common concentration k=mu_bar*(1-mu_bar)/(shared_variance/10000)-1; mu_bar is the mean of the 21 native shared-curve knots.',
            'prior': 'Unchanged fourth-power rating prior; posterior mean.',
            'account': '90% posterior mean plus 10% mean of available actual ratings, identically for both players; omit anchor when neither rating is supplied.',
            'monotonicity': 'Common Beta concentration gives a monotone likelihood ratio in observed accuracy; the identical account anchor preserves ordering.',
            'numerics': 'Fixed 0.01-point observation bin; exact incomplete-Beta probabilities with log-space 32-node quadrature only where the probability underflows.',
            'scope': 'One fixed rule; no population calibration, competitive weighting, commercial labels, mixture, fitted corrections or reported credible interval.'}


def prepare(evidence):
    curve = fit_pair(evidence['White'], evidence['Black'], args=ARGS)
    mean = np.clip(np.asarray(curve['shared_accuracy'])/100., EPSILON, 1-EPSILON)
    native_mean = float(np.mean(curve['monotone_expected_accuracy'])/100.)
    native_mean = float(np.clip(native_mean, EPSILON, 1-EPSILON))
    variance = float(curve['likelihood']['accuracy_variance'])/10000.
    concentration = max(native_mean*(1-native_mean)/max(variance, EPSILON)-1., EPSILON)
    return {'curve': curve, 'alpha': concentration*mean, 'beta': concentration*(1-mean),
            'concentration': concentration, 'native_mean': native_mean, 'shared_unit_variance': variance}


def _log_bin_mass(accuracy, alpha, beta):
    lower = max(0., accuracy-ACCURACY_BIN_WIDTH/2)/100.
    upper = min(100., accuracy+ACCURACY_BIN_WIDTH/2)/100.
    lower_cdf, upper_cdf = betainc(alpha, beta, lower), betainc(alpha, beta, upper)
    mass = np.maximum(np.where(lower_cdf < .5, upper_cdf-lower_cdf,
                               betaincc(alpha, beta, lower)-betaincc(alpha, beta, upper)), 0.)
    valid = np.isfinite(mass) & (mass > 0)
    logs = np.empty_like(mass)
    logs[valid] = np.log(mass[valid])
    if not np.all(valid):
        # Positive interior quadrature nodes handle exact 0/100 observations
        # without inventing a probability floor or replacing a tiny likelihood
        # by the prior. Ordinary played accuracies use exact CDF differences.
        width = (upper-lower)/2
        points = (lower+upper)/2+width*_NODES[:, None]
        a, b = alpha[~valid], beta[~valid]
        log_density = (a-1)*np.log(points)+(b-1)*np.log1p(-points)-betaln(a, b)
        logs[~valid] = logsumexp(log_density+np.log(_WEIGHTS[:, None]*width), axis=0)
    if not np.isfinite(logs).all():
        raise ValueError('The target-Beta likelihood could not be evaluated finitely.')
    return logs


def posterior_mean(accuracy, prepared):
    if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError('Arithmetic accuracy must be finite within [0, 100].')
    grid = ARGS.grid
    prior = prior_density(grid, args=ARGS)
    active = prior > 0
    logs = _log_bin_mass(accuracy, prepared['alpha'], prepared['beta'])
    logs[active] += np.log(prior[active])
    density = np.zeros_like(grid)
    density[active] = np.exp(logs[active]-logs[active].max())
    density /= trapezoid(density, grid)
    return float(trapezoid(grid*density, grid))


def predict(evidence, actual_ratings):
    prepared = prepare(evidence)
    points = dict.fromkeys(SIDES)
    accounts = []
    for side in SIDES:
        value = actual_ratings.get(side)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 4000:
                raise ValueError('Actual Elo must be finite within [0, 4000] or missing.')
            accounts.append(float(np.clip(value, *ARGS.rating_range)))
    anchor = float(np.mean(accounts)) if accounts else None
    if prepared['curve']['identifiable']:
        for side, player in zip(SIDES, prepared['curve']['players'], strict=True):
            if player['average_accuracy'] is not None:
                point = posterior_mean(player['average_accuracy'], prepared)
                points[side] = point if anchor is None else (1-ACCOUNT_WEIGHT)*point+ACCOUNT_WEIGHT*anchor
    return {METHOD: points}
