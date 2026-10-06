"""Two fixed Laplace-noise controls for the arithmetic shared curve.

The target-only likelihood is exp(-abs(A-C(r))/b), with b=sqrt(v/2), so
its variance equals the existing pooled conditional Maia accuracy variance v.
Only the noise distribution changes; no residual scale is fitted. Both players
use the same curve, variance and fourth-power tapered rating prior. One control
multiplies that prior by a common-account Cauchy density with scale400/log(10).
The other retains the original prior and blends5% of the common account mean
into the posterior mean afterwards. Neither uses population curves or labels.

Laplace location noise has monotone likelihood ratio, preserving shared-curve
ordering without projection. Beyond every modeled mean the common accuracy
term cancels, so the estimate saturates instead of claiming more resolution.
The Cauchy-account control can be sensitive to the account location; the5%
decision blend moves by at most5Elo when one of two accounts changes by200.
These are working likelihoods and point estimates, not calibrated intervals.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import prior_density
from analysis.player_rating.uncertainty_measurement import ARGS, measure


SIDES = ('White', 'Black')
CAUCHY_SCALE = 400./math.log(10.)
ACCOUNT_WEIGHT = .05
METHODS = ('curve_laplace_cauchy_account', 'curve_laplace_common_account5')


def describe():
    return {
        METHODS[0]: 'Target arithmetic Laplace likelihood with b=sqrt(v/2); common Cauchy account prior of scale400/log(10); posterior mean.',
        METHODS[1]: 'Target arithmetic Laplace likelihood with b=sqrt(v/2); existing rating prior;95% posterior mean plus5% common account mean.',
    }


def prepare(evidence, actual_ratings=None):
    """Cache target-game arithmetic measurements independently of accounts."""
    measurement = measure(evidence)
    return {'curve': measurement['curve'], 'grid': ARGS.grid,
            'prior': prior_density(ARGS.grid, args=ARGS),
            'observed': {side: moment['accuracy'] if moment else None
                         for side, moment in measurement['sides'].items()}}


def _anchor(actual_ratings):
    supplied = [actual_ratings.get(side) for side in SIDES]
    values = [value for value in supplied if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or not 0 <= value <= 4000 for value in values):
        raise ValueError('Actual ratings must be finite numbers in [0, 4000], or absent.')
    return float(np.mean(np.clip(values, *ARGS.rating_range))) if values else None


def points_at(accuracy, prepared, actual_ratings):
    """Evaluate both controls for one observation in a fixed game context."""
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError('Arithmetic accuracy must lie in [0, 100].')
    anchor = _anchor(actual_ratings)
    grid, prior = prepared['grid'], prepared['prior']
    curve = np.asarray(prepared['curve']['shared_accuracy'])
    scale = np.sqrt(max(float(prepared['curve']['likelihood']['accuracy_variance']), 1e-12)/2)
    active = prior > 0
    base = -abs(accuracy-curve)/scale
    base[active] += np.log(prior[active])
    result = {}
    for method in METHODS:
        log_mass = base.copy()
        if anchor is not None and method == METHODS[0]:
            log_mass -= np.log1p(((grid-anchor)/CAUCHY_SCALE)**2)
        density = np.zeros_like(grid)
        density[active] = np.exp(log_mass[active]-log_mass[active].max())
        density /= trapezoid(density, grid)
        point = float(trapezoid(grid*density, grid))
        if anchor is not None and method == METHODS[1]:
            point = (1-ACCOUNT_WEIGHT)*point+ACCOUNT_WEIGHT*anchor
        result[method] = point
    return result


def predict_prepared(prepared, actual_ratings):
    _anchor(actual_ratings)
    result = {method: dict.fromkeys(SIDES) for method in METHODS}
    if not prepared['curve']['identifiable']:
        return result
    for side, accuracy in prepared['observed'].items():
        if accuracy is not None:
            for method, value in points_at(accuracy, prepared, actual_ratings).items():
                result[method][side] = value
    return result


def predict(evidence, actual_ratings):
    return predict_prepared(prepare(evidence, actual_ratings), actual_ratings)
