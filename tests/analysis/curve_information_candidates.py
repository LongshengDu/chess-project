"""A fixed accuracy-coordinate prior for the original shared accuracy curve.

For the same original curve C(r), observed arithmetic accuracy A and constant
pooled Gaussian variance v, use

    p(r | A) proportional to taper(r)*C'(r)*exp(-(A-C(r))**2/(2*v)).

The C'(r) Jacobian is the Jeffreys factor for a Gaussian location model with
constant variance. Multiplying by the existing taper is an extra prior choice,
so the complete prior is a tapered uniform measure in modeled accuracy, not
an assertion of pure Jeffreys inference. It gives little rating mass to flat
regions of the curve instead of assigning them large mass merely because they
cover many Elo values. The target curve, variance and tails are unchanged.

The posterior mean receives the existing5% common account-mean contribution.
There are no fit parameters, reference inputs, population curves or edge rules.
The derivative is analytic for both PCHIP interpolation and exponential tails.
A curve with no positive derivative on prior support gives no estimate.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import GRID, SharedCurve, prior_density
from analysis.player_rating.uncertainty_measurement import ARGS, measure


METHOD = 'curve_information_prior'
SIDES = ('White', 'Black')
ACCOUNT_WEIGHT = .05


def describe():
    return {METHOD: 'Original shared Gaussian accuracy likelihood with tapered accuracy-coordinate prior proportional to taper(r)*C_prime(r); posterior mean plus5% common account anchor.'}


def curve_derivative(model, ratings):
    """Analytic nonnegative derivative of the existing bounded shared curve."""
    ratings = np.asarray(ratings, dtype=float)
    native = np.maximum(0., model.measured.derivative()(np.clip(ratings, GRID[0], GRID[-1])))
    left, right = model.accuracies[[0, -1]]
    lower_rate = model.slopes[0]/max(left, 1e-12)
    upper_rate = model.slopes[1]/max(100.-right, 1e-12)
    lower_exponent = lower_rate*(ratings-GRID[0])
    upper_exponent = -upper_rate*(ratings-GRID[-1])
    lower = lower_rate*left*np.exp(np.clip(lower_exponent, -700., 0.))
    upper = upper_rate*(100.-right)*np.exp(np.clip(upper_exponent, -700., 0.))
    # SharedCurve clips exponential arguments at-700 for numerical safety;
    # clipped tails are locally constant, including saturated endpoint cases.
    lower = np.where(lower_exponent < -700., 0., lower)
    upper = np.where(upper_exponent < -700., 0., upper)
    return np.where(ratings < GRID[0], lower, np.where(ratings > GRID[-1], upper, native))


def prepare(evidence, actual_ratings=None):
    """Prepare the target curve and its information-prior density once."""
    measurement = measure(evidence)
    curve = measurement['curve']
    grid = ARGS.grid
    model = SharedCurve(curve['monotone_expected_accuracy'])
    derivative = curve_derivative(model, grid)
    unnormalized = prior_density(grid, args=ARGS)*derivative
    normalizer = float(trapezoid(unnormalized, grid))
    prior = unnormalized/normalizer if normalizer > 0 else np.zeros_like(grid)
    return {'curve': curve, 'grid': grid, 'prior': prior, 'derivative': derivative,
            'identifiable': bool(curve['identifiable']) and normalizer > 0,
            'observed': {side: moment['accuracy'] if moment else None
                         for side, moment in measurement['sides'].items()},
            'diagnostics': {'prior': 'tapered_uniform_modeled_accuracy',
                            'prior_normalizer': normalizer,
                            'accuracy_variance': curve['likelihood']['accuracy_variance']}}


def _anchor(actual_ratings):
    values = [actual_ratings.get(side) for side in SIDES]
    available = [value for value in values if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or not 0 <= value <= 4000 for value in available):
        raise ValueError('Actual ratings must be finite numbers in [0, 4000], or absent.')
    return float(np.mean(np.clip(available, *ARGS.rating_range))) if available else None


def point(accuracy, prepared, actual_ratings):
    """One fixed-context accuracy-to-rating decision; None if no curve signal."""
    if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError('Arithmetic accuracy must lie in [0, 100].')
    anchor = _anchor(actual_ratings)
    if not prepared['identifiable']:
        return None
    grid, prior = prepared['grid'], prepared['prior']
    active = prior > 0
    curve = np.asarray(prepared['curve']['shared_accuracy'])
    variance = max(float(prepared['curve']['likelihood']['accuracy_variance']), 1e-12)
    log_mass = -.5*(accuracy-curve)**2/variance
    log_mass[active] += np.log(prior[active])
    density = np.zeros_like(grid)
    density[active] = np.exp(log_mass[active]-log_mass[active].max())
    density /= trapezoid(density, grid)
    estimate = float(trapezoid(grid*density, grid))
    return estimate if anchor is None else (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*anchor


def points_at(accuracy, prepared, actual_ratings):
    return {METHOD: point(accuracy, prepared, actual_ratings)}


def predict_prepared(prepared, actual_ratings):
    _anchor(actual_ratings)
    return {METHOD: {side: point(accuracy, prepared, actual_ratings) if accuracy is not None else None
                     for side, accuracy in prepared['observed'].items()}}


def predict(evidence, actual_ratings):
    return predict_prepared(prepare(evidence, actual_ratings), actual_ratings)
