"""Six fixed, simple shared-curve/account estimators for the fresh comparison.

Only the target game's arithmetic Maia accuracy curve, arithmetic played
accuracy, conditional measurement variance and supplied actual Elo enter these
rules. There is no population corpus, competitive weighting or reference input.
All formulas and constants were declared before scoring commercial labels.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import Args, SharedCurve, fit_pair, prior_density


ARGS = Args()
ACCOUNT_WEIGHT = .10
METHODS = (
    'simple_posterior_mean_account10',
    'simple_common_account_prior',
    'simple_common_account_prior_broad',
    'simple_tangent_tikhonov',
    'simple_inverse_curve_account10',
    'simple_linearized_posterior_account10',
)
SIDES = ('White', 'Black')
DESIGN = {
    'inputs': 'Target shared curve; arithmetic observed accuracy; shared conditional Maia variance; supplied actual Elo.',
    'account_weight': ACCOUNT_WEIGHT,
    'prior_width': 'Standard deviation of the existing fourth-power rating prior; broader control multiplies this SD by sqrt(2).',
    'simple_posterior_mean_account10': 'Gaussian accuracy likelihood with the existing prior; posterior mean; 90% point plus 10% supplied own actual Elo.',
    'simple_common_account_prior': 'Same Gaussian likelihood; multiply existing prior by a Gaussian centered at the mean of supplied actual ratings, with SD equal to the existing prior SD; posterior mean.',
    'simple_common_account_prior_broad': 'Same common account prior, with doubled variance; posterior mean.',
    'simple_tangent_tikhonov': 'Linearize the target curve at the common actual-rating mean m; r=m+[tau^2*s/(variance+tau^2*s^2)]*(accuracy-C(m)); tau is existing prior SD.',
    'simple_inverse_curve_account10': 'Invert the bounded target curve; clip observations beyond its supported accuracy endpoints; 90% inverse point plus 10% supplied own actual Elo.',
    'simple_linearized_posterior_account10': 'Let T(a) be the base posterior mean. Linearize at a=C(m): T(a)+Cov(r,C(r)|a)/variance*(accuracy-a); clip support, then blend 10% own actual Elo.',
    'missing_accounts': 'Direct account blends are omitted per missing side. Common anchors use the mean of available accounts; when neither exists, Gaussian-account priors are omitted and tangent rules expand at the existing prior mean.',
    'uncertainty': 'Point estimators only; no interval or calibration claim.',
    'monotonicity': 'For fixed target context and accounts, every rule is nondecreasing in an individual observed accuracy. All prior and expansion inputs are independent of played accuracy.',
}


def describe():
    return dict(DESIGN)


def prepare(evidence, actual_ratings):
    """Prepare only target-game numerics; metadata and references are ignored."""
    curve = fit_pair(evidence['White'], evidence['Black'], args=ARGS)
    grid = ARGS.grid
    prior = prior_density(grid, args=ARGS)
    prior /= trapezoid(prior, grid)
    prior_mean = float(trapezoid(grid*prior, grid))
    prior_variance = float(trapezoid((grid-prior_mean)**2*prior, grid))
    accounts = {}
    for side in SIDES:
        value = actual_ratings.get(side)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 4000:
                raise ValueError('Supplied actual Elo must be finite in [0, 4000] or missing.')
            accounts[side] = float(np.clip(value, *ARGS.rating_range))
        else:
            accounts[side] = None
    available = [value for value in accounts.values() if value is not None]
    anchor = float(np.mean(available)) if available else prior_mean
    model = SharedCurve(curve['monotone_expected_accuracy'])
    anchor_accuracy = float(model(anchor))
    # The PCHIP/exponential curve is differentiable at its joins. A symmetric
    # finite difference of 0.001 Elo is a numerical derivative, not a fitted scale.
    step = .001
    low, high = max(ARGS.rating_range[0], anchor-step), min(ARGS.rating_range[1], anchor+step)
    slope = max(0., float((model(high)-model(low))/(high-low)))
    return {'curve': curve, 'grid': grid, 'prior': prior, 'prior_mean': prior_mean,
            'prior_variance': prior_variance, 'accounts': accounts, 'has_account': bool(available),
            'anchor': anchor, 'anchor_accuracy': anchor_accuracy, 'slope': slope}


def _posterior(accuracy, prepared, *, account_variance=None):
    grid, prior = prepared['grid'], prepared['prior']
    curve = np.asarray(prepared['curve']['shared_accuracy'])
    variance = max(float(prepared['curve']['likelihood']['accuracy_variance']), 1e-12)
    active = prior > 0
    log_weight = -.5*(accuracy-curve)**2/variance
    log_weight[active] += np.log(prior[active])
    if account_variance is not None and prepared['has_account']:
        log_weight -= .5*(grid-prepared['anchor'])**2/account_variance
    density = np.zeros_like(grid)
    density[active] = np.exp(log_weight[active]-log_weight[active].max())
    density /= trapezoid(density, grid)
    mean = float(trapezoid(grid*density, grid))
    expected_accuracy = float(trapezoid(curve*density, grid))
    covariance = float(trapezoid((grid-mean)*(curve-expected_accuracy)*density, grid))
    return mean, max(0., covariance/variance)


def _blend(value, account):
    value = float(np.clip(value, *ARGS.rating_range))
    return value if account is None else (1-ACCOUNT_WEIGHT)*value+ACCOUNT_WEIGHT*account


def points_at(accuracy, prepared, side):
    """All six fixed formulas for one observation in a fixed game context."""
    if not np.isfinite(accuracy) or not 0 <= accuracy <= 100 or side not in SIDES:
        raise ValueError('A bounded arithmetic accuracy and White/Black side are required.')
    variance = float(prepared['curve']['likelihood']['accuracy_variance'])
    account, anchor = prepared['accounts'][side], prepared['anchor']
    tau_squared, slope = prepared['prior_variance'], prepared['slope']
    ordinary, _ = _posterior(accuracy, prepared)
    common, _ = _posterior(accuracy, prepared, account_variance=tau_squared)
    broad, _ = _posterior(accuracy, prepared, account_variance=2*tau_squared)
    gain = tau_squared*slope/max(variance+tau_squared*slope*slope, 1e-12)
    tangent = anchor+gain*(accuracy-prepared['anchor_accuracy'])
    curve = np.asarray(prepared['curve']['shared_accuracy'])
    equals = np.flatnonzero(np.isclose(curve, accuracy, rtol=0., atol=1e-10))
    inverse = (float(np.mean(prepared['grid'][equals[[0, -1]]])) if equals.size else
               float(np.interp(accuracy, curve, prepared['grid'])))
    expansion, derivative = _posterior(prepared['anchor_accuracy'], prepared)
    linear = expansion+derivative*(accuracy-prepared['anchor_accuracy'])
    return dict(zip(METHODS, (
        _blend(ordinary, account), common, broad, float(np.clip(tangent, *ARGS.rating_range)),
        _blend(inverse, account), _blend(linear, account)), strict=True))


def predict(evidence, actual_ratings):
    """Return method -> White/Black points without engines, labels or writes."""
    prepared = prepare(evidence, actual_ratings)
    results = {name: dict.fromkeys(SIDES) for name in METHODS}
    if not prepared['curve']['identifiable']:
        return results
    for side, player in zip(SIDES, prepared['curve']['players'], strict=True):
        if player['average_accuracy'] is None:
            continue
        for name, value in points_at(player['average_accuracy'], prepared, side).items():
            results[name][side] = value
    return results
