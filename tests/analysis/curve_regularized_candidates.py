"""One prespecified context-shrunk curve with a robust common-account prior.

Reuse the discrepancy model M(r)=w*C(r)+(1-w)*P(r), w=tau2/(tau2+v),
with accuracy variance V=v+tau2*v/(tau2+v). C is the local arithmetic
shared curve, P the frozen population mean curve, v the pooled conditional
Maia mean variance, and tau2 the native-grid mean between-context variance.
Both players have the same Gaussian accuracy likelihood truncated to0--100.
Their common prior is the existing tapered rating prior multiplied by a Cauchy
density centered on the mean supplied account rating, scale400/log(10).

No constants are tuned or new observations added to the frozen population.
This is the already-declared curve shrinkage combined with the already-declared
robust account prior. It has one monotone accuracy-to-rating map for each fixed
game context and account pair, with no edge switches or pair projection.
The between-context spread remains an exploratory proxy for prediction error;
the account-scale interpretation is not a calibrated single-game rating prior.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import ndtr

from analysis.player_rating.bayesian_shared_curve import prior_density
from analysis.player_rating.uncertainty_measurement import ARGS
from tests.analysis.curve_discrepancy_candidates import prepare as prepare_context


METHOD = 'curve_regularized_cauchy'
SIDES = ('White', 'Black')
CAUCHY_SCALE = 400./math.log(10.)


def describe():
    return {METHOD: 'Variance-based local/population shared-curve shrinkage; one bounded Gaussian accuracy likelihood and common Cauchy account prior of scale400/log(10); posterior mean.'}


def prepare(evidence, actual_ratings=None):
    """Prepare the unchanged reference-free curve shrinkage once per game."""
    context = prepare_context(evidence)
    model = context['models'].get('discrepancy_curve_shrinkage')
    return {'identifiable': context['identifiable'], 'observed': context['observed'],
            'model': model, 'grid': ARGS.grid, 'prior': prior_density(ARGS.grid, args=ARGS),
            'diagnostics': {key: value for key, value in context.items()
                            if key not in ('models', 'observed', 'identifiable')}}


def _anchor(actual_ratings):
    supplied = [actual_ratings.get(side) for side in SIDES]
    values = [value for value in supplied if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or not 0 <= value <= 4000 for value in values):
        raise ValueError('Actual ratings must be finite numbers in [0, 4000], or absent.')
    return float(np.mean(np.clip(values, *ARGS.rating_range))) if values else None


def point(accuracy, prepared, actual_ratings):
    """Posterior mean under one common, fixed-context monotone rating map."""
    if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError('Arithmetic accuracy must lie in [0, 100].')
    anchor = _anchor(actual_ratings)
    curve, variance = prepared['model']['curve'], max(prepared['model']['variance'], 1e-9)
    grid, prior = prepared['grid'], prepared['prior']
    active = prior > 0
    sigma = np.sqrt(variance)
    normalizer = ndtr((100.-curve)/sigma)-ndtr(-curve/sigma)
    log_density = -.5*(accuracy-curve)**2/variance-np.log(normalizer)
    log_density[active] += np.log(prior[active])
    if anchor is not None:
        log_density -= np.log1p(((grid-anchor)/CAUCHY_SCALE)**2)
    density = np.zeros_like(grid)
    density[active] = np.exp(log_density[active]-log_density[active].max())
    density /= trapezoid(density, grid)
    return float(trapezoid(grid*density, grid))


def predict_prepared(prepared, actual_ratings):
    _anchor(actual_ratings)
    points = dict.fromkeys(SIDES)
    if prepared['identifiable']:
        for side, accuracy in prepared['observed'].items():
            if accuracy is not None:
                points[side] = point(accuracy, prepared, actual_ratings)
    return {METHOD: points}


def predict(evidence, actual_ratings):
    return predict_prepared(prepare(evidence, actual_ratings), actual_ratings)
