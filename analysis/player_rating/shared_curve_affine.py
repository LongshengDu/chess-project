"""Current-game affine accuracy-to-rating estimation without a population corpus.

Both players use the current game's shared arithmetic Maia curve, conditional
accuracy variance and a common account-centered prior. Prior moments define the
minimum squared-error affine predictor. No other games, fitted reference labels
or calibration assets enter. Native Lichess Blitz calculations are converted by
the public rating service only after selecting the point estimate.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import prior_weights
from analysis.player_rating.interface import PlayerRating
from analysis.player_rating.uncertainty_measurement import ARGS as CURVE_ARGS, measure


METHOD = __name__.rsplit('.', 1)[-1]
NAME = 'Shared-curve affine fit'
VERSION = 1
SIDES = ('White', 'Black')


def _parameters():
    # The shared measurement's posterior display interval is not used here.
    return {key: value for key, value in asdict(CURVE_ARGS).items() if key != 'central_interval'}


def _account_anchor(actual_ratings):
    """Average available account inputs after clipping to the native support."""
    if not isinstance(actual_ratings, dict) or set(actual_ratings)-set(SIDES):
        raise ValueError('Actual ratings must be a mapping using White and Black keys.')
    used = {}
    for side in SIDES:
        value = actual_ratings.get(side)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not 0 <= value <= 4000):
            raise ValueError('Actual ratings must be finite values in [0, 4000], or absent.')
        used[side] = float(np.clip(value, *CURVE_ARGS.rating_range)) if value is not None else None
    available = [value for value in used.values() if value is not None]
    return (float(np.mean(available)) if available else None), used


def translated_prior(anchor):
    """Translate the fourth-power taper, then normalize on fixed native support."""
    if anchor is not None and (isinstance(anchor, bool) or not isinstance(anchor, (int, float))
                              or not math.isfinite(anchor)
                              or not CURVE_ARGS.rating_range[0] <= anchor <= CURVE_ARGS.rating_range[1]):
        raise ValueError('The common account anchor must be inside the native rating support.')
    grid = CURVE_ARGS.grid
    shift = anchor-float(np.mean(CURVE_ARGS.prior_range)) if anchor is not None else 0.
    raw = prior_weights(grid-shift, args=CURVE_ARGS)
    normalizer = float(trapezoid(raw, grid))
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('The translated prior has no finite mass.')
    density = raw/normalizer
    mean = float(trapezoid(grid*density, grid))
    return {'prior_weights': raw, 'prior_density': density, 'prior_normalizer': normalizer,
            'prior_mean': mean, 'prior_sd': float(np.sqrt(trapezoid((grid-mean)**2*density, grid))),
            'account_anchor': anchor, 'prior_shift': shift}


def affine_moments(accuracy_curve, measurement_variance, prior):
    """Integrate the common affine map using only current-game accuracy moments."""
    grid = CURVE_ARGS.grid
    curve = np.asarray(accuracy_curve, dtype=float)
    if (curve.shape != grid.shape or not np.isfinite(curve).all()
            or np.any((curve < 0) | (curve > 100))
            or isinstance(measurement_variance, bool) or not np.isfinite(measurement_variance)
            or measurement_variance < 0):
        raise ValueError('A bounded finite accuracy curve and nonnegative conditional variance are required.')
    density, rating_mean = prior['prior_density'], prior['prior_mean']
    accuracy_mean = float(trapezoid(curve*density, grid))
    covariance = float(trapezoid((grid-rating_mean)*(curve-accuracy_mean)*density, grid))
    curve_variance = float(trapezoid((curve-accuracy_mean)**2*density, grid))
    denominator = curve_variance+measurement_variance
    slope = max(0., covariance)/denominator if denominator > 0 else 0.
    return {'accuracy_mean': accuracy_mean, 'accuracy_variance': denominator,
            'curve_accuracy_variance': curve_variance,
            'rating_accuracy_covariance': covariance, 'affine_slope': slope,
            'affine_intercept': rating_mean-slope*accuracy_mean}


def calculate(evidence, actual_ratings=None):
    """Pure current-game fit; no asset loading, external games or fitted labels.

    The conditional variance reflects independent hypothetical Maia choices at
    the reached positions. It is not observed played-move variance, a measured
    model-error variance or a calibrated single-game rating uncertainty.
    """
    actual_ratings = {} if actual_ratings is None else actual_ratings
    anchor, used = _account_anchor(actual_ratings)
    measurement = measure(evidence)
    curve = measurement['curve']
    available = bool(curve['identifiable'])
    prior, base = translated_prior(anchor), translated_prior(None)
    variance = float(curve['likelihood']['accuracy_variance'])
    model = {'grid': CURVE_ARGS.grid.tolist(), 'accuracy_curve': curve['shared_accuracy'],
             'measurement_variance': variance,
             'base_prior_weights': base['prior_weights'].tolist(),
             'base_prior_density': base['prior_density'].tolist()}
    affine = {key: value for key, value in prior.items() if key not in ('prior_weights', 'prior_density')}
    affine.update(dict.fromkeys(('accuracy_mean', 'accuracy_variance', 'curve_accuracy_variance',
                                'rating_accuracy_covariance', 'affine_slope', 'affine_intercept')))
    curve.update(prior_weights=prior['prior_weights'].tolist(), prior_density=prior['prior_density'].tolist())
    curve['likelihood']['kind'] = 'conditional_accuracy_moments'
    # The measurement helper's Bayesian point gap is not this affine decision.
    curve.pop('white_minus_black', None)
    if available:
        affine.update(affine_moments(curve['shared_accuracy'], variance, prior))
    players, components = {}, {}
    for side in SIDES:
        moment = measurement['sides'][side]
        observed = moment['accuracy'] if moment else None
        adjustment, unbounded, point = None, None, None
        if available and observed is not None:
            adjustment = affine['affine_slope']*(observed-affine['accuracy_mean'])
            unbounded = affine['prior_mean']+adjustment
            point = float(np.clip(unbounded, *CURVE_ARGS.rating_range))
        components[side] = {'observed_accuracy': observed, 'accuracy_adjustment': adjustment,
                            'unbounded_estimate': unbounded, 'unrounded_estimate': point,
                            'actual_rating': actual_ratings.get(side), 'account_rating_used': used[side]}
        players[side] = {'estimate': math.floor(point+.5) if point is not None else None,
                         'unrounded_estimate': point, 'uncertainty': None, 'interval': None,
                         'moves_used': moment['moves'] if moment else 0, 'average_accuracy': observed,
                         'identifiable': point is not None,
                         'at_rating_limit': point in CURVE_ARGS.rating_range if point is not None else False,
                         'method': METHOD}
    shift = prior['prior_shift']
    return {'players': players, 'name': NAME, 'method_id': METHOD, 'parameters': _parameters(),
            'central_interval': None, 'rating_range': list(CURVE_ARGS.rating_range),
            'point_estimator': 'shared_curve_affine_action',
            'prior': {'kind': 'translated_symmetric_fourth_power',
                      'minimum': CURVE_ARGS.prior_range[0]+shift, 'maximum': CURVE_ARGS.prior_range[1]+shift,
                      'flat_range': [value+shift for value in CURVE_ARGS.flat_prior_range],
                      'truncated_to': list(CURVE_ARGS.rating_range), 'account_anchor': anchor, 'shift': shift},
            'interval_scope': 'Point-only minimum-MSE affine decision under a common translated account prior; no calibrated uncertainty interval or posterior density for the final estimate is available.',
            'method': 'Current-game full legal-move Maia policies; one shared arithmetic accuracy curve and conditional variance; fourth-power prior translated to mean available account Elo; common minimum-MSE affine predictor clipped to native rating support. No other games or population asset are used.',
            'diagnostics': {'kind': METHOD, 'curve': curve, 'model': model, 'affine': affine, 'components': components},
            'account_ratings_used': bool(available and anchor is not None)}


class Rating(PlayerRating):
    """Filename-selected current-game estimator, with no external data assets."""

    name = NAME
    version = VERSION

    @property
    def parameters(self):
        return _parameters()

    def fit(self, evidence):
        return calculate(evidence, {side: evidence[side].get('actual_rating') for side in SIDES})
