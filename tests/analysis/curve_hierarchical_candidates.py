"""Two fixed hierarchical affine estimators retaining the target shared curve.

Before reference scoring, declare M(r)=w*C(r)+(1-w)*P(r), where C is this
game's arithmetic Maia curve and P the frozen context-population mean curve.
The context-discrepancy proxy tau² is the average sample variance of calibration
native knots across retained games. With target mean measurement variance v,
w=tau²/(tau²+v) and residual noise s²=v+tau²*v/(tau²+v).

For either common rating prior pi, the best affine squared-error predictor is
E[R]+Cov(R,M(R))/(Var(M(R))+s²)*(A-E[M(R)]). Both players share this map.
The only two declared priors are the existing fourth-power shape translated
to the mean supplied account rating, and the unshifted fourth-power shape times
a Gaussian centered there with fixed SD400*pi/(sqrt(3)*log(10)), about315Elo.
Using the conventional Elo-logistic noise SD as a Gaussian prior SD is an
explicit modeling assumption, not a calibrated single-game strength interval.

Context spread is not measured Maia prediction error: shrinkage can remove real
positional difficulty. Account priors need not satisfy a small rating-sensitivity
bound. Clipping affine actions to0--3200 preserves weak order but can produce
ties at boundaries. No commercial references or tuned coefficients enter.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid

from analysis.player_rating.bayesian_shared_curve import SharedCurve, prior_weights
from analysis.player_rating.calibration import load_calibration
from analysis.player_rating.uncertainty_measurement import ARGS, measure


SIDES = ('White', 'Black')
ELO_LOGISTIC_SD = 400.*math.pi/(math.sqrt(3.)*math.log(10.))
METHODS = ('hierarchical_affine_translated_prior', 'hierarchical_affine_gaussian315_prior')


def describe():
    return {
        METHODS[0]: 'Variance-based empirical-Bayes shrinkage of target/population arithmetic curves; minimum-MSE affine action under the existing prior translated to common account Elo.',
        METHODS[1]: 'Same retained-target hierarchical curve and affine action; fixed Gaussian common-account prior SD400*pi/(sqrt(3)*log(10)) times existing taper. This SD is a modeling assumption, not fitted to reference ratings.',
    }


def shrinkage(local, population, between_variance, measurement_variance):
    """Combine uncertain context means with a fixed scalar residual-noise rule."""
    local, population = np.asarray(local, dtype=float), np.asarray(population, dtype=float)
    if (local.ndim != 1 or local.shape != population.shape or not len(local)
            or not np.isfinite([local, population]).all()
            or not np.isfinite([between_variance, measurement_variance]).all()
            or min(between_variance, measurement_variance) < 0):
        raise ValueError('Finite matching curves and nonnegative variances are required.')
    denominator = between_variance+measurement_variance
    weight = between_variance/denominator if denominator else 1.
    noise = measurement_variance+(between_variance*measurement_variance/denominator if denominator else 0.)
    return weight*local+(1-weight)*population, float(weight), float(noise)


def prepare(evidence, actuals=None):
    """Prepare model-only context data once, without consuming actual/reference Elo."""
    measurement = measure(evidence)
    curve = measurement['curve']
    prepared = {'identifiable': bool(curve['identifiable']),
                'observed': {side: moment['accuracy'] if moment else None
                             for side, moment in measurement['sides'].items()}}
    if not prepared['identifiable']:
        return prepared
    corpus = load_calibration().for_evidence(evidence)
    native = np.asarray([record.arithmetic.knots for record in corpus.records], dtype=float)
    grid = ARGS.grid
    population = np.mean([SharedCurve(knots)(grid) for knots in native], axis=0)
    between = float(np.var(native, axis=0, ddof=int(len(native) > 1)).mean())
    variance = float(curve['likelihood']['accuracy_variance'])
    local = np.asarray(curve['shared_accuracy'], dtype=float)
    model, weight, noise = shrinkage(local, population, between, variance)
    for values in (grid, local, population, model):
        values.setflags(write=False)
    prepared.update(grid=grid, local_curve=local, population_curve=population, model_curve=model,
                    measurement_variance=variance, between_context_variance=between,
                    residual_variance=noise, local_weight=weight,
                    contexts_used=len(corpus.records), contexts_excluded=corpus.excluded_count,
                    calibration_hash=corpus.content_hash)
    return prepared


def _anchor(actuals):
    if actuals is None:
        return None
    if not isinstance(actuals, dict) or set(actuals)-set(SIDES):
        raise ValueError('Account ratings must use White and Black keys.')
    available = []
    for side in SIDES:
        value = actuals.get(side)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 4000:
            raise ValueError('Account ratings must be finite values in [0, 4000], or absent.')
        available.append(float(np.clip(value, *ARGS.rating_range)))
    return float(np.mean(available)) if available else None


def affine_model(prepared, anchor, method):
    """Integrate prior moments, retaining the common prior for graph inspection."""
    if method not in METHODS:
        raise ValueError('Unknown predeclared hierarchical method.')
    grid, curve = prepared['grid'], prepared['model_curve']
    shift = anchor-float(np.mean(ARGS.prior_range)) if anchor is not None else 0.
    raw_prior = prior_weights(grid-shift if method == METHODS[0] else grid, args=ARGS)
    if method == METHODS[1] and anchor is not None:
        raw_prior *= np.exp(-.5*((grid-anchor)/ELO_LOGISTIC_SD)**2)
    normalizer = float(trapezoid(raw_prior, grid))
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('The declared common prior has no finite mass.')
    density = raw_prior/normalizer
    rating_mean = float(trapezoid(grid*density, grid))
    accuracy_mean = float(trapezoid(curve*density, grid))
    covariance = float(trapezoid((grid-rating_mean)*(curve-accuracy_mean)*density, grid))
    variance = float(trapezoid((curve-accuracy_mean)**2*density, grid))+prepared['residual_variance']
    coefficient = max(0., covariance)/variance if variance > 0 else 0.
    return {'prior_density': density, 'prior_normalizer': normalizer,
            'prior_mean': rating_mean, 'prior_sd': float(np.sqrt(trapezoid((grid-rating_mean)**2*density, grid))),
            'accuracy_mean': accuracy_mean, 'accuracy_variance': variance,
            'rating_accuracy_covariance': covariance, 'affine_slope': coefficient,
            'account_anchor': anchor, 'gaussian_sd': ELO_LOGISTIC_SD if method == METHODS[1] and anchor is not None else None}


def predict_prepared(prepared, actuals):
    """Return two common monotone affine maps; no player-specific correction."""
    anchor = _anchor(actuals)
    result = {method: dict.fromkeys(SIDES) for method in METHODS}
    if not prepared['identifiable']:
        return result
    for method in METHODS:
        model = affine_model(prepared, anchor, method)
        for side, accuracy in prepared['observed'].items():
            if accuracy is None:
                continue
            point = model['prior_mean']+model['affine_slope']*(accuracy-model['accuracy_mean'])
            result[method][side] = float(np.clip(point, *ARGS.rating_range))
    return result


def diagnostics(prepared, actuals):
    """JSON-safe curves, priors and exact affine moments for later figures."""
    anchor = _anchor(actuals)
    if not prepared['identifiable']:
        return {'identifiable': False, 'observed': dict(prepared['observed']), 'account_anchor': anchor}
    models = {}
    for method in METHODS:
        model = affine_model(prepared, anchor, method)
        models[method] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}
    return {'identifiable': True, 'observed': dict(prepared['observed']),
            **{key: prepared[key].tolist() for key in ('grid', 'local_curve', 'population_curve', 'model_curve')},
            **{key: prepared[key] for key in ('measurement_variance', 'between_context_variance', 'residual_variance',
                                             'local_weight', 'contexts_used', 'contexts_excluded', 'calibration_hash')},
            'account_anchor': anchor, 'methods': models}


def predict(evidence, actuals):
    return predict_prepared(prepare(evidence), actuals)
