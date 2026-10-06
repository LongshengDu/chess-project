"""Three predeclared population-moment rating rules, with no reference fitting.

Let R have the existing smooth fourth-power rating prior. Frozen, anonymous
Maia contexts provide m(r)=E[A|R=r] and the between-context variance b(r).
The target game's shared arithmetic mean variance s² replaces the calibration
games' mean sampling variance, avoiding counting sampling noise twice.

The affine estimator is E[R] + Cov(R,m(R))/Var(A) * (a-E[A]), where
Var(A)=Var(m(R))+E[b(R)]+s². This is the minimum-MSE affine predictor under
that synthetic joint model; it is not a posterior mean under arbitrary shapes.
Two fixed variants use the existing prior with a 5% common account anchor, or
translate the prior by the common account rating minus its original mean.

The third rule maps the moment-matched normal marginal percentile of accuracy
to the translated rating-prior percentile. This is a monotone distributional
transport diagnostic, not a conditional skill posterior or a calibrated Elo
conversion. Marginal rank preservation alone need not predict individual skill.

All maps are common to both players and nondecreasing; there is no pair-order
projection, edge gate, commercial coefficient fit, or family parameter search.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import ndtr

from analysis.player_rating.bayesian_shared_curve import prior_weights
from analysis.player_rating.calibration import load_calibration
from analysis.player_rating.uncertainty_measurement import ARGS, measure


SIDES = ('White', 'Black')
ACCOUNT_WEIGHT = .05


def describe():
    return {
        'population_affine_common_account5': 'Frozen Maia population minimum-MSE affine predictor; target arithmetic mean variance replaces calibration sampling variance; fixed existing rating prior plus 5% common account anchor.',
        'population_affine_account_prior': 'Same population affine predictor with the existing prior translated to the common supplied account rating; no final account blend.',
        'population_percentile_account_prior': 'Moment-normal accuracy marginal percentile transported to the account-centered rating-prior percentile; monotone rank diagnostic, not a posterior.',
    }


@dataclass(frozen=True)
class Prepared:
    grid: np.ndarray
    population_mean: np.ndarray
    between_variance: np.ndarray
    target_variance: float
    accuracies: tuple[float | None, float | None]
    identifiable: bool
    calibration_hash: str
    contexts_used: int


def prepare(evidence, actuals=None):
    """Prepare label-free moments once; sensitivity runs can reuse them."""
    measurement = measure(evidence)
    curve = measurement['curve']
    corpus = load_calibration().for_evidence(evidence)
    grid = ARGS.grid
    mean, variance = corpus.population(grid, 'arithmetic')
    within = np.mean([record.arithmetic.variance for record in corpus.records])
    between = np.maximum(0., variance-within)
    for values in (grid, mean, between):
        values.setflags(write=False)
    return Prepared(grid, mean, between, float(curve['likelihood']['accuracy_variance']),
                    tuple(measurement['sides'][side]['accuracy'] if measurement['sides'][side] else None
                          for side in SIDES), bool(curve['identifiable']), corpus.content_hash, len(corpus.records))


def _anchor(actuals):
    if actuals is None:
        return None
    if not isinstance(actuals, dict) or set(actuals)-set(SIDES):
        raise ValueError('Supplied account ratings must use White and Black keys.')
    values = []
    for side in SIDES:
        value = actuals.get(side)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 4000:
            raise ValueError('Account ratings must be finite numbers in [0, 4000], or absent.')
        values.append(float(np.clip(value, *ARGS.rating_range)))
    return float(np.mean(values)) if values else None


def moments(prepared, anchor=None):
    """Integrate the synthetic rating/accuracy joint distribution on one grid."""
    grid = prepared.grid
    midpoint = np.mean(ARGS.prior_range)
    shift = 0. if anchor is None else anchor-midpoint
    weights = prior_weights(grid-shift, args=ARGS)
    weights /= trapezoid(weights, grid)
    rating_mean = float(trapezoid(grid*weights, grid))
    accuracy_mean = float(trapezoid(prepared.population_mean*weights, grid))
    covariance = float(trapezoid((grid-rating_mean)*(prepared.population_mean-accuracy_mean)*weights, grid))
    variance = float(trapezoid(((prepared.population_mean-accuracy_mean)**2+prepared.between_variance)*weights, grid))
    variance += prepared.target_variance
    coefficient = max(0., covariance)/variance if variance > 0 else 0.
    cdf = cumulative_trapezoid(weights, grid, initial=0.)
    cdf[-1] = 1.
    return {'rating_mean': rating_mean, 'accuracy_mean': accuracy_mean, 'accuracy_variance': variance,
            'covariance': covariance, 'coefficient': coefficient, 'rating_cdf': cdf}


def predict_prepared(prepared, actuals):
    """Apply the three fixed common maps; predictions never receive PGN labels."""
    if not isinstance(prepared, Prepared):
        raise TypeError('prepare() must supply population moments first.')
    anchor = _anchor(actuals)
    output = {name: dict.fromkeys(SIDES) for name in describe()}
    if not prepared.identifiable:
        return output
    fixed, centered = moments(prepared), moments(prepared, anchor)
    for side, accuracy in zip(SIDES, prepared.accuracies, strict=True):
        if accuracy is None:
            continue
        fixed_point = fixed['rating_mean']+fixed['coefficient']*(accuracy-fixed['accuracy_mean'])
        fixed_point = float(np.clip(fixed_point, *ARGS.rating_range))
        if anchor is not None:
            fixed_point = (1-ACCOUNT_WEIGHT)*fixed_point+ACCOUNT_WEIGHT*anchor
        centered_point = centered['rating_mean']+centered['coefficient']*(accuracy-centered['accuracy_mean'])
        if centered['accuracy_variance'] <= 0 or centered['covariance'] <= 0:
            percentile_point = centered['rating_mean']
        else:
            probability = ndtr((accuracy-centered['accuracy_mean'])/math.sqrt(centered['accuracy_variance']))
            percentile_point = np.interp(probability, centered['rating_cdf'], prepared.grid)
        output['population_affine_common_account5'][side] = float(fixed_point)
        output['population_affine_account_prior'][side] = float(np.clip(centered_point, *ARGS.rating_range))
        output['population_percentile_account_prior'][side] = float(percentile_point)
    return output


def predict(evidence, actuals):
    return predict_prepared(prepare(evidence), actuals)
