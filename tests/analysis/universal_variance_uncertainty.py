"""Universal mixture models for conditional-variance uncertainty.

Three declared assumptions are compared: the unchanged Gaussian control; a
variance-matched Student-t(df4) target measurement; and Gaussian measurements with
an independent prediction-error variance equal to conditional move variance.
The last model doubles target and population conditional variance but leaves
between-context variance unchanged. There is no parameter search within families.

Student-t variance matches before truncation; normalization to accuracy0--100 can
change its conditional mean/variance, as it does for the truncated Gaussian.
All models use the same0.01-point measurement bins, equal model priors, rating
prior and posterior-mean decision. Account variants retain the fixed5% blend.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import stdtr

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_density
from tests.analysis.edge_bounded_accuracy import _interval, _stable_mass, beta_accuracy_mass
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass
from tests.analysis.universal_accuracy_likelihood import _numeric_fit


STUDENT_DF = 4.
ACCOUNT_WEIGHT = .05


@dataclass(frozen=True)
class VarianceModel:
    name: str
    student: bool = False
    conditional_variance_multiplier: float = 1.


MODELS = (
    VarianceModel('variance_gaussian_control_mean_all'),
    VarianceModel('variance_student_t4_mean_all', student=True),
    VarianceModel('variance_prediction_noise_mean_all', conditional_variance_multiplier=2.),
)
METHODS = tuple(name for model in MODELS for name in (model.name, model.name[:-4]+'_account_5pct_all'))


def student_accuracy_mass(accuracy, mean, variance):
    """Normalized bounded Student-t bins, matched to untruncated target variance."""
    mean = np.asarray(mean, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if (not np.isfinite(mean).all() or np.any((mean < 0) | (mean > 100))
            or not np.isfinite(variance).all() or np.any(variance < 0)):
        raise ValueError('Bounded finite means and nonnegative variances are required.')
    lower, upper = _interval(accuracy)
    scale = np.sqrt(np.maximum(variance, 1e-9)*(STUDENT_DF-2)/STUDENT_DF)
    lower_z, upper_z = (lower-mean)/scale, (upper-mean)/scale
    mass = _stable_mass(stdtr(STUDENT_DF, lower_z), stdtr(STUDENT_DF, upper_z),
                        stdtr(STUDENT_DF, -lower_z), stdtr(STUDENT_DF, -upper_z))
    low, high = -mean/scale, (100.-mean)/scale
    normalizer = _stable_mass(stdtr(STUDENT_DF, low), stdtr(STUDENT_DF, high),
                              stdtr(STUDENT_DF, -low), stdtr(STUDENT_DF, -high))
    if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
        raise ValueError('The bounded Student-t measurement has no finite mass.')
    return np.clip(mass/normalizer, 0., 1.)


def posterior_mean(accuracy, target_mean, target_variance, population_mean, population_conditional_variance,
                   between_context_variance, model, grid):
    """One proper likelihood-mixture posterior mean for any of the three models."""
    multiplier = model.conditional_variance_multiplier
    local_variance = target_variance*multiplier
    population_variance = np.asarray(between_context_variance)+population_conditional_variance*multiplier
    function = student_accuracy_mass if model.student else gaussian_accuracy_mass
    mass = .5*(function(accuracy, target_mean, local_variance)
               +beta_accuracy_mass(accuracy, population_mean, population_variance))
    weights = mass*prior_density(grid)
    total = trapezoid(weights, grid, axis=-1)
    if np.any(total <= 0) or not np.isfinite(total).all():
        raise ValueError('The proper likelihood mixture has no finite posterior mass.')
    return trapezoid(weights*np.asarray(grid), grid, axis=-1)/total


@lru_cache(maxsize=64)
def _cached_fit(target_json, calibration_json):
    target = json.loads(target_json)
    calibration = [{'fit': json.loads(value)} for value in calibration_json]
    _, variances, population, between = _population(calibration, ARGS.grid)
    own = target['diagnostics']['curve']
    if not np.array_equal(own['fine_ratings'], ARGS.grid):
        raise ValueError('The fit must use the current production rating grid.')
    results = {model.name: {} for model in MODELS}
    for side in ('White', 'Black'):
        player = target['players'][side]
        for model in MODELS:
            if player['average_accuracy'] is None or player['estimate'] is None:
                results[model.name][side] = player['estimate']
            else:
                results[model.name][side] = float(posterior_mean(
                    player['average_accuracy'], np.asarray(own['shared_accuracy']), own['likelihood']['accuracy_variance'],
                    population, variances.mean(), between, model, ARGS.grid))
    return results


def predict(evidence, fit, ratings, calibration_cases):
    """Apply each declared variance-uncertainty model to every observed player."""
    del evidence
    if not calibration_cases:
        raise ValueError('Independent calibration games are required.')
    target = _numeric_fit(fit)
    target['players'] = {side: {key: fit['players'][side][key] for key in ('average_accuracy', 'estimate')}
                         for side in ('White', 'Black')}
    raw = _cached_fit(json.dumps(target, sort_keys=True),
                      tuple(json.dumps(_numeric_fit(case['fit']), sort_keys=True) for case in calibration_cases))
    output = {}
    for name, values in raw.items():
        output[name] = dict(values)
        account = {}
        for side, estimate in values.items():
            actual = ratings.get(side)
            if estimate is None:
                account[side] = None
            elif actual is None or not np.isfinite(actual) or not 0 <= actual <= 3200:
                raise ValueError('Account ratings must be finite and lie in[0,3200].')
            else:
                account[side] = (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*actual
        output[name[:-4]+'_account_5pct_all'] = account
    return output
