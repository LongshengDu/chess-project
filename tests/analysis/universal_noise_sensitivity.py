"""Declared one-factor sensitivity of a proper Gaussian/Beta likelihood mixture.

The only changed statistical assumption is conditional accuracy sigma, evaluated
at fixed multipliers0.5,0.75,1.0. Target variance and mean population conditional
variance receive the squared multiplier. Between-context variance is unchanged.
The two model priors remain equal, the rating prior is unchanged, and the point
decision is always the posterior mean. The optional account blend is fixed at5%.

This is a coarse sensitivity analysis, not reference-trained parameter fitting.
All method coefficients are common across games and both players.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior
from tests.analysis.universal_accuracy_likelihood import _numeric_fit


@dataclass(frozen=True)
class NoiseArgs:
    accuracy_sigma_scale: float = 1.

    def __post_init__(self):
        value = self.accuracy_sigma_scale
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('accuracy_sigma_scale must be finite and strictly positive.')

    @property
    def method(self):
        return f'mixture_noise_{round(100*self.accuracy_sigma_scale):03d}_mean_all'


SCENARIOS = tuple(NoiseArgs(value) for value in (.5, .75, 1.))
ACCOUNT_WEIGHT = .05
METHODS = tuple(name for scenario in SCENARIOS
                for name in (scenario.method, scenario.method[:-4]+'_account_5pct_all'))


def scaled_variances(target_variance, calibration_variances, between_context_variance, args):
    """Scale measurement variance only; contextual heterogeneity stays unchanged."""
    square = args.accuracy_sigma_scale**2
    return target_variance*square, np.asarray(between_context_variance)+np.mean(calibration_variances)*square


@lru_cache(maxsize=64)
def _cached_fit(target_json, calibration_json):
    import json
    target = json.loads(target_json)
    calibration = [{'fit': json.loads(value)} for value in calibration_json]
    _, variances, pooled, between = _population(calibration, ARGS.grid)
    diagnostic = target['diagnostics']['curve']
    if not np.array_equal(np.asarray(diagnostic['fine_ratings']), ARGS.grid):
        raise ValueError('The target fit must use the current production rating grid.')
    means = np.asarray(diagnostic['shared_accuracy'])
    results = {scenario.method: {} for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        target_variance, population_variance = scaled_variances(
            diagnostic['likelihood']['accuracy_variance'], variances, between, scenario)
        for side in ('White', 'Black'):
            player = target['players'][side]
            if player['average_accuracy'] is None or player['estimate'] is None:
                results[scenario.method][side] = player['estimate']
            else:
                result = posterior(player['average_accuracy'], means, target_variance,
                                   pooled, population_variance, ARGS.grid)
                results[scenario.method][side] = float(result['mean'])
    return results


def predict(evidence, fit, ratings, calibration_cases):
    """Return all six predeclared universal noise-sensitivity estimates."""
    import json
    del evidence
    if not calibration_cases:
        raise ValueError('Independent calibration cases are required.')
    target = _numeric_fit(fit)
    target['players'] = {side: {key: fit['players'][side][key] for key in ('average_accuracy', 'estimate')}
                         for side in ('White', 'Black')}
    predictions = _cached_fit(json.dumps(target, sort_keys=True),
                              tuple(json.dumps(_numeric_fit(case['fit']), sort_keys=True) for case in calibration_cases))
    outputs = {}
    for name, pair in predictions.items():
        outputs[name] = dict(pair)
        blended = {}
        for side, estimate in pair.items():
            actual = ratings.get(side)
            if estimate is None:
                blended[side] = None
            elif actual is None or not np.isfinite(actual) or not 0 <= actual <= 3200:
                raise ValueError('Account ratings must be finite and lie in[0,3200].')
            else:
                blended[side] = (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*actual
        outputs[name[:-4]+'_account_5pct_all'] = blended
    return outputs
