"""WITHDRAWN: benchmark-derived population estimation; entry points reject use.

Historical formulas remain for inspection only. Use shared_curve_affine for
the current-game-only estimator. The former test-derived corpus is deleted.

Monotone arithmetic coverage estimation with a common account-rating anchor.

Both players use the same arithmetic Maia curve, conditional variance, frozen
population model and nondecreasing native-coverage decision map. A small common
account contribution preserves its ordering without imposing a minimum gap.
The combined point is a decision rule, not a posterior with a credible interval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from analysis.player_rating.calibration import CalibrationCorpus, load_calibration, WITHDRAWN_REASON
from analysis.player_rating.interface import PlayerRating
from analysis.player_rating.uncertainty_likelihood import (
    ACCURACY_BIN_WIDTH, ACCURACY_GRID_STEP, beta_accuracy_mass, coverage_mean,
    gaussian_accuracy_mass, posterior_mean,
)
from analysis.player_rating.uncertainty_measurement import ARGS as CURVE_ARGS, measure


METHOD = __name__.rsplit('.', 1)[-1]
NAME = 'Shared-curve arithmetic coverage'
VERSION = 1
SIDES = ('White', 'Black')


@dataclass(frozen=True)
class Args:
    """A common account contribution, fixed independently of reference labels."""

    account_weight: float = .05

    def __post_init__(self):
        value = self.account_weight
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('account_weight must be a finite value in [0, 1].')


ARGS = Args()


def _parameters(args, calibration):
    # The baseline's displayed central interval is not used by this point rule.
    curve = {key: value for key, value in asdict(CURVE_ARGS).items() if key != 'central_interval'}
    return {**curve, **asdict(args), 'accuracy_bin_width': ACCURACY_BIN_WIDTH,
            'accuracy_grid_step': ACCURACY_GRID_STEP, 'calibration_hash': calibration.content_hash}


def _account_anchor(actual_ratings):
    """Average available clipped account ratings, preserving missing inputs."""
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


def _component_mean(mass):
    """Return a diagnostic posterior mean, without replacing an underflowed one."""
    point, valid = posterior_mean(mass, CURVE_ARGS.grid)
    return float(point) if bool(valid) else None


def calculate(evidence, actual_ratings, calibration_records, *, args=None):
    """Pure numerical fit using validated evidence and a frozen calibration corpus.

    Account ratings affect only the common final anchor. Exact target contexts
    are excluded from population moments. No labels or external services enter
    the calculation, and input records are never modified.
    """
    raise ValueError(WITHDRAWN_REASON)
    args = ARGS if args is None else args
    if not isinstance(args, Args) or not isinstance(calibration_records, CalibrationCorpus):
        raise TypeError('An Args instance and a CalibrationCorpus are required.')
    actual_ratings = {} if actual_ratings is None else actual_ratings
    anchor, used = _account_anchor(actual_ratings)
    arithmetic = measure(evidence)
    curve = arithmetic['curve']
    available = bool(curve['identifiable'])
    corpus = calibration_records.for_evidence(evidence) if available else calibration_records
    variance = curve['likelihood']['accuracy_variance']
    if available:
        population, population_variance = corpus.population(CURVE_ARGS.grid, 'arithmetic')
        bounds = [curve['monotone_expected_accuracy'][0], curve['monotone_expected_accuracy'][-1]]
    players, components = {}, {}
    for side in SIDES:
        moment = arithmetic['sides'][side]
        observed = moment['accuracy'] if moment else None
        point, coverage = None, None
        details = {'arithmetic_accuracy': observed, 'shared_measurement_variance': variance,
                   'local_mean': None, 'population_mean': None, 'coverage_mean': None,
                   'native_reliability': None, 'maximum_projection_change': None,
                   'decreasing_steps': None, 'actual_rating': actual_ratings.get(side),
                   'account_rating_used': used[side], 'common_account_rating': anchor, 'account_weight': 0.}
        if available and moment is not None:
            coverage = coverage_mean(observed, curve['shared_accuracy'], variance,
                                     population, population_variance, bounds)
            quality = coverage['mean']
            weight = args.account_weight if anchor is not None else 0.
            point = (1-weight)*quality+weight*anchor if anchor is not None else quality
            details.update(coverage_mean=quality, native_reliability=coverage['native_reliability'],
                           maximum_projection_change=coverage['maximum_projection_change'],
                           decreasing_steps=coverage['decreasing_steps'], account_weight=weight,
                           local_mean=_component_mean(gaussian_accuracy_mass(observed, curve['shared_accuracy'], variance)),
                           population_mean=_component_mean(beta_accuracy_mass(observed, population, population_variance)))
        details['unrounded_estimate'] = point
        components[side] = details
        players[side] = {'estimate': math.floor(point+.5) if point is not None else None,
                         'unrounded_estimate': point, 'uncertainty': None, 'interval': None,
                         'moves_used': moment['moves'] if moment else 0,
                         'average_accuracy': observed, 'identifiable': point is not None,
                         'at_rating_limit': point in CURVE_ARGS.rating_range if point is not None else False,
                         'method': METHOD}
    return {'players': players, 'name': NAME, 'method_id': METHOD,
            'parameters': _parameters(args, calibration_records), 'central_interval': None,
            'rating_range': list(CURVE_ARGS.rating_range), 'point_estimator': 'monotone_arithmetic_coverage_mean',
            'prior': {'kind': 'symmetric_fourth_power', 'minimum': CURVE_ARGS.prior_range[0],
                      'maximum': CURVE_ARGS.prior_range[1], 'flat_range': list(CURVE_ARGS.flat_prior_range)},
            'interval_scope': 'Point-only monotone coverage decision with an optional common account anchor; no calibrated uncertainty interval is available. Component posteriors do not define a posterior for the final estimate.',
            'method': 'Full legal-move Maia policies; one shared arithmetic accuracy curve and variance; frozen population accuracy moments; nondecreasing native-coverage mean; optional common mean-account contribution; no competitive weighting or pair-order projection.',
            'diagnostics': {'kind': METHOD, 'curve': curve, 'components': components,
                            'calibration': {'content_hash': corpus.content_hash, 'contexts_used': len(corpus.records),
                                            'contexts_excluded': corpus.excluded_count, 'labels_used': False}},
            'account_ratings_used': any(row['account_weight'] > 0 for row in components.values())}


class Rating(PlayerRating):
    """Filename-selected arithmetic estimator with immutable local arguments."""

    name = NAME
    version = VERSION

    def __init__(self, args=None, calibration=None):
        raise ValueError(WITHDRAWN_REASON)
        self.args = ARGS if args is None else args
        if not isinstance(self.args, Args):
            raise TypeError('Arithmetic-coverage arguments must be an Args instance.')
        self.calibration = load_calibration() if calibration is None else calibration
        if not isinstance(self.calibration, CalibrationCorpus):
            raise TypeError('Arithmetic-coverage calibration must be a CalibrationCorpus.')

    @property
    def parameters(self):
        return _parameters(self.args, self.calibration)

    def fit(self, evidence):
        return calculate(evidence, {side: evidence[side].get('actual_rating') for side in SIDES},
                         self.calibration, args=self.args)
