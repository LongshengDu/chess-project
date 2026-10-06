"""Three fixed arithmetic-curve discrepancy controls, with no reference fitting.

Let C be the target game's shared curve, P the mean frozen population curve,
v its pooled arithmetic measurement variance and tau2 the mean between-context
curve variance across native Maia ratings 600--2600. J retained contexts receive
equal weight. The three predeclared rules are:

* Random context error: mean C, variance v+tau2.
* Empirical-Bayes curve shrinkage: w=tau2/(tau2+v), mean w*C+(1-w)*P,
  variance v+tau2*v/(tau2+v).
* Native mean centering: subtract mean(C-P) across native ratings from C's
  native knots, clip to0--100 and extend with the existing smooth tails;
  variance v+tau2/J.

Every rule uses a single bounded Gaussian accuracy likelihood, one constant
variance and the same rating prior for both players. The posterior mean is
nondecreasing in accuracy because the likelihood has a monotone likelihood
ratio on the common nondecreasing curve. A5% common account-mean contribution
preserves this ordering. These are point estimators, not calibrated intervals.
The context spread is a model-discrepancy proxy, not measured Maia model error;
shrinking or removing real positional difficulty can itself introduce bias.
All formulas are fixed before scoring; commercial ratings never enter this API.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import ndtr

from analysis.player_rating.bayesian_shared_curve import SharedCurve, prior_density
from analysis.player_rating.calibration import load_calibration
from analysis.player_rating.uncertainty_measurement import ARGS, measure


SIDES = ('White', 'Black')
ACCOUNT_WEIGHT = .05


def describe():
    return {
        'discrepancy_context_variance': 'Local arithmetic curve with additive between-context accuracy variance;5% common account anchor.',
        'discrepancy_curve_shrinkage': 'Empirical-Bayes local/population curve shrinkage from conditional and between-context variance;5% common account anchor.',
        'discrepancy_native_center': 'Remove local/population native-grid mean offset while retaining local curve shape;5% common account anchor.',
    }


def prepare(evidence, actual_ratings=None):
    """Prepare model-only measurements once; account ratings are not consumed."""
    measurement = measure(evidence)
    curve = measurement['curve']
    prepared = {'identifiable': bool(curve['identifiable']),
                'observed': {side: moment['accuracy'] if moment else None
                             for side, moment in measurement['sides'].items()},
                'models': {}}
    if not prepared['identifiable']:
        return prepared
    corpus = load_calibration().for_evidence(evidence)
    native = np.array([record.arithmetic.knots for record in corpus.records])
    curves = np.array([SharedCurve(knots)(ARGS.grid) for knots in native])
    population = curves.mean(axis=0)
    population_knots = native.mean(axis=0)
    tau2 = float(np.var(native, axis=0, ddof=int(len(native) > 1)).mean())
    variance = float(curve['likelihood']['accuracy_variance'])
    denominator = tau2+variance
    weight = tau2/denominator if denominator else 1.
    local = np.asarray(curve['shared_accuracy'])
    offset = float(np.mean(np.asarray(curve['monotone_expected_accuracy'])-population_knots))
    centered = SharedCurve(np.clip(np.asarray(curve['monotone_expected_accuracy'])-offset, 0., 100.))(ARGS.grid)
    prepared.update(
        models={
            'discrepancy_context_variance': {'curve': local, 'variance': variance+tau2},
            'discrepancy_curve_shrinkage': {'curve': weight*local+(1-weight)*population,
                                          'variance': variance+(tau2*variance/denominator if denominator else 0.)},
            'discrepancy_native_center': {'curve': centered, 'variance': variance+tau2/len(native)},
        },
        variance=variance, between_context_variance=tau2, local_weight=weight,
        native_offset=offset, contexts_used=len(native), calibration_hash=corpus.content_hash,
        contexts_excluded=corpus.excluded_count,
    )
    return prepared


def points_at(accuracy, prepared):
    """Posterior means for one observed arithmetic accuracy in a fixed context."""
    if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError('Arithmetic accuracy must lie in [0, 100].')
    grid = ARGS.grid
    prior = prior_density(grid, args=ARGS)
    active = prior > 0
    points = {}
    for method, model in prepared['models'].items():
        curve = model['curve']
        variance = max(model['variance'], 1e-9)
        sigma = np.sqrt(variance)
        # Density of an accuracy observation under a Gaussian truncated to its
        # physical0--100 support. Work in log space to avoid edge underflow.
        normalizer = ndtr((100.-curve)/sigma)-ndtr(-curve/sigma)
        log_density = -.5*(accuracy-curve)**2/variance-np.log(normalizer)
        log_density[active] += np.log(prior[active])
        density = np.zeros_like(grid)
        density[active] = np.exp(log_density[active]-log_density[active].max())
        density /= trapezoid(density, grid)
        points[method] = float(trapezoid(grid*density, grid))
    return points


def predict_prepared(prepared, actual_ratings):
    """Apply the fixed account anchor without recomputing chess evidence."""
    accounts = [actual_ratings.get(side) for side in SIDES]
    available = [value for value in accounts if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or not 0 <= value <= 4000 for value in available):
        raise ValueError('Actual ratings must be finite numbers in [0, 4000], or absent.')
    anchor = float(np.mean(np.clip(available, *ARGS.rating_range))) if available else None
    results = {name: dict.fromkeys(SIDES) for name in describe()}
    if not prepared['identifiable']:
        return results
    for side, accuracy in prepared['observed'].items():
        if accuracy is None:
            continue
        for method, point in points_at(accuracy, prepared).items():
            results[method][side] = point if anchor is None else (1-ACCOUNT_WEIGHT)*point+ACCOUNT_WEIGHT*anchor
    return results


def predict(evidence, actual_ratings):
    return predict_prepared(prepare(evidence, actual_ratings), actual_ratings)
