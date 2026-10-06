"""Use each player's own Maia measurement variance around the shared curve.

The expected curve stays shared. Independent moves give each side a different
variance of its weighted average, however, because its positions and move count
differ. Compare a rating-averaged variance with its full rating-dependent form.
Both alternatives keep fixed quarter-coverage/three-quarter predictive pooling
and a 10% account contribution, followed by the explicit original-order constraint.
No actual played qualities or reference ratings determine the measurement variance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass, posterior
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.universal_native_coverage import _points, native_reliability
from tests.analysis.universal_competitiveness import project_order, weighted_fit


def measurement_variance(record, power):
    """Conditional variance of a weighted mean, including every legal candidate."""
    variances, weights = [], []
    for row in record['observations']:
        q = np.asarray(row['qualities']['position'], dtype=float)
        if len(q) == 1:
            continue
        policy = np.asarray(row['maia_probabilities'], dtype=float)
        mean = policy @ q
        variances.append(np.maximum(0., policy @ (q*q)-mean*mean))
        p = row['position_win_probability']
        weights.append((4*p*(1-p))**power)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    return weights**2 @ variances


def quality_estimates(evidence, fit, calibration_cases, modes=('average', 'varying')):
    """Return unprojected, account-free points for the declared noise models."""
    grid = ARGS.grid
    transformed = weighted_fit(evidence, .5)
    transformed_calibration = [{'fit': weighted_fit(c['evidence'], .5)} for c in calibration_cases]
    _, v, population, between = _population(calibration_cases, grid)
    _, cv, competitive_population, competitive_between = _population(transformed_calibration, grid)
    curve = fit['diagnostics']['curve']
    competitive_curve = transformed['diagnostics']['curve']
    accuracy_grid = np.linspace(0., 100., 1001)
    pop = _points(beta_accuracy_mass(accuracy_grid[:, None], population, between+v.mean()), grid)
    results = {}
    for mode in modes:
        if mode not in ('average', 'varying'):
            raise ValueError('Unknown side-uncertainty model.')
        estimates = {}
        for side in ('White', 'Black'):
            variances = measurement_variance(evidence[side], 0.)
            competitive_variances = measurement_variance(evidence[side], .5)
            ordinary_variance = float(variances.mean()) if mode == 'average' else np.interp(grid, GRID, variances)
            competitive_variance = (float(competitive_variances.mean()) if mode == 'average'
                                    else np.interp(grid, GRID, competitive_variances))
            local = _points(gaussian_accuracy_mass(accuracy_grid[:, None], curve['shared_accuracy'], ordinary_variance), grid)
            reliability = native_reliability(accuracy_grid, curve['monotone_expected_accuracy'][0],
                                             curve['monotone_expected_accuracy'][-1], float(variances.mean()))
            if np.any((~local['valid']) & (reliability > 1e-12)) or not np.all(pop['valid']):
                raise ValueError('Active measurement likelihood has no finite posterior mass.')
            values = np.where(local['valid'], reliability*local['mean']+(1-reliability)*pop['mean'], pop['mean'])
            coverage = np.interp(fit['players'][side]['average_accuracy'], accuracy_grid,
                                 isotonic_regression(values).x)
            predictive = posterior(transformed['players'][side]['average_accuracy'],
                                   competitive_curve['shared_accuracy'], competitive_variance,
                                   competitive_population, competitive_between+cv.mean(), grid)
            quality = .25*coverage+.75*float(predictive['mean'])
            estimates[side] = float(quality)
        results['side_uncertainty_'+mode] = estimates
    return results


def predict(evidence, fit, ratings, calibration_cases):
    values = quality_estimates(evidence, fit, calibration_cases)
    return {name: project_order({side: .9*quality+.1*ratings[side]
                                 for side, quality in pair.items()}, fit)
            for name, pair in values.items()}
