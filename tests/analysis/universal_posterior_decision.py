"""Fixed decision rules for the universal Gaussian/Beta accuracy mixture.

Posterior mean, mode, and inverse expected accuracy are different estimands under
different losses/parameterizations; none is fitted to the commercial references.
The sensitivity-weighted alternative uses a mean-gradient information proxy;
it is not the exact Jeffreys prior of the overlapping mixture. Evidence remains
the same universal mixture for all observations, including curve-edge observations.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_density
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior


def predict(evidence, fit, ratings, calibration_cases):
    del evidence
    grid = ARGS.grid
    _, variances, population, between = _population(calibration_cases, grid)
    population_variance = between + variances.mean()
    diagnostic = fit['diagnostics']['curve']
    curve = np.asarray(diagnostic['shared_accuracy'])
    variance = diagnostic['likelihood']['accuracy_variance']
    results = {}
    for side, player in fit['players'].items():
        observed = player['average_accuracy']
        value = posterior(observed, curve, variance, population, population_variance, grid)
        density = value['density']
        mode = float(grid[np.argmax(density)])
        expected_accuracy = .5*(curve+population)
        mean_accuracy = trapezoid(density*expected_accuracy, grid)
        accuracy_action = float(np.interp(mean_accuracy, expected_accuracy, grid))
        # Fisher information proxy sums component mean sensitivities. It is not
        # the exact Fisher information of the overlapping mixture likelihood.
        information = .5*np.gradient(curve, grid)**2/variance + .5*np.gradient(population, grid)**2/population_variance
        weighted = density*np.sqrt(np.maximum(information, 1e-15))
        cdf = cumulative_trapezoid(weighted, grid, initial=0.)
        actions = {'mixture_mean': float(value['mean']), 'mixture_mode': mode,
                   'mixture_accuracy_action': accuracy_action,
                   'mixture_sensitivity_median': float(np.interp(.5, cdf/cdf[-1], grid))}
        for name, estimate in actions.items():
            results.setdefault(name, {})[side] = estimate
            results.setdefault(name+'_account', {})[side] = .95*estimate + .05*ratings[side]
    return results
