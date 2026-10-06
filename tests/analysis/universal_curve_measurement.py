"""Errors-in-variables inference for a noisy sampled game-accuracy curve.

The game's expected Maia curve is a sample mean over position contexts. Estimate
its sampling variance from per-position expected qualities, not realized player
move variance. Other games provide a population mean and random-context variance;
subtract their own sampling variance before estimating true context variation.
Gaussian conjugate shrinkage then combines the local and population curves.
All shrinkage coefficients follow these moments, with no reference fitting.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import isotonic_regression

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID, SharedCurve, prior_density
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass


def context_variance(evidence):
    variances = []
    for record in evidence.values():
        means = []
        for row in record['observations']:
            q = np.asarray(row['qualities']['position'])
            if len(q) > 1:
                means.append(np.asarray(row['maia_probabilities'])@q)
        if len(means) < 2:
            raise ValueError('Context variance needs at least two positions per side.')
        variances.append(np.var(means, axis=0, ddof=1)/len(means))
    return np.interp(ARGS.grid, GRID, np.sum(variances, axis=0)/4)


def predict(evidence, fit, ratings, calibration_cases):
    grid = ARGS.grid
    diagnostic = fit['diagnostics']['curve']
    local = np.asarray(diagnostic['shared_accuracy'])
    local_noise = context_variance(evidence)
    curves = np.stack([c['fit']['diagnostics']['curve']['shared_accuracy'] for c in calibration_cases])
    sampling = np.stack([context_variance(c['evidence']) for c in calibration_cases])
    population = curves.mean(axis=0)
    between = curves.var(axis=0, ddof=1)
    tau_squared = np.maximum(between-sampling.mean(axis=0), 0.)
    coefficient = tau_squared/np.maximum(tau_squared+local_noise, 1e-12)
    curve = isotonic_regression(population+coefficient*(local-population)).x
    variance = diagnostic['likelihood']['accuracy_variance'] + coefficient*local_noise
    prior = prior_density(grid)
    results = {}
    for measurement in ('normal_constant', 'normal_variable', 'beta_variable'):
        for side, player in fit['players'].items():
            observed = player['average_accuracy']
            if measurement == 'normal_constant':
                mass = gaussian_accuracy_mass(observed, curve, float(variance.mean()))
            elif measurement == 'normal_variable':
                mass = gaussian_accuracy_mass(observed, curve, variance)
            else:
                mass = beta_accuracy_mass(observed, curve, variance)
            density = np.maximum(mass, 1e-300)*prior
            total = trapezoid(density, grid)
            cdf = cumulative_trapezoid(density, grid, initial=0.)/total
            actions = {'mean': float(trapezoid(grid*density, grid)/total),
                       'median': float(np.interp(.5, cdf, grid))}
            for decision, estimate in actions.items():
                name = f'curve_measurement_{measurement}_{decision}'
                results.setdefault(name, {})[side] = estimate
                results.setdefault(name+'_account', {})[side] = .95*estimate+.05*ratings[side]
    return results
