"""Universal side-conditioned quality likelihoods with an ordered pair decision.

Shared curves discard each side's distinct position difficulty. This experiment
restores that information in either full or equal shared/side form and compares
bounded Beta with Gaussian measurement distributions. An optional population
component uses only other games' Maia predictions. Parameters are structural
choices fixed before reference scoring, not commercial-rating corrections.

Both sides are inferred jointly conditional on their arithmetic-accuracy order.
This constraint is observable without a commercial label and holds in every
tested game's original shared-curve ordering. Joint posterior means, rather than
independent medians, preserve the strict ordering under the conditional support.
The constraint is an assumption, not independent evidence about who played best.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import isotonic_regression

from analysis.player_rating.bayesian_shared_curve import ARGS, SharedCurve, prior_density, side_moments
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass

ACCOUNT_WEIGHT = .05


def ordered_means(densities, grid, higher_side):
    """Condition independent likelihoods on the observable higher/lower order."""
    lower_side = 'Black' if higher_side == 'White' else 'White'
    higher, lower = densities[higher_side], densities[lower_side]
    higher = higher / trapezoid(higher, grid)
    lower = lower / trapezoid(lower, grid)
    lower_cdf = cumulative_trapezoid(lower, grid, initial=0.)
    higher_survival = -cumulative_trapezoid(higher[::-1], grid[::-1], initial=0.)[::-1]
    high_marginal, low_marginal = higher*lower_cdf, lower*higher_survival
    high_mass, low_mass = trapezoid(high_marginal, grid), trapezoid(low_marginal, grid)
    if min(high_mass, low_mass) <= 0:
        raise ValueError('Ordered pair has no finite posterior mass.')
    return {higher_side: float(trapezoid(grid*high_marginal, grid)/high_mass),
            lower_side: float(trapezoid(grid*low_marginal, grid)/low_mass)}


def predict(evidence, fit, ratings, calibration_cases):
    grid = ARGS.grid
    diagnostic = fit['diagnostics']['curve']
    shared = np.asarray(diagnostic['shared_accuracy'])
    shared_variance = float(diagnostic['likelihood']['accuracy_variance'])
    _, calibration_variances, population, between = _population(calibration_cases, grid)
    population_variance = between + calibration_variances.mean()
    prior = prior_density(grid)
    observations = {side: fit['players'][side]['average_accuracy'] for side in ('White', 'Black')}
    higher_side = max(observations, key=observations.get)
    moments = {side: side_moments(evidence[side]) for side in observations}
    side_curves = {side: SharedCurve(isotonic_regression(moment['mean']).x)(grid)
                   for side, moment in moments.items()}
    results = {}
    for label, side_weight in (('side', 1.), ('shared_side', .5)):
        for measurement, likelihood in (('normal', gaussian_accuracy_mass), ('beta', beta_accuracy_mass)):
            for context in ('local', 'population_mixture'):
                densities = {}
                for side, observed in observations.items():
                    curve = shared + side_weight*(side_curves[side]-shared)
                    mass = likelihood(observed, curve, shared_variance)
                    if context == 'population_mixture':
                        mass = .5*mass + .5*beta_accuracy_mass(observed, population, population_variance)
                    densities[side] = np.maximum(mass, 1e-300)*prior
                estimates = ordered_means(densities, grid, higher_side)
                name = f'pair_{label}_{measurement}_{context}'
                results[name] = estimates
                results[name+'_account'] = {side: (1-ACCOUNT_WEIGHT)*value+ACCOUNT_WEIGHT*ratings[side]
                                           for side, value in estimates.items()}
    return results
