"""Continuous reference-free combinations of context and population performance.

Equal weights express no preference between the two reference scales; they are
not learned from commercial labels. Both component Gaussian likelihoods use a
rating-independent variance, giving stochastic monotonicity in observed accuracy.
A fixed mixture of their posteriors therefore retains that ordering. The optional
5% account contribution has bounded influence (10 Elo per +/-200 account shift).
These all-player alternatives deliberately avoid a discontinuous edge switch.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from analysis.player_rating.bayesian_shared_curve import ARGS, curve_posterior
from tests.analysis.edge_global_quality import _population


def predict(evidence, fit, ratings, calibration_cases):
    del evidence
    grid = ARGS.grid
    _, variances, pooled, between = _population(calibration_cases, grid)
    # A common noise scale makes more accuracy stochastically imply more strength.
    population_variance = float(np.mean(between)+np.mean(variances))
    own = fit['diagnostics']['curve']
    result = {name: {} for name in ('consensus_point_all', 'consensus_posterior_all',
                                    'consensus_posterior_account_all')}
    for side in ('White', 'Black'):
        player = fit['players'][side]
        observed = player['average_accuracy']
        if observed is None or player['estimate'] is None:
            for pair in result.values(): pair[side] = None
            continue
        local = curve_posterior(observed, own['shared_accuracy'], own['likelihood']['accuracy_variance'])
        global_fit = curve_posterior(observed, pooled, population_variance)
        result['consensus_point_all'][side] = .5*(local['unrounded_estimate']+global_fit['unrounded_estimate'])
        density = .5*(np.asarray(local['posterior_density'])+np.asarray(global_fit['posterior_density']))
        cdf = cumulative_trapezoid(density, grid, initial=0.)
        median = float(np.interp(.5, cdf/cdf[-1], grid))
        result['consensus_posterior_all'][side] = median
        result['consensus_posterior_account_all'][side] = .95*median+.05*float(ratings[side])
    return result
