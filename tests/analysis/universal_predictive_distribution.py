"""Universal exact-policy predictive accuracy and local/population mixtures.

Use the full independent-position convolution, rather than Gaussian/Beta moments,
for the local game's shared accuracy distribution as well as calibration games.
The equal-side local mixture is not the distribution of pooling both colors'
moves: it is uncertainty over which side's position contexts generated a player.
All quality-rounding and native-rating limits match the existing FFT experiment.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from analysis.player_rating.bayesian_shared_curve import GRID, prior_weights
from tests.analysis.edge_predictive_accuracy import _game_distributions, _bin_probabilities, _population_likelihood


def predict(evidence, fit, ratings, calibration_cases):
    local_distributions = _game_distributions(evidence)
    grid = np.arange(GRID[0], GRID[-1]+1, 5.)
    prior = prior_weights(grid)
    results = {}
    for side, player in fit['players'].items():
        accuracy = player['average_accuracy']
        local = np.mean([_bin_probabilities(row, accuracy) for row in local_distributions], axis=0)
        population = _population_likelihood(calibration_cases, accuracy)
        for context, likelihood in (('local', local), ('mixture', .5*(local+population)),
                                    ('geometric', np.sqrt(local*population))):
            logs = np.interp(grid, GRID, np.log(np.maximum(likelihood, 1e-300)))
            density = np.exp(logs-logs.max())*prior
            total = trapezoid(density, grid)
            cdf = cumulative_trapezoid(density, grid, initial=0.)/total
            actions = {'mean': float(trapezoid(grid*density, grid)/total),
                       'median': float(np.interp(.5, cdf, grid)),
                       'mode': float(grid[np.argmax(density)])}
            for decision, estimate in actions.items():
                name = f'policy_predictive_{context}_{decision}'
                results.setdefault(name, {})[side] = estimate
                results.setdefault(name+'_account', {})[side] = .95*estimate+.05*ratings[side]
    return results
