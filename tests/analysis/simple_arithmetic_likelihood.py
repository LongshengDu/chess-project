"""Arithmetic-only ablations using the unchanged frozen calibration corpus.

The first decision is the posterior mean of one equal-prior local Gaussian /
population Beta accuracy likelihood mixture. The second is the existing
monotone native-coverage mean. Both use one shared arithmetic curve and one
shared variance, with no competitive measurement or side-variance averaging.

A fixed 5% contribution from the mean available account rating is common to
both players. This cannot change their ordering or its contrast; a single
account changed by200 changes both points by5 when both accounts are present.
No pair projection or minimum separation is imposed. The Gaussian/Beta mixture
does not itself guarantee monotonic ordering. The coverage decision includes
its documented monotone mapping projection, not a per-game order correction.
"""
from __future__ import annotations

import numpy as np

from analysis.player_rating.calibration import load_calibration
from analysis.player_rating.uncertainty_likelihood import coverage_mean, predictive_mean
from analysis.player_rating.uncertainty_measurement import ARGS, measure


SIDES = ('White', 'Black')
ACCOUNT_WEIGHT = .05


def describe():
    """Fixed method descriptions for the shared reference-scoring harness."""
    return {
        'arithmetic_predictive_common_account': 'Arithmetic Gaussian/Beta likelihood-mixture mean; one shared variance and 5% common account anchor.',
        'arithmetic_coverage_common_account': 'Monotone arithmetic native-coverage mean; one shared variance and 5% common account anchor.',
    }


def predict(evidence, actual_ratings):
    """Estimate both players from arithmetic quality; references are not inputs."""
    values = [actual_ratings.get(side) for side in SIDES]
    available = [value for value in values if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or not 0 <= value <= 4000 for value in available):
        raise ValueError('Actual ratings must be finite numbers in [0, 4000], or absent.')
    anchor = float(np.mean(np.clip(available, *ARGS.rating_range))) if available else None
    arithmetic = measure(evidence)
    curve = arithmetic['curve']
    results = {name: dict.fromkeys(SIDES) for name in describe()}
    if not curve['identifiable']:
        return results
    # Exact numeric target exclusion uses the frozen production corpus. No new
    # games, played outcomes, account ratings, or labels are added to the asset.
    corpus = load_calibration().for_evidence(evidence)
    population, population_variance = corpus.population(ARGS.grid, 'arithmetic')
    mean, variance = curve['shared_accuracy'], curve['likelihood']['accuracy_variance']
    bounds = [curve['monotone_expected_accuracy'][0], curve['monotone_expected_accuracy'][-1]]
    for side in SIDES:
        moment = arithmetic['sides'][side]
        if moment is None:
            continue
        accuracy = moment['accuracy']
        points = {
            'arithmetic_predictive_common_account': predictive_mean(
                accuracy, mean, variance, population, population_variance, .5)['mean'],
            'arithmetic_coverage_common_account': coverage_mean(
                accuracy, mean, variance, population, population_variance, bounds)['mean'],
        }
        for method, point in points.items():
            results[method][side] = point if anchor is None else (1-ACCOUNT_WEIGHT)*point+ACCOUNT_WEIGHT*anchor
    return results
