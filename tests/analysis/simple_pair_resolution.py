"""Two reference-free controls for a shared curve's positional-difficulty bias.

At the players' common account rating, Maia may expect different accuracies in
White's and Black's positions. Shift each side's shared expected-accuracy curve
by that side's deviation from the equal-side average. The fixed full correction
trusts this discrepancy; the half correction shrinks it halfway toward zero.
Both keep the original Gaussian likelihood, Maia-derived variance, rating prior,
and posterior-median decision. No commercial label, game name, population corpus,
competitive weighting, fitted coefficient, or post-fit ordering constraint is
used. The account average is a difficulty anchor, not a rating prior.

The conditional Maia variance is reused only for this controlled comparison; it
does not establish empirical human rating resolution or independent-move truth.
"""
from __future__ import annotations

import numpy as np

from analysis.player_rating.bayesian_shared_curve import (
    Args, GRID, curve_posterior, side_moments, summarize,
)


ARGS = Args()
SIDES = ('White', 'Black')
CORRECTIONS = {'pair_difficulty_full': 1., 'pair_difficulty_half': .5}


def describe():
    """Describe the fixed controls for comparison tables without fitting data."""
    return {
        'pair_difficulty_full': 'Shared-curve posterior with full Maia side-difficulty correction at the common account rating.',
        'pair_difficulty_half': 'Shared-curve posterior with half Maia side-difficulty correction at the common account rating.',
    }


def difficulty(evidence, actual_ratings):
    """Return label-free observed and model-expected differences at one anchor."""
    values = [actual_ratings.get(side) for side in SIDES]
    available = [float(value) for value in values if value is not None]
    if any(not np.isfinite(value) or not 0 <= value <= 4000 for value in available):
        raise ValueError('Account ratings must be finite values from 0 through 4000.')
    # No measured Maia policy exists outside this native interval.
    anchor = float(np.clip(np.mean(available) if available else np.mean(GRID), GRID[0], GRID[-1]))
    moments = {side: side_moments(evidence[side], args=ARGS) for side in SIDES}
    expected = {side: float(np.interp(anchor, GRID, moment['mean']))
                for side, moment in moments.items() if moment is not None}
    shared = float(np.mean(list(expected.values()))) if expected else None
    return {'anchor_rating': anchor, 'expected_accuracy': expected,
            'shared_expected_accuracy': shared,
            'offsets': {side: value-shared for side, value in expected.items()},
            'observed_accuracy': {side: moment['accuracy'] if moment else None
                                  for side, moment in moments.items()}}


def predict(evidence, actual_ratings):
    """Fit both fixed discrepancy assumptions through the same shared curve."""
    baseline = summarize(evidence, args=ARGS)
    curve = baseline['diagnostics']['curve']
    context = difficulty(evidence, actual_ratings)
    expected = np.asarray(curve['shared_accuracy'])
    variance = curve['shared_mean_variance']
    predictions = {}
    for name, correction in CORRECTIONS.items():
        estimates = {}
        for side in SIDES:
            observed = context['observed_accuracy'][side]
            if not curve['identifiable'] or observed is None:
                estimates[side] = None
                continue
            # Expected qualities retain their physical 0--100 support. Within
            # those bounds this equals subtracting the discrepancy from A_s.
            adjusted = np.clip(expected+correction*context['offsets'][side], 0., 100.)
            estimates[side] = curve_posterior(observed, adjusted, variance, args=ARGS)['unrounded_estimate']
        predictions[name] = estimates
    return predictions
