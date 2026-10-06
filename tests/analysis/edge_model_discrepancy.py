"""Fixed, reference-free edge experiments with explicit Maia model discrepancy.

These candidates keep every measured-curve intersection unchanged. They are not
production estimators and contain no fitted commercial-reference coefficients.
Account Elo enters only through a deliberately broad performance prior, except
the explicitly labelled other-player bias model. A weak prior cannot resolve an
unidentified likelihood; a strong one necessarily increases account sensitivity.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import isotonic_regression
from scipy.special import gammaln, logsumexp

from analysis.player_rating.bayesian_shared_curve import SharedCurve, prior_density, side_moments


# Structural assumptions fixed before examining commercial estimates. The wide
# 800-Elo prior allows substantial game performance variation. The 400-Elo
# variation below is used only to avoid treating the opponent's account as truth.
ACCOUNT_SIGMA = 800.
OPPONENT_PERFORMANCE_SIGMA = 400.
STUDENT_DF = 4.
CONTAMINATION_PROBABILITY = .1
CONTEXT_POOLING = .5
METHODS = (
    'discrepancy_gaussian_bias',
    'discrepancy_student_t',
    'discrepancy_contamination',
    'discrepancy_context_pooling',
    'discrepancy_other_player_bias',
)


def _median(grid, log_likelihood, account):
    """Integrate over the production prior times a broad account-rating prior."""
    prior = prior_density(grid)
    supported = prior > 0
    log_weights = np.asarray(log_likelihood, dtype=float).copy()
    if account is not None and np.isfinite(account):
        log_weights -= .5*((grid-float(account))/ACCOUNT_SIGMA)**2
    log_weights[supported] += np.log(prior[supported])
    weights = np.zeros_like(grid)
    weights[supported] = np.exp(log_weights[supported]-log_weights[supported].max())
    cdf = cumulative_trapezoid(weights, grid, initial=0)
    cdf /= cdf[-1]
    return float(np.interp(.5, cdf, grid))


def _gaussian(observed, expected, variance):
    variance = np.maximum(variance, 1e-10)
    return -.5*((observed-expected)**2/variance + np.log(2*np.pi*variance))


def _curve(moment, grid):
    return SharedCurve(np.clip(isotonic_regression(moment['mean']).x, 0, 100))(grid)


def predict(evidence, fit, ratings):
    """Return five candidate point estimates, modifying only nonintersections.

``fit`` is the current production summarize result; ``ratings`` maps White and
Black to account Elo. Both players' complete equal-opponent Maia evidence is
required. Forced moves are excluded by the same side_moments implementation as
production. The observation, shared curve and production sigma are unchanged.
    """
    diagnostic = fit['diagnostics']['curve']
    grid = np.asarray(diagnostic['fine_ratings'], dtype=float)
    shared = np.asarray(diagnostic['shared_accuracy'], dtype=float)
    measured = np.asarray(diagnostic['monotone_expected_accuracy'], dtype=float)
    variance = max(float(diagnostic['likelihood']['accuracy_variance']), 1e-10)
    moments = {side: side_moments(evidence[side]) for side in ('White', 'Black')}
    curves = {side: _curve(moment, grid) for side, moment in moments.items() if moment is not None}
    if len(curves) == 2:
        # The half-difference is the empirical context discrepancy of either
        # side from the equal-side shared expectation. It is a sensitivity
        # proxy, not an externally validated estimate of Maia's true error.
        discrepancy_variance = ((curves['White']-curves['Black'])/2)**2
    else:
        discrepancy_variance = np.zeros_like(grid)
    outputs = {name: {} for name in METHODS}
    for side, other in (('White', 'Black'), ('Black', 'White')):
        player = fit['players'][side]
        baseline = player.get('unrounded_estimate', player['estimate'])
        observed = player['average_accuracy']
        if (baseline is None or observed is None
                or measured[0]-1e-10 <= observed <= measured[-1]+1e-10):
            for values in outputs.values():
                values[side] = baseline
            continue

        account = ratings.get(side)
        total_variance = variance+discrepancy_variance
        outputs['discrepancy_gaussian_bias'][side] = _median(
            grid, _gaussian(observed, shared, total_variance), account)

        # Match the Student-t variance to the Gaussian variance; df=4 leaves a
        # finite variance while allowing large positive/negative prediction error.
        scale_squared = total_variance*(STUDENT_DF-2)/STUDENT_DF
        student = (gammaln((STUDENT_DF+1)/2)-gammaln(STUDENT_DF/2)
                   -.5*np.log(STUDENT_DF*np.pi*scale_squared)
                   -(STUDENT_DF+1)/2*np.log1p((observed-shared)**2/(STUDENT_DF*scale_squared)))
        outputs['discrepancy_student_t'][side] = _median(grid, student, account)

        normal = _gaussian(observed, shared, variance)
        mixture = logsumexp(np.vstack((
            math.log(1-CONTAMINATION_PROBABILITY)+normal,
            np.full_like(grid, math.log(CONTAMINATION_PROBABILITY/100.)),
        )), axis=0)
        outputs['discrepancy_contamination'][side] = _median(grid, mixture, account)

        own_curve = curves.get(side, shared)
        context = shared+CONTEXT_POOLING*(own_curve-shared)
        remaining_variance = variance+(1-CONTEXT_POOLING)**2*discrepancy_variance
        outputs['discrepancy_context_pooling'][side] = _median(
            grid, _gaussian(observed, context, remaining_variance), account)

        # Common-game bias is weakly informed by the other player's observation.
        # Their account rating is a noisy anchor, with 400-Elo game variation.
        # Without the account anchor, game bias and both ratings are unidentified.
        other_accuracy = fit['players'][other]['average_accuracy']
        other_account = ratings.get(other)
        bias, bias_variance = 0., discrepancy_variance
        if other_accuracy is not None and other_account is not None and np.isfinite(other_account):
            other_expectation = float(np.interp(other_account, grid, shared))
            local_slope = float(np.interp(other_account, grid, np.gradient(shared, grid)))
            anchor_noise = variance+(local_slope*OPPONENT_PERFORMANCE_SIGMA)**2
            shrinkage = discrepancy_variance/(discrepancy_variance+anchor_noise)
            bias = shrinkage*(other_accuracy-other_expectation)
            bias_variance = discrepancy_variance*(1-shrinkage)
        outputs['discrepancy_other_player_bias'][side] = _median(
            grid, _gaussian(observed, shared+bias, variance+bias_variance), account)
    return outputs
