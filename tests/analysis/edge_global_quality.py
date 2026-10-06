"""Label-free, leave-game-out population-curve experiments for edge observations.

The calibration corpus supplies only Maia-predicted accuracy curves and their
conditional variances. Neither played qualities nor account/commercial ratings
from calibration games enter the population model. This is model-internal
calibration over a small, non-independent position corpus, not validation against
an independent human cohort. Commercial references must only be used by the
calling harness after predictions have been produced.

All choices below are fixed before reference evaluation. Only observations
outside their own measured 600--2600 shared curve use these fallbacks; an
intersecting observation returns its production estimate exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID, SharedCurve, prior_weights


METHODS = {
    'global_pooled': 'Population curve with between-game prediction variance.',
    'global_pooled_weak_account': 'Population curve and a weak 800-Elo account prior.',
    'global_pooled_account_5pct': 'Population posterior median with a fixed 5% account blend.',
    'global_residual_transport': 'Transport the target game residual at actual Elo to the population curve.',
    'global_partial_pool': 'Precision-based shrinkage of the game curve toward the population curve.',
    'global_account_tangent': 'Account-anchored accuracy delta using the population curve local slope.',
    'global_pooled_mixture': 'Equal-game mixture of Maia accuracy likelihoods, retaining difficulty heterogeneity.',
    'global_population_game_quality': 'Population posterior predictive accuracy with corpus-derived game noise.',
    'global_population_account_5pct': 'Population game-quality median with a fixed 5% account contribution.',
}


def _posterior_median(log_likelihood, grid, account_rating=None):
    """Use the unchanged production prior; optionally add a broad account prior."""
    weights = prior_weights(grid)
    valid = weights > 0
    logs = np.asarray(log_likelihood, dtype=float).copy()
    logs[valid] += np.log(weights[valid])
    if account_rating is not None:
        # Deliberately weak and predeclared; this is not a fitted hyperparameter.
        logs -= .5*((grid-account_rating)/800.)**2
    density = np.zeros_like(grid)
    density[valid] = np.exp(logs[valid]-logs[valid].max())
    cdf = cumulative_trapezoid(density, grid, initial=0)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError('Population fallback has no finite posterior mass.')
    return float(np.interp(.5, cdf/cdf[-1], grid))


def _normal_log_likelihood(accuracy, mean, variance):
    variance = np.maximum(np.asarray(variance, dtype=float), 1e-12)
    # The normalization term matters when model variance varies with rating.
    return -.5*((accuracy-mean)**2/variance+np.log(variance))


def _curve(fit):
    knots = np.asarray(fit['diagnostics']['curve']['monotone_expected_accuracy'], dtype=float)
    return SharedCurve(knots)


def _variance(fit):
    return float(fit['diagnostics']['curve']['likelihood']['accuracy_variance'])


def _population(calibration_cases, grid):
    if not calibration_cases:
        raise ValueError('An independent calibration-game corpus is required.')
    # One vote per game avoids domination by long games and their repeated positions.
    curves = np.stack([_curve(case['fit'])(grid) for case in calibration_cases])
    variances = np.asarray([_variance(case['fit']) for case in calibration_cases])
    pooled = curves.mean(axis=0)
    between = np.var(curves, axis=0, ddof=int(len(curves) > 1))
    return curves, variances, pooled, between


def predict(evidence, fit, ratings, calibration_cases, *, edge_only=True):
    """Return all fixed population fallbacks, retaining all intersecting estimates.

    ``calibration_cases`` must exclude the target game. Each case needs a ``fit``;
    no game IDs or reference estimates are inspected. ``ratings`` contains the
    target player's actual ``White``/``Black`` Elo. Evidence is accepted to match
    the common experimental interface; the saved fit already supplies the same
    shared curve, played arithmetic mean and conditional measurement variance.
    """
    del evidence
    grid = ARGS.grid
    own_curve = _curve(fit)
    own = own_curve(grid)
    measured = own_curve(GRID[[0, -1]])
    variance = _variance(fit)
    curves, calibration_variances, pooled, between = _population(calibration_cases, grid)
    results = {name: {} for name in METHODS}

    for side in ('White', 'Black'):
        player = fit['players'][side]
        accuracy = player['average_accuracy']
        baseline = player['estimate']
        if accuracy is None or baseline is None or (edge_only and measured[0] <= accuracy <= measured[1]):
            for values in results.values():
                values[side] = baseline
            continue

        account = ratings.get(side)
        if account is None or not np.isfinite(account):
            raise ValueError('Actual account ratings are required by the anchored experiments.')
        account = float(account)
        population_variance = variance+between
        pooled_logs = _normal_log_likelihood(accuracy, pooled, population_variance)
        results['global_pooled'][side] = _posterior_median(pooled_logs, grid)
        results['global_pooled_weak_account'][side] = _posterior_median(pooled_logs, grid, account)
        # A predeclared sensitivity control, not a reference-trained correction:
        # moving the supplied rating by 200 shifts this estimate by exactly 10.
        results['global_pooled_account_5pct'][side] = .95*results['global_pooled'][side]+.05*account

        transported = accuracy-float(own_curve(account))+float(np.interp(account, grid, pooled))
        # A residual may extend beyond [0,100]; it is an unbounded measurement of
        # a bounded latent mean, not a newly invented valid chess-accuracy value.
        logs = _normal_log_likelihood(transported, pooled, population_variance)
        results['global_residual_transport'][side] = _posterior_median(logs, grid)

        # A game curve with low conditional measurement noise keeps greater weight.
        # No commercial target or account rating determines this reliability.
        reliability = between/np.maximum(between+variance, 1e-12)
        mixed_curve = reliability*own+(1-reliability)*pooled
        model_variance = variance+(1-reliability)*between
        logs = _normal_log_likelihood(accuracy, mixed_curve, model_variance)
        results['global_partial_pool'][side] = _posterior_median(logs, grid)

        # Local first-order inverse of the population accuracy curve at actual Elo.
        # Unlike a win-rate-to-Elo formula, the scale is entirely Maia-derived.
        slope = float(np.interp(account, grid, np.gradient(pooled, grid)))
        slope = max(slope, 1e-10)
        at_account = float(np.interp(account, grid, pooled))
        tangent_mean = at_account+slope*(grid-account)
        tangent_variance = variance+float(np.interp(account, grid, between))
        logs = _normal_log_likelihood(accuracy, tangent_mean, tangent_variance)
        results['global_account_tangent'][side] = _posterior_median(logs, grid)

        mixture_logs = _normal_log_likelihood(
            accuracy, curves, variance+calibration_variances[:, None])
        logs = logsumexp(mixture_logs, axis=0)-np.log(len(curves))
        results['global_pooled_mixture'][side] = _posterior_median(logs, grid)

        # The law of total variance for a random synthetic calibration game:
        # E[Var(accuracy|game,r)] + Var(E[accuracy|game,r]). This estimates global
        # game quality, without importing the target game's idiosyncratic sigma.
        logs = _normal_log_likelihood(accuracy, pooled, between+calibration_variances.mean())
        results['global_population_game_quality'][side] = _posterior_median(logs, grid)
        results['global_population_account_5pct'][side] = .95*results['global_population_game_quality'][side]+.05*account

    return results
