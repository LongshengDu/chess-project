"""One declared coarse check of the weak account-rating contribution.

The structural estimator is fixed: one-quarter arithmetic native coverage and
three-quarters competitive predictive quality. Compare account weights 2.5%,
5% and 10%; all bound a 200-Elo own-rating shift to at most 20 Elo. This is
explicit exploratory adjustment of a minor regularization assumption, not a
commercial-label optimizer. Predictions use the same weight for every player.

Component points are taken before any White/Black order constraint. First blend
the two quality opinions, then blend account Elo, then impose the color-order
constraint exactly once. An already projected pair cannot be algebraically
unblended to recover the component evidence.
"""
from __future__ import annotations

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis import universal_native_coverage
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior
from tests.analysis.universal_competitiveness import project_order, weighted_fit

ACCOUNT_WEIGHTS = (.025, .05, .10)
COVERAGE_WEIGHT = .25


def unprojected_quality(evidence, fit, calibration_cases):
    """Combine account-free component points; this function receives no ratings."""
    # Only the account-free output is used. Zero placeholders satisfy the common
    # component interface; no account blend is reversed and no pair is projected.
    coverage = universal_native_coverage.predict(
        evidence, fit, {'White': 0., 'Black': 0.}, calibration_cases)['native_coverage_mean_all']
    competitive_fit = weighted_fit(evidence, .5)
    calibration = [{'fit': weighted_fit(case['evidence'], .5)} for case in calibration_cases]
    _, variances, population, between = _population(calibration, ARGS.grid)
    own = competitive_fit['diagnostics']['curve']
    quality = {}
    for side in ('White', 'Black'):
        result = posterior(competitive_fit['players'][side]['average_accuracy'],
                           np.asarray(own['shared_accuracy']), own['likelihood']['accuracy_variance'],
                           population, between+variances.mean(), ARGS.grid)
        quality[side] = COVERAGE_WEIGHT*coverage[side]+(1-COVERAGE_WEIGHT)*float(result['mean'])
    return quality


def predict(evidence, fit, ratings, calibration_cases):
    for side in ('White', 'Black'):
        rating = ratings.get(side)
        if rating is None or not np.isfinite(rating) or not 0 <= rating <= 3200:
            raise ValueError('Account ratings must be finite and lie in[0,3200].')
    quality = unprojected_quality(evidence, fit, calibration_cases)
    return {f'universal_account_{round(1000*weight):03d}':
            project_order({side: (1-weight)*quality[side]+weight*ratings[side] for side in quality}, fit)
            for weight in ACCOUNT_WEIGHTS}
