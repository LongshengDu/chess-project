"""Reference-free common-center controls using per-move quality information.

The shared arithmetic-coverage fit supplies a fixed White/Black contrast. Two
predeclared alternatives replace only its common center:

* Quality midranks: solve the equal-side mean played-quality percentile = 1/2.
  The midpoint percentile gives quality ties half their probability mass, so
  its expectation under the same Maia policy is exactly 1/2. A nonincreasing
  projection removes Maia's local rating reversals before inversion.
* Severity: maximize the equal-side average conditional log probability of the
  played severity categories. The categories use the project's existing Lichess
  5/10/15 winning-percentage-loss boundaries, not reference-fitted thresholds.

Maia probabilities are interpolated only within their measured 600--2600 range.
These are estimating equations/composite likelihoods, not calibrated posteriors.
The original common 5% account anchor is applied after the new center, preserving
the production contrast and account sensitivity. No commercial/game identifiers,
population rebuilding, tuned blend, forced separation, or coach/engine calls.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression

from analysis.lichess_accuracy import move_accuracy
from analysis.player_rating.arithmetic_coverage import Rating
from analysis.player_rating.parameters import RATINGS


SIDES = ('White', 'Black')
GRID = np.asarray(RATINGS, dtype=float)
QUALITY_BOUNDARIES = np.sort([move_accuracy(100., 100.-loss) for loss in (5., 10., 15.)])


def describe():
    return {
        'joint_quality_midrank_center': 'Preserve arithmetic-coverage contrast; common center solves Maia played-quality midrank=1/2.',
        'joint_quality_severity_center': 'Preserve arithmetic-coverage contrast; common center maximizes Maia severity-category composite likelihood.',
    }


def _quality_signals(record):
    """Return each informative move's quality midrank and severity probability."""
    midranks, probabilities = [], []
    for row in record['observations']:
        q = np.asarray(row['qualities']['position'], dtype=float)
        if len(q) <= 1 or np.ptp(q) < 1e-10:
            continue
        policy = np.asarray(row['maia_probabilities'], dtype=float)
        played = row['played_index']
        midranks.append(policy @ (q < q[played])+.5*(policy @ (q == q[played])))
        # side='left' puts an exact loss threshold in its worse severity band.
        categories = np.searchsorted(QUALITY_BOUNDARIES, q, side='left')
        probabilities.append(policy @ (categories == categories[played]))
    return np.asarray(midranks), np.asarray(probabilities)


def _centers(evidence, offsets):
    """Fit a single center with both shifted ratings inside native Maia support."""
    lower = max(GRID[0]-offset for offset in offsets.values())
    upper = min(GRID[-1]-offset for offset in offsets.values())
    if lower >= upper:
        return dict.fromkeys(describe())
    centers = np.linspace(lower, upper, int(np.ceil((upper-lower)/5))+1)
    rank_curves, severity_curves = [], []
    for side, offset in offsets.items():
        ranks, probabilities = _quality_signals(evidence[side])
        if not len(ranks):
            continue
        ratings = centers+offset
        rank_curves.append(np.interp(ratings, GRID, ranks.mean(axis=0)))
        # Fractional ratings mix adjacent Maia probabilities before taking logs.
        probability = np.stack([np.interp(ratings, GRID, row) for row in probabilities])
        severity_curves.append(np.log(np.maximum(probability, np.finfo(float).tiny)).mean(axis=0))
    result = dict.fromkeys(describe())
    if not rank_curves:
        return result
    percentile = isotonic_regression(np.mean(rank_curves, axis=0), increasing=False).x
    if np.ptp(percentile) > 1e-12:
        equal = np.flatnonzero(np.isclose(percentile, .5, atol=1e-12, rtol=0))
        result['joint_quality_midrank_center'] = (float((centers[equal[0]]+centers[equal[-1]])/2)
            if len(equal) else float(np.interp(.5, percentile[::-1], centers[::-1])))
    severity = np.mean(severity_curves, axis=0)
    if np.ptp(severity) > 1e-12:
        result['joint_quality_severity_center'] = float(centers[np.argmax(severity)])
    return result


def predict(evidence, actual_ratings):
    """Return two fixed common-center decisions; references are never inputs."""
    contextual = {side: {**evidence[side], 'actual_rating': actual_ratings.get(side)} for side in SIDES}
    baseline = Rating().fit(contextual)
    components = baseline['diagnostics']['components']
    available = {side: components[side]['coverage_mean'] for side in SIDES
                 if baseline['players'][side]['unrounded_estimate'] is not None}
    if not available:
        return {name: dict.fromkeys(SIDES) for name in describe()}
    original_center = float(np.mean(list(available.values())))
    offsets = {side: value-original_center for side, value in available.items()}
    centers = _centers(evidence, offsets)
    result = {}
    for name, center in centers.items():
        estimates = {}
        for side in SIDES:
            component = components[side]
            if side not in offsets or center is None:
                estimates[side] = baseline['players'][side]['unrounded_estimate']
                continue
            quality = center+offsets[side]
            anchor, weight = component['common_account_rating'], component['account_weight']
            estimates[side] = quality if anchor is None else (1-weight)*quality+weight*anchor
        result[name] = estimates
    return result
