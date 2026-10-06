"""Non-Gaussian synthetic-game accuracy likelihood, never reference calibrated.

For every other game's side and native Maia rating, independent draws from each
position's complete legal-move policy are convolved exactly after rounding move
qualities to one accuracy point. An equal-side, equal-game predictive mixture
retains skewness and all-perfect probability mass that Gaussian means discard.
The positions are fixed: this does not simulate resulting chess continuations.

The numerical approximation is explicit: quality rounding has at most 0.5 point
error per move, and likelihood observes a fixed one-point-wide mean-accuracy bin.
Only native 600--2600 ratings are supported; no synthetic policy extrapolation
is attempted. Calibration uses legal qualities/policies, never played indices,
account Elo or reference ratings. This remains an exploratory small-corpus model.
"""
from __future__ import annotations

from hashlib import blake2b

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.integrate import cumulative_trapezoid

from analysis.player_rating.bayesian_shared_curve import GRID, prior_weights


QUALITY_STEP = 1.
OBSERVATION_BIN_WIDTH = 1.
METHODS = ('predictive_accuracy_fft', 'predictive_accuracy_fft_account_5pct')
_DISTRIBUTION_CACHE = {}


def _quantized_positions(record):
    """Return full policies and rounded legal qualities, excluding forced moves."""
    positions = []
    for observation in record['observations']:
        quality = np.asarray(observation['qualities']['position'], dtype=np.float64)
        if len(quality) <= 1:
            continue
        policy = np.asarray(observation['maia_probabilities'], dtype=np.float64)
        if (quality.ndim != 1 or policy.shape != (len(GRID), len(quality))
                or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 100))
                or not np.isfinite(policy).all() or np.any(policy < 0)
                or not np.allclose(policy.sum(axis=1), 1., atol=1e-6, rtol=0)):
            raise ValueError('Complete bounded qualities and normalized legal policies are required.')
        policy = policy/policy.sum(axis=1, keepdims=True)
        bins = np.floor(quality/QUALITY_STEP+.5).astype(np.int64)
        positions.append((bins, policy))
    return positions


def _side_distribution(positions):
    """FFT polynomial product gives the distribution of summed move qualities."""
    if not positions:
        return None
    count = len(positions)
    bins_per_move = int(round(100/QUALITY_STEP))+1
    length = (bins_per_move-1)*count+1
    fft_length = next_fast_len(length)
    policies = np.zeros((len(GRID), count, bins_per_move))
    for position, (bins, probabilities) in enumerate(positions):
        for rating in range(len(GRID)):
            policies[rating, position] = np.bincount(
                bins, weights=probabilities[rating], minlength=bins_per_move)
    distributions = []
    for policies_at_rating in policies:
        transformed = rfft(policies_at_rating, n=fft_length, axis=1)
        distribution = irfft(np.prod(transformed, axis=0), n=fft_length)[:length]
        # FFT cancellation introduces tiny negative values; clip and renormalize.
        distribution = np.maximum(distribution, 0.)
        distributions.append(distribution/distribution.sum())
    return {'moves': count, 'pmf': np.asarray(distributions)}


def _game_distributions(evidence):
    """Content cache excludes played indices, account ratings and all metadata."""
    positions = [_quantized_positions(evidence[side]) for side in ('White', 'Black')]
    digest = blake2b(digest_size=20)
    digest.update(np.asarray([QUALITY_STEP], dtype=np.float64).tobytes())
    for side in positions:
        digest.update(np.asarray([len(side)], dtype=np.int64).tobytes())
        for bins, policy in side:
            digest.update(np.asarray([len(bins)], dtype=np.int64).tobytes())
            digest.update(bins.tobytes())
            digest.update(policy.tobytes())
    key = digest.digest()
    if key not in _DISTRIBUTION_CACHE:
        _DISTRIBUTION_CACHE[key] = [_side_distribution(side) for side in positions if side]
    return _DISTRIBUTION_CACHE[key]


def _bin_probabilities(distribution, accuracy):
    """Bin mass uses the same mean-accuracy window across different game lengths."""
    pmf, count = distribution['pmf'], distribution['moves']
    centers = np.arange(pmf.shape[1])*QUALITY_STEP/count
    half = OBSERVATION_BIN_WIDTH/2
    included = (centers >= accuracy-half) & (centers < accuracy+half)
    return pmf[:, included].sum(axis=1)


def _population_likelihood(calibration_cases, accuracy):
    games = []
    for case in calibration_cases:
        sides = _game_distributions(case['evidence'])
        if sides:
            games.append(np.mean([_bin_probabilities(side, accuracy) for side in sides], axis=0))
    if not games:
        raise ValueError('Calibration games require non-forced legal-move evidence.')
    return np.mean(games, axis=0)


def _median(likelihood):
    grid = np.arange(GRID[0], GRID[-1]+1, 5.)
    log_likelihood = np.interp(grid, GRID, np.log(np.maximum(likelihood, 1e-300)))
    density = np.exp(log_likelihood-log_likelihood.max())*prior_weights(grid)
    cdf = cumulative_trapezoid(density, grid, initial=0)
    return float(np.interp(.5, cdf/cdf[-1], grid))


def predict(evidence, fit, ratings, calibration_cases, *, edge_only=True):
    """Predict from other-game synthetic policies, preserving in-curve estimates."""
    del evidence
    curves = fit['diagnostics']['curve']['monotone_expected_accuracy']
    result = {name: {} for name in METHODS}
    for side in ('White', 'Black'):
        player = fit['players'][side]
        accuracy, estimate = player['average_accuracy'], player['estimate']
        if accuracy is None or estimate is None or (edge_only and curves[0] <= accuracy <= curves[-1]):
            for values in result.values():
                values[side] = estimate
            continue
        base = _median(_population_likelihood(calibration_cases, accuracy))
        account = float(ratings[side])
        if not np.isfinite(account):
            raise ValueError('A finite actual account rating is required by the account blend.')
        result['predictive_accuracy_fft'][side] = base
        result['predictive_accuracy_fft_account_5pct'][side] = .95*base+.05*account
    return result
