"""Bayesian shared-curve fit from Maia's full legal-move distribution.

Both players share an isotonic accuracy curve, bounded exponential tails and a
tapered prior. Method arguments live here; account/reference ratings never enter
inference. The central interval is conditional posterior mass, not calibrated error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression

from analysis.player_rating.interface import PlayerRating
from analysis.player_rating.parameters import RATINGS

METHOD = __name__.rsplit('.', 1)[-1]
NAME = 'Bayesian shared-curve fit'
VERSION = 8
GRID = np.asarray(RATINGS, dtype=float)


@dataclass(frozen=True)
class Args:
    """Method-local support, prior, accuracy-noise multiplier and displayed mass."""

    rating_range: tuple[float, float] = (0., 3200.)
    flat_prior_range: tuple[float, float] = (800., 2400.)
    prior_range: tuple[float, float] = (200., 3000.)
    central_interval: float = .20
    accuracy_sigma_scale: float = 1.
    top_probability: float = 1.

    def __post_init__(self):
        for name in ('rating_range', 'flat_prior_range', 'prior_range'):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list)) or len(values) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in values)):
                raise ValueError(f'{name} must contain two finite rating bounds.')
            object.__setattr__(self, name, tuple(map(float, values)))
        low, high = self.rating_range
        start, end = self.flat_prior_range
        prior_low, prior_high = self.prior_range
        if low != math.floor(low) or high != math.floor(high):
            raise ValueError('The rating range must use whole Elo bounds for integer display.')
        if not low <= prior_low < start <= end < prior_high <= high:
            raise ValueError('The flat prior range must lie strictly inside prior_range, within rating_range.')
        p = self.central_interval
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 < p < 1:
            raise ValueError('central_interval must be a finite probability strictly between 0 and 1.')
        scale = self.accuracy_sigma_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('accuracy_sigma_scale must be a finite positive multiplier.')
        probability = self.top_probability
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 < probability <= 1):
            raise ValueError('top_probability must be a finite probability in (0, 1].')

    @property
    def grid(self):
        """Include both support edges, with a numerical spacing of at most 5 Elo."""
        low, high = self.rating_range
        return np.linspace(low, high, max(3, math.ceil((high-low)/5)+1))


ARGS = Args()


class SharedCurve:
    """Measured PCHIP with value/slope-matched exponential tails bounded to 0–100."""

    def __init__(self, accuracies):
        self.accuracies = np.asarray(accuracies, dtype=float)
        if (self.accuracies.shape != GRID.shape or not np.isfinite(self.accuracies).all()
                or np.any((self.accuracies < 0) | (self.accuracies > 100))
                or np.any(np.diff(self.accuracies) < -1e-10)):
            raise ValueError('Shared accuracy knots must be finite, bounded and nondecreasing.')
        self.measured = PchipInterpolator(GRID, self.accuracies, extrapolate=False)
        self.slopes = np.maximum(0., self.measured.derivative()(GRID[[0, -1]]))

    def __call__(self, ratings):
        ratings = np.asarray(ratings, dtype=float)
        values = self.measured(np.clip(ratings, GRID[0], GRID[-1]))
        left, right = self.accuracies[[0, -1]]
        lower = left * np.exp(np.clip(self.slopes[0]*(ratings-GRID[0])/max(left, 1e-12), -700, 0))
        upper = 100 - (100-right) * np.exp(np.clip(-self.slopes[1]*(ratings-GRID[-1])/max(100-right, 1e-12), -700, 0))
        values = np.where(ratings < GRID[0], lower, np.where(ratings > GRID[-1], upper, values))
        return np.clip(values, 0., 100.)


def prior_weights(ratings, *, args=None):
    """Fourth-power shoulders with smooth joins to zero and the flat plateau."""
    args = args or ARGS
    ratings = np.asarray(ratings, dtype=float)
    low, high = args.prior_range
    start, end = args.flat_prior_range

    def ramp(t):
        t = np.clip(t, 0., 1.)
        return t**4/(t**4+(1-t)**4)

    return ramp((ratings-low)/(start-low))*ramp((high-ratings)/(high-end))


def prior_density(ratings, *, args=None):
    """Normalize analytically: each symmetric shoulder has mean weight 1/2."""
    args = args or ARGS
    low, high = args.prior_range
    start, end = args.flat_prior_range
    normalizer = (high-low+end-start)/2
    return prior_weights(ratings, args=args)/normalizer


class Rating(PlayerRating):
    """The filename-selected Bayesian shared-curve estimator."""

    name = NAME
    version = VERSION

    def __init__(self, args=None):
        self.args = args or ARGS
        if not isinstance(self.args, Args):
            raise TypeError('Bayesian shared-curve arguments must be an Args instance.')

    @property
    def parameters(self):
        return asdict(self.args)

    def fit(self, evidence):
        return summarize(evidence, args=self.args)


def top_probability_policy(policy, probability):
    """Retain the smallest probability prefix above the cutoff, including ties."""
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not np.isfinite(probability) or not 0 < probability <= 1):
        raise ValueError('Invalid retained probability.')
    p = np.asarray(policy, dtype=float)
    if (p.ndim != 2 or not p.shape[0] or not p.shape[1] or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('Normalized nonnegative policy rows are required.')
    p = p / p.sum(axis=1, keepdims=True)
    if probability == 1:
        return p
    ordered = np.sort(p, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    cumulative[np.isclose(cumulative, probability, atol=1e-12, rtol=0)] = probability
    boundary = np.minimum(np.sum(cumulative <= probability, axis=1), p.shape[1]-1)
    cutoff = ordered[np.arange(len(p)), boundary]
    retained = np.where(p >= cutoff[:, None], p, 0.)
    return retained / retained.sum(axis=1, keepdims=True)


def side_moments(record, view='position', *, args=None):
    """Conditional Maia moments and actual accuracy, excluding legally forced moves."""
    args = args or ARGS
    observations = record['observations']
    if not observations:
        return None
    means, variances, observed = [], [], []
    masses, sizes, played_retained = [], [], []
    forced = 0
    for row in observations:
        q = np.asarray(row['qualities'][view], dtype=float)
        policy = np.asarray(row['maia_probabilities'], dtype=float)
        index = row['played_index']
        if (q.ndim != 1 or len(q) == 0 or isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer)) or not 0 <= index < len(q)):
            raise ValueError('Played index must identify one legal move.')
        if (policy.shape != (len(GRID), len(q)) or not np.isfinite(q).all()
                or not np.isfinite(policy).all() or np.any(policy < 0)
                or np.any((q < 0) | (q > 100))
                or not np.allclose(policy.sum(axis=1), 1., atol=1e-6, rtol=0)):
            raise ValueError('Complete normalized Maia policies and bounded qualities are required.')
        if len(q) == 1:
            forced += 1
            continue
        retained = top_probability_policy(policy, args.top_probability)
        mask = retained > 0
        masses.append(np.sum(np.where(mask, policy, 0.), axis=1))
        sizes.append(mask.sum(axis=1))
        played_retained.append(mask[:, index])
        mean = retained @ q
        means.append(mean)
        variances.append(np.maximum(0., retained @ (q*q) - mean*mean))
        observed.append(q[index])
    n = len(observed)
    if not n:
        return None
    return {'mean': np.mean(means, axis=0),
            'mean_variance': np.sum(variances, axis=0)/(n*n),
            'accuracy': float(np.mean(observed)), 'moves': n,
            'selection': {'forced_positions_removed': forced, 'positions_used': n,
                          'mean_retained_mass': np.mean(masses, axis=0).tolist(),
                          'minimum_retained_mass': float(np.min(masses)),
                          'mean_retained_moves': np.mean(sizes, axis=0).tolist(),
                          'played_move_retained_fraction': np.mean(played_retained, axis=0).tolist()}}


def curve_posterior(accuracy, curve, variance, *, args=None):
    """Original Gaussian accuracy likelihood, with a shared sigma multiplier."""
    args = args or ARGS
    grid = args.grid
    curve = np.asarray(curve, dtype=float)
    if (curve.shape != grid.shape or not np.isfinite(curve).all()
            or np.any((curve < 0) | (curve > 100))
            or not np.isfinite(accuracy) or not 0 <= accuracy <= 100
            or not np.isfinite(variance) or variance < 0):
        raise ValueError('Invalid shared-curve posterior inputs.')
    prior = prior_density(grid, args=args)
    interior = prior > 0
    effective_variance = variance * args.accuracy_sigma_scale**2
    logs = -.5*(accuracy-curve)**2/max(effective_variance, 1e-12)
    logs[interior] += np.log(prior[interior])
    density = np.zeros_like(grid)
    density[interior] = np.exp(logs[interior]-logs[interior].max())
    cdf = cumulative_trapezoid(density, grid, initial=0)
    density /= cdf[-1]
    cdf /= cdf[-1]
    tail = (1-args.central_interval)/2
    quantiles = np.interp([tail, .5, 1-tail], cdf, grid)
    return {'estimate': int(round(quantiles[1])),
            'unrounded_estimate': float(quantiles[1]),
            'conditional_interval': quantiles[[0, 2]].tolist(),
            'posterior_density': density.tolist()}


def fit_pair(white, black, view='position', *, args=None):
    """Fit two players using one common game-accuracy scale, without labels.

    A single observed side supplies the curve when a PGN has only one side's
    moves. Unobserved players and wholly flat curves receive no point estimate.
    The shared prior never uses either player's account or reference rating.
    """
    args = args or ARGS
    moments = [side_moments(record, view, args=args) for record in (white, black)]
    selection = {
        side: moment['selection'] if moment else {
            'forced_positions_removed': sum(len(row['qualities'][view]) == 1 for row in record['observations']),
            'positions_used': 0}
        for side, record, moment in zip(('White', 'Black'), (white, black), moments, strict=True)}
    available = [m for m in moments if m is not None]
    expected = np.mean([m['mean'] for m in available], axis=0) if available else np.zeros(len(GRID))
    # Floating-point policy sums can put an all-perfect curve just above 100.
    projected = np.clip(isotonic_regression(expected).x, 0., 100.)
    curve = SharedCurve(projected)(args.grid)
    variance = float(np.mean([m['mean_variance'].mean() for m in available])) if available else 0.
    identifiable = bool(available and np.ptp(curve) > 1e-10)
    players, densities = [], {}
    for side, moment in zip(('White', 'Black'), moments, strict=True):
        if moment is None or not identifiable:
            player = {'estimate': None, 'unrounded_estimate': None,
                      'conditional_interval': list(args.rating_range)}
            densities[side] = None
        else:
            player = curve_posterior(moment['accuracy'], curve, variance, args=args)
            densities[side] = player.pop('posterior_density')
        players.append(dict(player, average_accuracy=moment['accuracy'] if moment else None,
                            moves=moment['moves'] if moment else 0))
    margin = (players[0]['estimate']-players[1]['estimate']
              if all(p['estimate'] is not None for p in players) else None)
    return {'players': players, 'rating_grid': GRID.tolist(), 'maia_expected_accuracy': expected.tolist(),
            'monotone_expected_accuracy': projected.tolist(), 'shared_mean_variance': variance,
            'identifiable': identifiable, 'white_minus_black': margin,
            'fine_ratings': args.grid.tolist(), 'shared_accuracy': curve.tolist(),
            'likelihood': {'kind': 'gaussian_accuracy', 'sigma_scale': args.accuracy_sigma_scale,
                           'accuracy_sigma': float(np.sqrt(variance)*args.accuracy_sigma_scale),
                           'accuracy_variance': variance*args.accuracy_sigma_scale**2},
            'top_probability': args.top_probability, 'selection': selection,
            'largest_raw_curve_reversal': float(max(0., -np.diff(expected).min())),
            'largest_isotonic_adjustment': float(np.max(np.abs(projected-expected))),
            'zero_variance_floor_used': bool(identifiable and variance*args.accuracy_sigma_scale**2 < 1e-12),
            'prior_weights': prior_weights(args.grid, args=args).tolist(),
            'prior_density': prior_density(args.grid, args=args).tolist(),
            'posterior_densities': densities}


def summarize(evidence, *, args=None):
    """Return the common played-Elo schema used by the coach and web application."""
    args = args or ARGS
    fit = fit_pair(evidence['White'], evidence['Black'], args=args)
    players = {}
    for side, player in zip(('White', 'Black'), fit['players'], strict=True):
        estimate = player['estimate']
        low, high = player['conditional_interval']
        interval = [int(np.floor(low)), int(np.ceil(high))]
        players[side] = {
            'estimate': estimate,
            'unrounded_estimate': player['unrounded_estimate'],
            'uncertainty': int(np.ceil(max(estimate-interval[0], interval[1]-estimate)/10)*10)
                           if estimate is not None else None,
            'interval': interval, 'moves_used': player['moves'],
            'average_accuracy': player['average_accuracy'],
            'identifiable': estimate is not None,
            'at_rating_limit': estimate in args.rating_range,
            'interval_touches_limit': [interval[0] <= args.rating_range[0], interval[1] >= args.rating_range[1]],
            'method': METHOD,
        }
    return {'players': players, 'name': NAME, 'method_id': METHOD,
            'parameters': asdict(args),
            'central_interval': args.central_interval, 'rating_range': list(args.rating_range),
            'point_estimator': 'posterior_median',
            'prior': {'kind': 'symmetric_fourth_power', 'minimum': args.prior_range[0], 'maximum': args.prior_range[1],
                      'flat_range': list(args.flat_prior_range)},
            'interval_scope': f'Central {args.central_interval:.0%} of the conditional shared-curve posterior within {args.rating_range[0]:g}–{args.rating_range[1]:g}; shared Maia-derived accuracy sigma multiplied by {args.accuracy_sigma_scale:g}. The multiplier is a width assumption, not calibrated rating error or a fixed Elo bound. Excludes serial dependence, engine uncertainty, Maia model error and extrapolation uncertainty.',
            'method': ('Full legal-move Maia probability distribution' if args.top_probability == 1 else
                       f'Top {args.top_probability:.0%} cumulative Maia probability, renormalized with cutoff ties') +
                      '; forced positions excluded; equal-side arithmetic mean curve; isotonic PCHIP; bounded exponential tails; shared Gaussian accuracy likelihood with a sigma multiplier; flat prior with fourth-power shoulders; posterior median.',
            'diagnostics': {'curve': {key: value for key, value in fit.items() if key != 'players'}},
            'account_ratings_used': False}
