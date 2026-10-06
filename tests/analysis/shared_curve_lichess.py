"""Experimental shared curve using probability-weighted Lichess game accuracy.

The top-99% cutoff is explicit; production curve fitting, tails, prior and
posterior are reused. Actual-position volatility weights stay
fixed. The reference harmonic component is the reciprocal of the expected
reciprocal sum, not the exact expectation of a randomly sampled game's score.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.optimize import isotonic_regression

from analysis.player_rating.bayesian_shared_curve import (
    Args, GRID, SharedCurve, curve_posterior, prior_density, prior_weights,
    top_probability_policy,
)
from analysis.player_rating.evidence import validate_evidence

METHOD = 'experimental_lichess_shared_curve'
ARGS = Args(top_probability=.99, accuracy_sigma_scale=1.)


def side_moments(record, *, args=None):
    """Aggregate retained alternatives with both Lichess mean components.

    With h(q)=1/max(1,q), alpha_i=w_i/sum(w), B=sum_i E[h_i],
    the mean is (sum_i alpha_i E[q_i] + n/B)/2. Delta-method variance
    includes Cov(q,h) because both components use the same selected move.
    All positions, including one-legal-move positions, are included.
    """
    args = args or ARGS
    observations = record['observations']
    if not observations:
        return None
    weights = np.asarray([row['weight'] for row in observations], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Lichess volatility weights must be finite and positive.')
    weights /= weights.sum()
    means, reciprocals, variance_q, variance_h, covariance = [], [], [], [], []
    observed, masses, sizes, played_retained = [], [], [], []
    forced = 0
    for row in observations:
        q = np.asarray(row['qualities']['position'], dtype=float)
        policy = np.asarray(row['maia_probabilities'], dtype=float)
        index = row['played_index']
        if (q.ndim != 1 or not len(q) or isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer)) or not 0 <= index < len(q)
                or not np.isfinite(q).all() or np.any((q < 0) | (q > 100))
                or policy.shape != (len(GRID), len(q))):
            raise ValueError('Complete bounded move qualities and a played index are required.')
        retained = top_probability_policy(policy, args.top_probability)
        h = 1/np.maximum(1., q)
        mean, reciprocal = retained @ q, retained @ h
        centered_q, centered_h = q-mean[:, None], h-reciprocal[:, None]
        means.append(mean)
        reciprocals.append(reciprocal)
        variance_q.append(np.sum(retained*centered_q**2, axis=1))
        variance_h.append(np.sum(retained*centered_h**2, axis=1))
        covariance.append(np.sum(retained*centered_q*centered_h, axis=1))
        observed.append(q[index])
        mask = retained > 0
        masses.append(np.sum(np.where(mask, policy, 0.), axis=1))
        sizes.append(mask.sum(axis=1))
        played_retained.append(mask[:, index])
        forced += len(q) == 1
    n = len(observations)
    means, reciprocals = np.asarray(means), np.asarray(reciprocals)
    weighted = weights @ means
    reciprocal_sum = reciprocals.sum(axis=0)
    harmonic = n/reciprocal_sum
    derivative = n/reciprocal_sum**2
    alpha = weights[:, None]
    variance = .25*np.sum(alpha**2*np.asarray(variance_q)
                         + derivative**2*np.asarray(variance_h)
                         - 2*alpha*derivative*np.asarray(covariance), axis=0)
    observed = np.asarray(observed)
    played_weighted = float(weights @ observed)
    played_harmonic = float(n/np.sum(1/np.maximum(1., observed)))
    return {
        'mean': np.clip((weighted+harmonic)/2, 0., 100.),
        'mean_variance': np.maximum(0., variance),
        'accuracy': (played_weighted+played_harmonic)/2, 'moves': n,
        'weighted_mean': weighted, 'harmonic_mean': harmonic,
        'played_weighted_mean': played_weighted,
        'played_harmonic_mean': played_harmonic,
        'selection': {
            'forced_positions_removed': 0, 'forced_positions_included': forced,
            'positions_used': n, 'mean_retained_mass': np.mean(masses, axis=0).tolist(),
            'minimum_retained_mass': float(np.min(masses)),
            'mean_retained_moves': np.mean(sizes, axis=0).tolist(),
            'played_move_retained_fraction': np.mean(played_retained, axis=0).tolist(),
        },
    }


def summarize(evidence, *, args=None):
    """Experimental fit; account ratings and commercial labels never enter it."""
    args = args or ARGS
    evidence = validate_evidence(evidence)
    moments = {side: side_moments(evidence[side], args=args) for side in ('White', 'Black')}
    available = [moment for moment in moments.values() if moment is not None]
    expected = np.mean([m['mean'] for m in available], axis=0) if available else np.zeros(len(GRID))
    projected = np.clip(isotonic_regression(expected).x, 0., 100.)
    curve = SharedCurve(projected)(args.grid)
    variance = float(np.mean([m['mean_variance'].mean() for m in available])) if available else 0.
    identifiable = bool(available and np.ptp(curve) > 1e-10)
    players, densities, components = {}, {}, {}
    for side, moment in moments.items():
        if moment is None or not identifiable:
            posterior = {'estimate': None, 'unrounded_estimate': None,
                         'conditional_interval': list(args.rating_range)}
            densities[side] = None
        else:
            posterior = curve_posterior(moment['accuracy'], curve, variance, args=args)
            densities[side] = posterior.pop('posterior_density')
        low, high = posterior['conditional_interval']
        players[side] = {
            'estimate': posterior['estimate'], 'unrounded_estimate': posterior['unrounded_estimate'],
            'interval': [int(np.floor(low)), int(np.ceil(high))],
            'average_accuracy': moment['accuracy'] if moment else None,
            'moves_used': moment['moves'] if moment else 0,
            'identifiable': moment is not None and identifiable, 'method': METHOD,
        }
        components[side] = {
            'expected_accuracy': moment['mean'].tolist(),
            'volatility_weighted_mean': moment['weighted_mean'].tolist(),
            'probability_weighted_harmonic_mean': moment['harmonic_mean'].tolist(),
            'delta_variance': moment['mean_variance'].tolist(),
            'played_volatility_weighted_mean': moment['played_weighted_mean'],
            'played_harmonic_mean': moment['played_harmonic_mean'],
        } if moment else None
    diagnostics = {
        'rating_grid': GRID.tolist(), 'fine_ratings': args.grid.tolist(),
        'maia_expected_accuracy': expected.tolist(), 'monotone_expected_accuracy': projected.tolist(),
        'shared_accuracy': curve.tolist(), 'shared_mean_variance': variance,
        'likelihood': {'kind': 'gaussian_accuracy', 'variance_method': 'first_order_delta',
                       'sigma_scale': args.accuracy_sigma_scale,
                       'accuracy_sigma': float(np.sqrt(variance)*args.accuracy_sigma_scale),
                       'accuracy_variance': variance*args.accuracy_sigma_scale**2},
        'top_probability': args.top_probability,
        'selection': {side: moment['selection'] if moment else {'positions_used': 0}
                      for side, moment in moments.items()},
        'largest_raw_curve_reversal': float(max(0., -np.diff(expected).min())),
        'largest_isotonic_adjustment': float(np.max(np.abs(projected-expected))),
        'zero_variance_floor_used': bool(identifiable and variance*args.accuracy_sigma_scale**2 < 1e-12),
        'prior_weights': prior_weights(args.grid, args=args).tolist(),
        'prior_density': prior_density(args.grid, args=args).tolist(),
        'posterior_densities': densities, 'components': components,
    }
    return {
        'players': players, 'name': 'Experimental Lichess shared-curve fit', 'method_id': METHOD,
        'parameters': asdict(args), 'central_interval': args.central_interval,
        'rating_range': list(args.rating_range), 'point_estimator': 'posterior_median',
        'account_ratings_used': False, 'diagnostics': {'curve': diagnostics},
        'aggregation': 'Mean of volatility-weighted arithmetic and probability-weighted harmonic accuracy; all positions included; actual-position volatility weights fixed.',
        'approximation': 'Reference harmonic is n/sum(E[1/max(1,q)]), not E[n/sum(1/max(1,q))]. Variance uses the first-order delta method with covariance, conditional independence and sigma scale 1 by default.',
    }
