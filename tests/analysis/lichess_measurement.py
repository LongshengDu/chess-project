"""Research-only arithmetic and Lichess aggregate accuracy measurements.

G(q) = (sum(alpha_i*q_i) + n/sum(1/max(1,q_i)))/2 uses the saved
Lichess volatility weights. They stay fixed at the actual game's positions;
draws are independent choices on this fixed path, not alternative legal games.

The plug-in G(E[Q]) differs from E[G(Q)]. The historical reciprocal plug-in
instead substitutes E[1/max(1,Q)] before taking the outer reciprocal. Both
plug-in variances use a first-order delta approximation. The draw method
integrates G and its squared deviation with scrambled Sobol quadrature, giving
conditional aggregate variance, not the variance of the numerical mean. Four
independent scrambles estimate numerical error separately. All candidates use
the entire normalized legal-move policy; no rating/reference labels enter.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import qmc

from analysis.player_rating.bayesian_shared_curve import SharedCurve, top_probability_policy
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.uncertainty_measurement import ARGS, measure as arithmetic_measure


METHODS = ('arithmetic', 'lichess_plugin', 'lichess_policy_expectation', 'lichess_reciprocal_plugin', 'volatility_weighted')
SIDES = ('White', 'Black')
REPLICATES = 4


def describe():
    return {
        'arithmetic': 'Unchanged arithmetic measurement: full legal policies, forced positions excluded, exact conditional variance of the mean.',
        'lichess_plugin': 'Lichess aggregate of expected move accuracies G(E[Q]); first-order delta variance; all positions and fixed actual-path volatility weights.',
        'lichess_policy_expectation': 'Expected Lichess aggregate E[G(Q)] and its conditional variance from independent policy draws on the fixed reached positions; scrambled Sobol integration with separate replicate error.',
        'lichess_reciprocal_plugin': 'Historical reciprocal plug-in: weighted E[Q] combined with n/sum(E[1/max(1,Q)])); first-order delta variance including covariance.',
        'volatility_weighted': 'Volatility-weighted arithmetic accuracy only, with no harmonic component; exact conditional mean and variance; all positions and fixed actual-path weights.',
    }


def aggregate(qualities, weights):
    """Lichess aggregate over the last axis, including its harmonic floor of 1."""
    q, weights = np.asarray(qualities, dtype=float), np.asarray(weights, dtype=float)
    if (q.ndim < 1 or not q.shape[-1] or weights.shape != (q.shape[-1],)
            or not np.isfinite(q).all() or np.any((q < 0) | (q > 100))
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('Bounded move qualities and positive matching volatility weights are required.')
    alpha = weights/weights.sum()
    return .5*(q @ alpha + q.shape[-1]/np.sum(1/np.maximum(1., q), axis=-1))


def _rows(record):
    rows, weights, observed = [], [], []
    for row in record['observations']:
        q = np.asarray(row['qualities']['position'], dtype=float)
        index, weight = row['played_index'], row['weight']
        if (q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any((q < 0) | (q > 100))
                or isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(q)
                or isinstance(weight, bool) or not isinstance(weight, (float, int))
                or not math.isfinite(weight) or weight <= 0):
            raise ValueError('Complete bounded qualities, a played index and positive weight are required.')
        policy = top_probability_policy(row['maia_probabilities'], 1.)
        if policy.shape != (len(RATINGS), len(q)):
            raise ValueError('Complete normalized policies at every Maia rating are required.')
        # Coupling by quality preserves the law and is invariant to legal-list order.
        order = np.argsort(q, kind='stable')
        rows.append((q[order], policy[:, order]))
        weights.append(weight)
        observed.append(q[index])
    return rows, np.asarray(weights, dtype=float), np.asarray(observed, dtype=float)


def _policy_moments(rows):
    means, reciprocal, variance, reciprocal_variance, covariance = [], [], [], [], []
    for q, policy in rows:
        h = 1/np.maximum(1., q)
        mean, rec = policy @ q, policy @ h
        centered_q, centered_h = q-mean[:, None], h-rec[:, None]
        means.append(mean)
        reciprocal.append(rec)
        variance.append(np.sum(policy*centered_q**2, axis=1))
        reciprocal_variance.append(np.sum(policy*centered_h**2, axis=1))
        covariance.append(np.sum(policy*centered_q*centered_h, axis=1))
    return tuple(np.asarray(value) for value in (means, reciprocal, variance, reciprocal_variance, covariance))


def _sample_moments(rows, alpha, samples, seed):
    """Common-random-number Sobol integrations, with independent replicate errors."""
    per_rep = samples//REPLICATES
    means, seconds, weighted_means, harmonic_means = [], [], [], []
    informative_count = sum(len(q) > 1 for q, _ in rows)
    joint_means, joint_covariances = [], []
    for replicate in range(REPLICATES):
        sampler = qmc.Sobol(d=len(rows), scramble=True, seed=seed+replicate)
        uniforms = sampler.random_base2(int(math.log2(per_rep)))
        weighted = np.zeros((len(RATINGS), per_rep))
        reciprocal = np.zeros_like(weighted)
        arithmetic_sum = np.zeros_like(weighted)
        for index, (q, policy) in enumerate(rows):
            cdf = np.cumsum(policy, axis=1)
            cdf[:, -1] = 1.
            draws = np.stack([q[np.searchsorted(prob, uniforms[:, index], side='right')] for prob in cdf])
            weighted += alpha[index]*draws
            reciprocal += 1/np.maximum(1., draws)
            if len(q) > 1:
                arithmetic_sum += draws
        harmonic = len(rows)/reciprocal
        values = .5*(weighted+harmonic)
        means.append(values.mean(axis=1))
        seconds.append((values*values).mean(axis=1))
        weighted_means.append(weighted.mean(axis=1))
        harmonic_means.append(harmonic.mean(axis=1))
        if informative_count:
            paired = np.stack((arithmetic_sum/informative_count, values), axis=-1)
            paired_mean = paired.mean(axis=1)
            centered = paired-paired_mean[:, None, :]
            joint_means.append(paired_mean)
            joint_covariances.append(np.einsum('rsi,rsj->rij', centered, centered)/per_rep)
    means, seconds = np.asarray(means), np.asarray(seconds)
    mean = means.mean(axis=0)
    variance = np.maximum(0., seconds.mean(axis=0)-mean*mean)
    # Quadrature variance is an integral moment, not an IID sample-variance estimator.
    variance_replicates = np.maximum(0., seconds-means*means)
    joint = None
    if informative_count:
        joint_means = np.asarray(joint_means)
        joint_mean = joint_means.mean(axis=0)
        centered = joint_means-joint_mean
        # Within-scramble covariance plus between-scramble mean variation is
        # the covariance of all draws. Gram products preserve PSD numerically.
        joint_covariance = np.mean(joint_covariances, axis=0)+np.einsum('kri,krj->rij', centered, centered)/REPLICATES
        joint = {'mean': joint_mean, 'covariance': joint_covariance,
                 'replicate_means': joint_means,
                 'numerical_mean_se': joint_means.std(axis=0, ddof=1)/math.sqrt(REPLICATES),
                 'feature_order': ['arithmetic_without_forced', 'lichess_all_positions'],
                 'covariance_method': 'population covariance of paired draws; no independent-marginal substitution'}
    return {'mean': mean, 'variance': variance, 'replicate_means': means, 'joint': joint,
            'numerical_mean_se': means.std(axis=0, ddof=1)/math.sqrt(REPLICATES),
            'numerical_variance_se': variance_replicates.std(axis=0, ddof=1)/math.sqrt(REPLICATES),
            'weighted_mean': np.mean(weighted_means, axis=0),
            'harmonic_mean': np.mean(harmonic_means, axis=0)}


def side_moments(record, method, *, samples=16384, seed=0):
    """One player's matching observed aggregate, policy expectation and variance."""
    if method not in METHODS[1:]:
        raise ValueError('Choose a Lichess or volatility-weighted measurement method.')
    _validate_sampling(samples, seed)
    rows, weights, observed = _rows(record)
    if not rows:
        return None
    n, alpha = len(rows), weights/weights.sum()
    means, reciprocal, variance_q, variance_h, covariance = _policy_moments(rows)
    weighted = alpha @ means
    reciprocal_sum = reciprocal.sum(axis=0)
    reciprocal_harmonic = n/reciprocal_sum
    plugin_reciprocal_sum = np.sum(1/np.maximum(1., means), axis=0)
    plugin_harmonic = n/plugin_reciprocal_sum
    if method == 'lichess_policy_expectation':
        measured = _sample_moments(rows, alpha, samples, seed)
        if measured['joint'] is not None:
            informative = np.asarray([len(q) > 1 for q, _ in rows])
            count = int(informative.sum())
            measured['joint'].update(
                observed=np.asarray([observed[informative].mean(), aggregate(observed, weights)]),
                exact_arithmetic_policy_mean=means[informative].mean(axis=0),
                exact_arithmetic_policy_variance=variance_q[informative].sum(axis=0)/count**2,
                arithmetic_moves=count)
    elif method == 'lichess_plugin':
        # At the harmonic-floor kink mu=1 use its left derivative (zero).
        derivative = .5*(alpha[:, None]+n/plugin_reciprocal_sum**2 *
                         np.where(means > 1., 1/np.maximum(1., means)**2, 0.))
        measured = {'mean': .5*(weighted+plugin_harmonic),
                    'variance': np.sum(derivative**2*variance_q, axis=0),
                    'weighted_mean': weighted, 'harmonic_mean': plugin_harmonic}
    elif method == 'lichess_reciprocal_plugin':
        derivative = n/reciprocal_sum**2
        measured = {'mean': .5*(weighted+reciprocal_harmonic),
                    'variance': .25*np.sum(alpha[:, None]**2*variance_q+derivative**2*variance_h
                                           -2*alpha[:, None]*derivative*covariance, axis=0),
                    'weighted_mean': weighted, 'harmonic_mean': reciprocal_harmonic}
    else:
        measured = {'mean': weighted, 'variance': alpha**2 @ variance_q,
                    'weighted_mean': weighted, 'harmonic_mean': None}
    measured.update(accuracy=float(alpha @ observed if method == 'volatility_weighted' else aggregate(observed, weights)),
                    moves=n, effective_moves=float(1/np.sum(alpha**2)),
                    played_weighted_mean=float(alpha @ observed),
                    played_harmonic_mean=float(n/np.sum(1/np.maximum(1., observed))),
                    plugin_mean=.5*(weighted+plugin_harmonic),
                    reciprocal_plugin_mean=.5*(weighted+reciprocal_harmonic),
                    forced_positions_included=sum(len(q) == 1 for q, _ in rows))
    measured['mean'] = np.clip(measured['mean'], 0., 100.)
    measured['variance'] = np.maximum(0., measured['variance'])
    return measured


def _validate_sampling(samples, seed):
    if (type(samples) is not int or samples < REPLICATES*2 or samples % REPLICATES
            or (samples//REPLICATES) & (samples//REPLICATES-1)):
        raise ValueError('Total samples must be four times a power of two, with at least two points per replicate.')
    if type(seed) is not int or seed < 0:
        raise ValueError('The common sampling seed must be a nonnegative integer.')


def measure(evidence, method='arithmetic', *, samples=16384, seed=0):
    """Return the production-compatible shared-curve and per-side moment shape.

    Account values and reference metadata are not read. The two Lichess
    plug-ins share the sampled method's observed statistic and fixed weights.
    Population contexts must be remeasured with the same selected method.
    """
    if method not in METHODS:
        raise ValueError('Unknown accuracy measurement method.')
    _validate_sampling(samples, seed)
    if method == 'arithmetic':
        result = arithmetic_measure(evidence)
        result['measurement'] = {'method': method, 'description': describe()[method], 'samples': 0}
        return result
    moments = {side: side_moments(evidence[side], method, samples=samples, seed=seed) for side in SIDES}
    available = [moment for moment in moments.values() if moment is not None]
    expected = np.mean([row['mean'] for row in available], axis=0) if available else np.zeros(len(RATINGS))
    monotone = np.clip(isotonic_regression(expected).x, 0., 100.)
    curve = SharedCurve(monotone)(ARGS.grid)
    variance = float(np.mean([row['variance'].mean() for row in available])) if available else 0.
    sampled = method == 'lichess_policy_expectation'
    se = None
    if sampled and available:
        replicates = np.mean([row['replicate_means'] for row in available], axis=0)
        se = replicates.std(axis=0, ddof=1)/math.sqrt(REPLICATES)
    diagnostics = {'rating_grid': list(RATINGS), 'fine_ratings': ARGS.grid.tolist(),
                   'maia_expected_accuracy': expected.tolist(), 'monotone_expected_accuracy': monotone.tolist(),
                   'shared_accuracy': curve.tolist(), 'shared_mean_variance': variance,
                   'identifiable': bool(available and np.ptp(curve) > 1e-10), 'top_probability': 1.,
                   'likelihood': {'kind': 'conditional_aggregate_moments', 'sigma_scale': 1.,
                                  'variance_method': ('policy_aggregate_quadrature' if sampled else
                                                      'exact_weighted_mean' if method == 'volatility_weighted' else 'first_order_delta'),
                                  'accuracy_variance': variance, 'accuracy_sigma': math.sqrt(variance)},
                   'selection': {side: {'positions_used': row['moves'] if row else 0,
                                        'forced_positions_removed': 0,
                                        'forced_positions_included': row['forced_positions_included'] if row else 0}
                                 for side, row in moments.items()},
                   'largest_raw_curve_reversal': float(max(0., -np.diff(expected).min())),
                   'largest_isotonic_adjustment': float(np.max(np.abs(monotone-expected))),
                   'numerical_mean_se': se.tolist() if se is not None else None}
    return {'curve': diagnostics, 'sides': moments,
            'measurement': {'method': method, 'description': describe()[method],
                            'samples': samples if sampled else 0, 'seed': seed if sampled else None,
                            'replicates': REPLICATES if sampled else 0,
                            'sampling': 'independently scrambled Sobol, common across ratings and colors' if sampled else None,
                            'fixed_actual_path_weights': True, 'forced_positions': 'included',
                            'numerical_error': 'SE from four independent scramble means; approximate, distinct from conditional accuracy variance' if sampled else None}}
