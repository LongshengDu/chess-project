"""Joint arithmetic/competitive accuracy with exact Maia-derived covariance.

Both summaries observe the same move-quality draws. If their normalized weights
are a_i=1/n and b_i=sqrt(4p_i(1-p_i))/sum_j sqrt(4p_j(1-p_j)), their conditional
covariance is sum_i Var(Q_i|rating) [a_i,b_i] [a_i,b_i]^T. This retains their
correlation rather than treating two versions of accuracy as independent data.

Both sides share a pair of mean curves and a rating-averaged local covariance,
matching the constant local sigma assumption of the scalar mixture. Other games
add their between-context covariance to the population likelihood. Equal-prior
local/population mixtures use either a bivariate Gaussian or a Beta arithmetic
accuracy marginal with truncated Gaussian competitive accuracy conditional on it.

Mean/median decisions use a separately disclosed5% account blend. An optional
reported projection enforces the original arithmetic-accuracy W/B ordering with
a one-Elo minimum gap; this constraint does not supply independent ranking data.
"""
from __future__ import annotations

from hashlib import blake2b
import json

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import isotonic_regression
from scipy.special import logsumexp, ndtr

from analysis.player_rating.bayesian_shared_curve import ARGS, SharedCurve, prior_density
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.universal_competitiveness import annotate_probabilities, project_order
from tests.analysis.universal_hurdle_quality import _stable_interval


_MOMENT_CACHE = {}
ACCOUNT_WEIGHT = .05
ACCURACY_BIN = .0001  # Both observations expressed on0--1; this is0.01accuracy point.
COVARIANCE_FLOOR = 1e-10


def _side(record):
    rows = [row for row in record['observations'] if len(row['qualities']['position']) > 1]
    if not rows:
        return None
    probability = np.array([row['position_win_probability'] for row in rows])
    competitive = np.sqrt(4*probability*(1-probability))
    if not np.isfinite(competitive).all() or competitive.sum() <= 0:
        raise ValueError('Before-position probabilities must supply nonzero competitive weights.')
    weights = np.stack((np.full(len(rows), 1/len(rows)), competitive/competitive.sum()), axis=1)
    fingerprint = blake2b(digest_size=20)
    fingerprint.update(weights.tobytes())
    positions = []
    for row in rows:
        q = np.asarray(row['qualities']['position'], dtype=float)/100
        policy = np.asarray(row['maia_probabilities'], dtype=float)
        fingerprint.update(np.array([len(q)], dtype=np.int64).tobytes())
        fingerprint.update(q.tobytes()); fingerprint.update(policy.tobytes())
        positions.append((q, policy))
    key = fingerprint.digest()
    if key not in _MOMENT_CACHE:
        means, variances = [], []
        for q, policy in positions:
            mean = policy @ q
            means.append(mean)
            variances.append(np.maximum(0., policy @ (q*q)-mean*mean))
        mean = np.einsum('ir,ij->rj', means, weights)
        covariance = np.einsum('ir,ij,ik->rjk', variances, weights, weights)
        _MOMENT_CACHE[key] = (mean, covariance)
    return _MOMENT_CACHE[key]


def _game(evidence):
    sides = [_side(evidence[side]) for side in ('White', 'Black')]
    available = [side for side in sides if side is not None]
    if not available:
        raise ValueError('Non-forced move evidence is required.')
    mean = np.mean([side[0] for side in available], axis=0)
    constant_covariance = np.mean([side[1] for side in available], axis=(0, 1))
    curves = np.stack([SharedCurve(np.clip(isotonic_regression(mean[:, j]*100).x, 0., 100.))(ARGS.grid)/100
                       for j in range(2)], axis=1)
    return curves, constant_covariance


def _positive_covariance(covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return np.einsum('...ik,...k,...jk->...ij', eigenvectors,
                     np.maximum(eigenvalues, COVARIANCE_FLOOR), eigenvectors)


def _observed(record):
    rows = [row for row in record['observations'] if len(row['qualities']['position']) > 1]
    q = np.array([row['qualities']['position'][row['played_index']]/100 for row in rows])
    probability = np.array([row['position_win_probability'] for row in rows])
    weights = np.sqrt(4*probability*(1-probability))
    return np.array([q.mean(), weights @ q/weights.sum()])


def _likelihood(observed, mean, covariance, bounded):
    covariance = _positive_covariance(covariance)
    residual = observed-mean
    if not bounded:
        inverse = np.linalg.inv(covariance)
        return -.5*(np.einsum('ri,rij,rj->r', residual, inverse, residual)
                     +np.linalg.slogdet(covariance)[1]+2*np.log(2*np.pi))
    raw_variance = covariance[:, 0, 0]
    raw_mass = beta_accuracy_mass(observed[0]*100, mean[:, 0]*100, raw_variance*10000)
    conditional_mean = mean[:, 1]+covariance[:, 1, 0]/raw_variance*residual[:, 0]
    conditional_variance = np.maximum(covariance[:, 1, 1]-covariance[:, 1, 0]**2/raw_variance,
                                      COVARIANCE_FLOOR)
    sigma = np.sqrt(conditional_variance)
    lower, upper = max(0., observed[1]-ACCURACY_BIN/2), min(1., observed[1]+ACCURACY_BIN/2)
    conditional_mass = _stable_interval((lower-conditional_mean)/sigma, (upper-conditional_mean)/sigma,
                                         ndtr, lambda x: ndtr(-x))
    normalization = _stable_interval(-conditional_mean/sigma, (1-conditional_mean)/sigma,
                                      ndtr, lambda x: ndtr(-x))
    conditional_mass = np.divide(conditional_mass, normalization,
                                  out=np.zeros_like(normalization), where=normalization > 0)
    return np.log(np.maximum(raw_mass, 1e-300))+np.log(np.maximum(conditional_mass, 1e-300))


def predict(evidence, fit, ratings, calibration_cases):
    """Evaluate two proper universal joint models and optional ordering constraints."""
    local_mean, constant_covariance = _game(evidence)
    local_covariance = np.broadcast_to(constant_covariance, (len(ARGS.grid), 2, 2))
    calibration = [_game(case['evidence']) for case in calibration_cases]
    means = np.stack([item[0] for item in calibration])
    population_mean = means.mean(axis=0)
    centered = means-population_mean
    between = np.einsum('gri,grj->rij', centered, centered)/max(1, len(calibration)-1)
    population_covariance = between+np.mean([item[1] for item in calibration], axis=0)
    outputs = {}
    for bounded, family in ((False, 'joint_gaussian'), (True, 'joint_conditional_beta')):
        for point in ('mean', 'median'):
            outputs[family+'_'+point+'_account'] = {}
        for side in ('White', 'Black'):
            observed = _observed(evidence[side])
            likelihoods = [_likelihood(observed, mean, covariance, bounded)
                           for mean, covariance in ((local_mean, local_covariance),
                                                     (population_mean, population_covariance))]
            logs = logsumexp(likelihoods, axis=0)-np.log(2.)
            density = np.exp(logs-logs.max())*prior_density(ARGS.grid)
            density /= trapezoid(density, ARGS.grid)
            cdf = cumulative_trapezoid(density, ARGS.grid, initial=0.)
            points = {'mean': float(trapezoid(density*ARGS.grid, ARGS.grid)),
                      'median': float(np.interp(.5, cdf/cdf[-1], ARGS.grid))}
            for point, value in points.items():
                outputs[family+'_'+point+'_account'][side] = (1-ACCOUNT_WEIGHT)*value+ACCOUNT_WEIGHT*float(ratings[side])
    outputs.update({name+'_ordered': project_order(pair, fit) for name, pair in list(outputs.items())})
    return outputs


def main():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, read_json, rank
    from tests.analysis.edge_quality_likelihood import is_edge
    cases, _ = load_cases(ROOT/'games')
    for case in cases:
        moves = read_json(ROOT/'games/output'/f'{case["game"]}-full'/'analysis.json')['moves']
        case['input']['evidence'] = annotate_probabilities(case['input']['evidence'], moves)
    predictions = [predict(**case['input'], calibration_cases=[other['input'] for other in cases if other is not case])
                   for case in cases]
    rows = []
    for case, prediction in zip(cases, predictions, strict=True):
        for method, pair in prediction.items():
            for side, value in pair.items():
                sensitivity = {}
                for changed_side in ('White', 'Black'):
                    sensitivity[changed_side] = [value]
                    for shift in (-200., -100., 100., 200.):
                        changed = dict(prediction[method.removesuffix('_ordered')])
                        changed[changed_side] += ACCOUNT_WEIGHT*shift
                        if method.endswith('_ordered'):
                            changed = project_order(changed, case['input']['fit'])
                        sensitivity[changed_side].append(changed[side])
                other = 'Black' if side == 'White' else 'White'
                rows.append({'game': case['game'], 'side': side, 'method': method,
                             'estimate': value, 'reference': case['references'][side],
                             'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': float(np.max(abs(np.array(sensitivity[side])-value))),
                             'own_elo_span': float(np.ptp(sensitivity[side])),
                             'opponent_elo_max_change': float(np.max(abs(np.array(sensitivity[other])-value)))})
    ranked = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/joint-accuracy.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'rankings': ranked, 'rows': rows, 'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(ranked, indent=2))
    print(output)


if __name__ == '__main__':
    main()
