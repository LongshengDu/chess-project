"""Joint average quality/error-frequency inference; no reference calibration.

The observation is (average accuracy, error frequency). Their joint Maia-derived
covariance prevents counting accuracy and its related error-frequency statistic
as independent evidence. Error frequency plus mean loss identifies a frequency/
severity distinction that arithmetic accuracy alone discards. All players use
the same inference rule; White/Black ordering is evaluated, never overwritten.

Three predeclared models: bivariate Gaussian at5 accuracy points of loss; a Beta
quality marginal with conditional Gaussian error frequency at5 points; the same
bounded model at1 point as a numerical-engine-noise sensitivity. Local and other-
game population predictive likelihoods have equal model prior probabilities.
Posterior means minimize squared rating error. Actual Elo is intentionally unused.
"""
from __future__ import annotations

from hashlib import blake2b
import json

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import isotonic_regression
from scipy.special import betainc, betaincc, logsumexp, ndtr

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID, SharedCurve, prior_density


_MOMENT_CACHE = {}
METHODS = {'hurdle_gaussian_loss5': (5., False),
           'hurdle_conditional_beta_loss5': (5., True),
           'hurdle_conditional_beta_loss1': (1., True)}


def _side_moments(record, threshold):
    fingerprint = blake2b(digest_size=20)
    fingerprint.update(np.array([threshold], dtype=np.float64).tobytes())
    positions = []
    for observation in record['observations']:
        qualities = np.asarray(observation['qualities']['position'], dtype=np.float64)/100
        if len(qualities) <= 1:
            continue
        policy = np.asarray(observation['maia_probabilities'], dtype=np.float64)
        policy = policy/policy.sum(axis=1, keepdims=True)
        feature = np.column_stack((qualities, qualities < (100-threshold)/100))
        positions.append((feature, policy))
        fingerprint.update(np.array(feature.shape, dtype=np.int64).tobytes())
        fingerprint.update(feature.tobytes()); fingerprint.update(policy.tobytes())
    key = fingerprint.digest()
    if key not in _MOMENT_CACHE:
        if not positions:
            return None
        means, covariances = [], []
        for feature, policy in positions:
            mean = policy@feature
            second = np.einsum('rk,ki,kj->rij', policy, feature, feature)
            means.append(mean)
            covariances.append(second-np.einsum('ri,rj->rij', mean, mean))
        n = len(positions)
        _MOMENT_CACHE[key] = (np.mean(means, axis=0), np.sum(covariances, axis=0)/(n*n))
    return _MOMENT_CACHE[key]


def _game_moments(evidence, threshold):
    sides = [_side_moments(evidence[side], threshold) for side in ('White', 'Black')]
    sides = [side for side in sides if side is not None]
    if not sides:
        raise ValueError('Game evidence requires non-forced move observations.')
    return np.mean([side[0] for side in sides], axis=0), np.mean([side[1] for side in sides], axis=0)


def _extend(mean, covariance):
    grid = ARGS.grid
    accuracy = SharedCurve(np.clip(isotonic_regression(mean[:, 0]*100).x, 0, 100))(grid)/100
    non_error = SharedCurve(np.clip(isotonic_regression((1-mean[:, 1])*100).x, 0, 100))(grid)/100
    extended = np.stack((accuracy, 1-non_error), axis=-1)
    covariance = np.stack([np.interp(grid, GRID, covariance[:, i, j])
                           for i in range(2) for j in range(2)], axis=-1).reshape(-1, 2, 2)
    # Symmetric eigendecomposition guarantees only a tiny numerical SPD floor.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = np.einsum('rik,rk,rjk->rij', eigenvectors, np.maximum(eigenvalues, 1e-10), eigenvectors)
    return extended, covariance


def _population(cases, threshold):
    moments = [_game_moments(case['evidence'], threshold) for case in cases]
    means = np.stack([row[0] for row in moments])
    centered = means-means.mean(axis=0)
    between = np.einsum('gri,grj->rij', centered, centered)/max(1, len(means)-1)
    covariance = np.mean([row[1] for row in moments], axis=0)+between
    return _extend(means.mean(axis=0), covariance)


def _observed(record, threshold):
    qualities = [row['qualities']['position'][row['played_index']]/100
                 for row in record['observations'] if len(row['qualities']['position']) > 1]
    if not qualities:
        raise ValueError('A player needs non-forced moves.')
    qualities = np.asarray(qualities)
    return np.array([qualities.mean(), np.mean(qualities < (100-threshold)/100)]), len(qualities)


def _stable_interval(lower, upper, cdf, survival):
    left = cdf(lower)
    return np.maximum(np.where(left < .5, cdf(upper)-left, survival(lower)-survival(upper)), 0.)


def _log_likelihood(observed, count, mean, covariance, bounded):
    residual = observed-mean
    if not bounded:
        inverse = np.linalg.inv(covariance)
        return -.5*(np.einsum('ri,rij,rj->r', residual, inverse, residual)
                     +np.linalg.slogdet(covariance)[1]+2*np.log(2*np.pi))
    variance_accuracy = covariance[:, 0, 0]
    probability = np.clip(mean[:, 0], 1e-9, 1-1e-9)
    maximum = probability*(1-probability)
    variance = np.clip(variance_accuracy, maximum*1e-9, maximum*(1-1e-9))
    concentration = maximum/variance-1
    alpha, beta = probability*concentration, (1-probability)*concentration
    lower, upper = max(0, observed[0]-.00005), min(1, observed[0]+.00005)
    accuracy_mass = _stable_interval(lower, upper,
                                    lambda x: betainc(alpha, beta, x),
                                    lambda x: betaincc(alpha, beta, x))
    conditional_mean = mean[:, 1]+covariance[:, 1, 0]/variance_accuracy*residual[:, 0]
    conditional_sigma = np.sqrt(np.maximum(covariance[:, 1, 1]
                                           -covariance[:, 1, 0]**2/variance_accuracy, 1e-10))
    lower, upper = max(0, observed[1]-.5/count), min(1, observed[1]+.5/count)
    frequency_mass = _stable_interval((lower-conditional_mean)/conditional_sigma,
                                      (upper-conditional_mean)/conditional_sigma,
                                      ndtr, lambda x: ndtr(-x))
    support = _stable_interval(-conditional_mean/conditional_sigma,
                               (1-conditional_mean)/conditional_sigma,
                               ndtr, lambda x: ndtr(-x))
    frequency_mass = np.divide(frequency_mass, support, out=np.zeros_like(support), where=support > 0)
    return np.log(np.maximum(accuracy_mass, 1e-300))+np.log(np.maximum(frequency_mass, 1e-300))


def predict(evidence, fit, ratings, calibration_cases):
    """Evaluate every player; target evidence never enters population calibration."""
    del fit, ratings
    grid = ARGS.grid
    result = {}
    for name, (threshold, bounded) in METHODS.items():
        local = _extend(*_game_moments(evidence, threshold))
        population = _population(calibration_cases, threshold)
        result[name] = {}
        for side in ('White', 'Black'):
            observed, count = _observed(evidence[side], threshold)
            logs = [_log_likelihood(observed, count, *moments, bounded)
                    for moments in (local, population)]
            logs = logsumexp(logs, axis=0)-np.log(2)
            density = np.exp(logs-logs.max())*prior_density(grid)
            density /= trapezoid(density, grid)
            result[name][side] = float(trapezoid(density*grid, grid))
    return result


def run():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases
    cases, _ = load_cases(ROOT/'games')
    predictions = [predict(**case['input'], calibration_cases=[other['input'] for other in cases if other is not case])
                   for case in cases]
    rows, ranking = [], []
    for name in METHODS:
        for case, prediction in zip(cases, predictions, strict=True):
            for side in ('White', 'Black'):
                accuracy = case['input']['fit']['players'][side]['average_accuracy']
                curve = case['input']['fit']['diagnostics']['curve']['monotone_expected_accuracy']
                rows.append({'game': case['game'], 'side': side, 'method': name,
                             'estimate': prediction[name][side], 'reference': case['references'][side],
                             'edge': not curve[0] <= accuracy <= curve[-1]})
        selected = [r for r in rows if r['method'] == name]
        errors = [abs(r['estimate']-r['reference']) for r in selected]
        ranking.append({'method': name, 'mae': float(np.mean(errors)), 'max_error': max(errors),
                        'edge_mae': float(np.mean([abs(r['estimate']-r['reference']) for r in selected if r['edge']])),
                        'ordering': int(sum(np.sign(selected[i]['estimate']-selected[i+1]['estimate']) ==
                                            np.sign(selected[i]['reference']-selected[i+1]['reference'])
                                            for i in range(0, len(selected), 2))),
                        'account_sensitivity': 0.})
    ranking.sort(key=lambda r: r['mae'])
    output = ROOT/'tests/analysis/output/universal-rating-methods/hurdle-quality.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'ranking': ranking, 'players': rows}, indent=2), encoding='utf-8')
    print(json.dumps(ranking, indent=2))


if __name__ == '__main__':
    run()
