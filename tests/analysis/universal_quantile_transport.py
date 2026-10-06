"""Account-anchored quantile transport with reference-free population calibration.

Three fixed universal mechanisms: Beta-CDF transport at supplied Elo, Gaussian
standardized-residual transport at supplied Elo, and Beta-CDF transport after
marginalizing uniformly over Elo +/-200. The latter window is the prespecified
rating-uncertainty sensitivity range, not a parameter fitted to references.

Each target percentile is mapped onto the leave-game-out population distribution
at the same account context. Population likelihoods then infer a performance
rating, reported as both posterior median and mean. No additional account blend,
curve-edge switch, commercial label or game identifier enters predictions.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import brentq
from scipy.special import betainc, betaincc, betaincinv, betainccinv

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_density
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass, beta_parameters
from tests.analysis.edge_global_quality import _population


ACCOUNT_HALF_WINDOW = 200.
QUADRATURE_NODES = 21
BASE_METHODS = ('qt_beta_account', 'qt_gaussian_account', 'qt_beta_window')


def _infer(logs, grid):
    prior = prior_density(grid)
    valid = prior > 0
    density = np.zeros_like(grid)
    density[valid] = np.exp(logs[valid]-np.max(logs[valid]))*prior[valid]
    density /= trapezoid(density, grid)
    cdf = cumulative_trapezoid(density, grid, initial=0.)
    return {'median': float(np.interp(.5, cdf/cdf[-1], grid)),
            'mean': float(trapezoid(density*grid, grid))}


def _beta_transport(observed, target_mean, target_variance, population_mean, population_variance, weights):
    local_a, local_b = beta_parameters(target_mean, target_variance)
    population_a, population_b = beta_parameters(population_mean, population_variance)
    quantile = float(np.sum(weights*betainc(local_a, local_b, observed/100.)))
    survival = float(np.sum(weights*betaincc(local_a, local_b, observed/100.)))
    if len(weights) == 1:
        inverse = (betaincinv(population_a[0], population_b[0], quantile) if quantile <= .5
                   else betainccinv(population_a[0], population_b[0], survival))
        return float(100*inverse)
    if quantile <= .5:
        residual = lambda x: float(np.sum(weights*betainc(population_a, population_b, x))-quantile)
    else:
        residual = lambda x: float(np.sum(weights*betaincc(population_a, population_b, x))-survival)
    value = brentq(residual, 0., 1., xtol=1e-14)
    return float(100*value)


def predict(evidence, fit, ratings, calibration_cases):
    """Transport every player's accuracy by the same three population rules."""
    del evidence
    grid = ARGS.grid
    _, conditional_variances, population_mean, between = _population(calibration_cases, grid)
    population_variance = np.maximum(between+conditional_variances.mean(), 1e-12)
    curve = fit['diagnostics']['curve']
    target_mean = np.asarray(curve['shared_accuracy'])
    target_variance = max(float(curve['likelihood']['accuracy_variance']), 1e-12)
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(QUADRATURE_NODES)
    quadrature_weights /= 2.
    result = {method+'_'+summary+'_all': {} for method in BASE_METHODS for summary in ('median', 'mean')}
    for side in ('White', 'Black'):
        observed = fit['players'][side]['average_accuracy']
        if observed is None:
            for predictions in result.values():
                predictions[side] = None
            continue
        account = float(ratings[side])
        if not np.isfinite(account) or not 0 <= account <= 3200:
            raise ValueError('Account ratings must be finite within0--3200.')
        anchor = np.array([account])
        local_at_account = np.interp(anchor, grid, target_mean)
        population_at_account = np.interp(anchor, grid, population_mean)
        variance_at_account = np.interp(anchor, grid, population_variance)
        adjusted_beta = _beta_transport(observed, local_at_account, target_variance,
                                        population_at_account, variance_at_account, np.ones(1))
        residual = (observed-local_at_account[0])/np.sqrt(target_variance)
        adjusted_gaussian = population_at_account[0]+residual*np.sqrt(variance_at_account[0])
        lower = max(0., account-ACCOUNT_HALF_WINDOW)
        upper = min(3200., account+ACCOUNT_HALF_WINDOW)
        window = (lower+upper)/2+(upper-lower)/2*nodes
        adjusted_window = _beta_transport(
            observed, np.interp(window, grid, target_mean), target_variance,
            np.interp(window, grid, population_mean), np.interp(window, grid, population_variance),
            quadrature_weights)
        logs = {
            'qt_beta_account': np.log(np.maximum(beta_accuracy_mass(adjusted_beta, population_mean, population_variance), 1e-300)),
            'qt_gaussian_account': -.5*((adjusted_gaussian-population_mean)**2/population_variance+np.log(population_variance)),
            'qt_beta_window': np.log(np.maximum(beta_accuracy_mass(adjusted_window, population_mean, population_variance), 1e-300)),
        }
        for method, likelihood in logs.items():
            for summary, value in _infer(likelihood, grid).items():
                result[method+'_'+summary+'_all'][side] = value
    return result


def main():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, rank
    from tests.analysis.edge_quality_likelihood import is_edge
    cases, _ = load_cases(ROOT/'games')
    all_predictions = []
    for index, case in enumerate(cases):
        data = case['input']
        calibration = [other['input'] for i, other in enumerate(cases) if i != index]
        original = predict(**data, calibration_cases=calibration)
        sensitivity = {method: {side: [] for side in ('White', 'Black')} for method in original}
        for side in ('White', 'Black'):
            for shift in np.linspace(-200., 200., 9):
                shifted_ratings = dict(data['ratings'])
                shifted_ratings[side] += shift
                shifted = predict(data['evidence'], data['fit'], shifted_ratings, calibration)
                for method in original:
                    sensitivity[method][side].append(shifted[method][side])
                    other = 'Black' if side == 'White' else 'White'
                    if shifted[method][other] != original[method][other]:
                        raise AssertionError('A target rating shift unexpectedly changed the opponent.')
        all_predictions.append((original, sensitivity))
    rows = []
    for case, (prediction, sensitivity) in zip(cases, all_predictions, strict=True):
        for method, pair in prediction.items():
            for side, value in pair.items():
                changes = np.asarray(sensitivity[method][side])
                rows.append({'game': case['game'], 'side': side, 'method': method,
                             'estimate': value, 'reference': case['references'][side],
                             'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': float(np.max(abs(changes-value))),
                             'own_elo_span': float(np.ptp(changes)), 'opponent_elo_max_change': 0.})
    ranked = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/quantile-transport.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'rankings': ranked, 'rows': rows, 'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(ranked, indent=2))
    print(output)


if __name__ == '__main__':
    main()
