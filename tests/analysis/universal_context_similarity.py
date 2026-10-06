"""Context-similar population calibration without reference-trained weights.

Three prespecified kernels condition population calibration on Maia evidence:
full raw/competitive curve distance; centered curve-shape distance; Gaussian
predictive Wasserstein distance including conditional standard deviations.
Each Gaussian-kernel bandwidth is the median positive pairwise distance among
calibration contexts. The target game, reference ratings and actual ratings
cannot determine that bandwidth. Both measurement populations use the same
game weights, normalized to one, with finite-sample weighted covariance.
An additional reference-free alternative weights calibration contexts by their
prior-averaged Gaussian mean Fisher information (squared slope/noise variance).

The existing25%native-coverage /75%square-root-competitive opinion consensus,
10%account blend and original arithmetic-accuracy ordering constraint are fixed.
This changes population context selection only, not opinion/account weights.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.special import betaln, logsumexp

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID, prior_density
from tests.analysis.edge_predictive_mixture import posterior, gaussian_accuracy_mass
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass, beta_parameters
from tests.analysis.universal_competitiveness import annotate_probabilities, project_order, weighted_fit
from tests.analysis.universal_native_coverage import accuracy_mapping, native_reliability, _points


METHODS = ('context_curve', 'context_shape', 'context_predictive', 'context_information')
ACCOUNT_WEIGHT = .10
COVERAGE_WEIGHT = .25


def _features(fits):
    curves = np.stack([fit['diagnostics']['curve']['monotone_expected_accuracy'] for fit in fits])
    sigmas = np.sqrt([fit['diagnostics']['curve']['likelihood']['accuracy_variance'] for fit in fits])
    return curves, sigmas


def context_weights(target, calibration, method):
    """Reference-free Gaussian similarity with a calibration-only median bandwidth."""
    target_curves, target_sigmas = _features(target)
    curves = np.stack([_features(pair)[0] for pair in calibration])
    sigmas = np.stack([_features(pair)[1] for pair in calibration])
    if method == 'context_information':
        rating_weights = prior_density(GRID)
        rating_weights /= rating_weights.sum()
        slopes = np.gradient(curves, GRID, axis=-1)
        information = np.mean(np.sum(slopes*slopes*rating_weights, axis=-1)
                              /np.maximum(sigmas*sigmas, 1e-12), axis=1)
        weights = information/information.sum() if information.sum() > 0 else np.ones(len(calibration))/len(calibration)
        return weights, 0.
    if method == 'context_shape':
        target_curves = target_curves-target_curves.mean(axis=-1, keepdims=True)
        curves = curves-curves.mean(axis=-1, keepdims=True)
    target_distance = np.mean((curves-target_curves)**2, axis=(1, 2))
    pair_distance = np.mean((curves[:, None]-curves[None, :])**2, axis=(2, 3))
    if method == 'context_predictive':
        # W2^2(N(mu1,sigma1²),N(mu2,sigma2²))=(mu1-mu2)²+(sigma1-sigma2)².
        target_distance += np.mean((sigmas-target_sigmas)**2, axis=1)
        pair_distance += np.mean((sigmas[:, None]-sigmas[None, :])**2, axis=2)
    elif method not in ('context_curve', 'context_shape'):
        raise ValueError('Unknown context-distance construction.')
    pair_values = pair_distance[np.triu_indices(len(calibration), 1)]
    positive = pair_values[pair_values > 1e-12]
    if not len(positive):
        return np.ones(len(calibration))/len(calibration), 0.
    bandwidth_squared = float(np.median(positive))
    logs = -.5*target_distance/bandwidth_squared
    weights = np.exp(logs-logs.max())
    return weights/weights.sum(), float(np.sqrt(bandwidth_squared))


def _population(fits, weights):
    curves = np.asarray([fit['diagnostics']['curve']['shared_accuracy'] for fit in fits])
    variance = np.asarray([fit['diagnostics']['curve']['likelihood']['accuracy_variance'] for fit in fits])
    mean = weights @ curves
    correction = max(1-float(weights @ weights), 1e-12)
    between = weights @ ((curves-mean)**2)/correction
    return mean, between+float(weights @ variance)


def _safe_coverage_mapping(target, target_variance, population, population_variance, bounds):
    """Use existing points; repair only fully underflowed Beta rows in log space."""
    try:
        return accuracy_mapping(target, target_variance, population, population_variance, bounds, ARGS.grid)
    except ValueError as error:
        if 'no finite posterior mass' not in str(error):
            raise
    accuracy = np.linspace(0., 100., 1001)
    reliability = native_reliability(accuracy, *bounds, target_variance)
    local = _points(gaussian_accuracy_mass(accuracy[:, None], np.asarray(target), target_variance), ARGS.grid)
    pop = _points(beta_accuracy_mass(accuracy[:, None], population, population_variance), ARGS.grid)
    bad = np.flatnonzero(~pop['valid'])
    if len(bad):
        # Gauss--Legendre nodes lie strictly inside the0.01-point observation
        # interval, permitting stable log density integration at either boundary.
        nodes, weights = np.polynomial.legendre.leggauss(64)
        alpha, beta = beta_parameters(population, population_variance)
        lower = np.maximum(0., accuracy[bad]-.005)/100
        upper = np.minimum(100., accuracy[bad]+.005)/100
        x = (lower[:, None]+upper[:, None])/2+(upper-lower)[:, None]/2*nodes
        log_density = ((alpha[None, None, :]-1)*np.log(x[:, :, None])
                       +(beta[None, None, :]-1)*np.log1p(-x[:, :, None])
                       -betaln(alpha, beta)[None, None, :])
        log_mass = (logsumexp(log_density+np.log(weights)[None, :, None], axis=1)
                    +np.log((upper-lower)/2)[:, None])
        repaired = _points(np.exp(log_mass-log_mass.max(axis=1, keepdims=True)), ARGS.grid)
        pop['mean'][bad] = repaired['mean']
        pop['valid'][bad] = repaired['valid']
    if not np.all(pop['valid']) or np.any((~local['valid']) & (reliability > 1e-12)):
        raise ValueError('Stable integration could not identify an active posterior.')
    raw = np.where(local['valid'], reliability*local['mean']+(1-reliability)*pop['mean'], pop['mean'])
    return {'accuracy_grid': accuracy, 'mean': isotonic_regression(raw).x}


def _predict_with_weights(fit, competitive, calibration, weights, ratings):
    original_population, original_variance = _population([pair[0] for pair in calibration], weights)
    competitive_population, competitive_variance = _population([pair[1] for pair in calibration], weights)
    raw = fit['diagnostics']['curve']
    competitive_curve = competitive['diagnostics']['curve']
    bounds = [raw['monotone_expected_accuracy'][0], raw['monotone_expected_accuracy'][-1]]
    mapping = _safe_coverage_mapping(raw['shared_accuracy'], raw['likelihood']['accuracy_variance'],
                                     original_population, original_variance, bounds)
    # Infer both component decisions before adding accounts or projecting order.
    quality = {}
    for side in ('White', 'Black'):
        coverage = float(np.interp(fit['players'][side]['average_accuracy'], mapping['accuracy_grid'], mapping['mean']))
        value = posterior(competitive['players'][side]['average_accuracy'],
                          np.asarray(competitive_curve['shared_accuracy']),
                          competitive_curve['likelihood']['accuracy_variance'],
                          competitive_population, competitive_variance, ARGS.grid)
        quality[side] = COVERAGE_WEIGHT*coverage+(1-COVERAGE_WEIGHT)*float(value['mean'])
    return project_order({side: (1-ACCOUNT_WEIGHT)*quality[side]+ACCOUNT_WEIGHT*ratings[side]
                          for side in ('White', 'Black')}, fit)


def predict(evidence, fit, ratings, calibration_cases):
    competitive = weighted_fit(evidence, .5)
    calibration = [(case['fit'], weighted_fit(case['evidence'], .5)) for case in calibration_cases]
    output = {}
    for method in METHODS:
        weights, _ = context_weights((fit, competitive), calibration, method)
        output[method] = _predict_with_weights(fit, competitive, calibration, weights, ratings)
    return output


def main():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, read_json, rank
    from tests.analysis.edge_quality_likelihood import is_edge
    cases, _ = load_cases(ROOT/'games')
    for case in cases:
        moves = read_json(ROOT/'games/output'/f'{case["game"]}-full'/'analysis.json')['moves']
        case['input']['evidence'] = annotate_probabilities(case['input']['evidence'], moves)
    predictions, diagnostics = [], []
    for index, case in enumerate(cases):
        others = [other for other in cases if other is not case]
        data = case['input']
        predictions.append(predict(**data, calibration_cases=[other['input'] for other in others]))
        target = (data['fit'], weighted_fit(data['evidence'], .5))
        calibration = [(other['input']['fit'], weighted_fit(other['input']['evidence'], .5)) for other in others]
        for method in METHODS:
            weights, bandwidth = context_weights(target, calibration, method)
            diagnostics.append({'game': case['game'], 'method': method, 'bandwidth': bandwidth,
                                'effective_calibration_games': float(1/(weights @ weights)),
                                'calibration_weights': {other['game']: float(weight) for other, weight in zip(others, weights, strict=True)}})
        print(f'Predicted {case["game"]}.', flush=True)
    rows = []
    for case, prediction in zip(cases, predictions, strict=True):
        for method, pair in prediction.items():
            for side, value in pair.items():
                rows.append({'game': case['game'], 'side': side, 'method': method,
                             'estimate': value, 'reference': case['references'][side],
                             'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': 20., 'own_elo_span': 40., 'opponent_elo_max_change': 10.})
    ranked = rank(rows)
    rounded = rank([{**row, 'estimate': round(row['estimate'])} for row in rows])
    output = ROOT/'tests/analysis/output/universal-rating-methods/context-similarity.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'rankings': ranked, 'rounded_rankings': rounded, 'rows': rows,
                                  'context_diagnostics': diagnostics, 'assumptions': __doc__,
                                  'sensitivity_note': 'Conservative analytic bounds from10%blend andEuclideanpairprojection.'}, indent=2), encoding='utf-8')
    print(json.dumps({'rankings': ranked, 'rounded_rankings': rounded}, indent=2))
    print(output)


if __name__ == '__main__':
    main()
