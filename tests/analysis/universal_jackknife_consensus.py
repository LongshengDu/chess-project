"""Label-free jackknife GLS consensus of two universal quality estimators.

Delete each of the15 calibration games in turn, recompute both point estimates,
and average the resulting2x2 covariance matrices for White and Black. The shared
simplex weight minimizes conditional corpus-sampling variance; it is not a
commercial-error or bias-minimizing weight. Account Elo does not enter weights.

For tractable covariance estimation, each jackknife coverage estimate evaluates
the unprojected coverage rule only at the actual observed accuracy. The final
consensus uses the full isotonic-projected coverage map and the integrated-power
predictive estimate. This is an explicit approximation when isotonic correction
affects an observed point. The target game is excluded from every calibration.
"""
from __future__ import annotations

import json

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass, posterior
from tests.analysis.universal_competitiveness import annotate_probabilities, project_order
from tests.analysis.universal_competitive_marginal import marginal_fit
from tests.analysis.universal_native_coverage import _points, native_reliability, predict as coverage_predict


def _population_moments(calibration):
    _, variances, mean, between = _population(calibration, ARGS.grid)
    return mean, between+variances.mean()


def _raw_components(fit, weighted_fit, calibration, weighted_calibration):
    """Two unblended point estimates at both observed values; no account input."""
    grid = ARGS.grid
    own = fit['diagnostics']['curve']
    accuracy = np.array([fit['players'][side]['average_accuracy'] for side in ('White', 'Black')])
    population, variance = _population_moments(calibration)
    local = _points(gaussian_accuracy_mass(accuracy[:, None], own['shared_accuracy'],
                                          own['likelihood']['accuracy_variance']), grid)['mean']
    global_points = _points(beta_accuracy_mass(accuracy[:, None], population, variance), grid)['mean']
    knots = own['monotone_expected_accuracy']
    reliability = native_reliability(accuracy, knots[0], knots[-1], own['likelihood']['accuracy_variance'])
    coverage = reliability*local+(1-reliability)*global_points
    weighted_own = weighted_fit['diagnostics']['curve']
    weighted_accuracy = np.array([weighted_fit['players'][side]['average_accuracy'] for side in ('White', 'Black')])
    weighted_population, weighted_variance = _population_moments(weighted_calibration)
    mixture = posterior(weighted_accuracy[:, None], np.asarray(weighted_own['shared_accuracy']),
                        weighted_own['likelihood']['accuracy_variance'], weighted_population,
                        weighted_variance, grid)['mean']
    return np.stack((coverage, mixture), axis=-1)


def simplex_gls_weight(jackknife_points):
    """Return common coverage weight and average delete-one jackknife covariance."""
    points = np.asarray(jackknife_points, dtype=float)
    count = len(points)
    if points.shape != (count, 2, 2) or count < 2 or not np.isfinite(points).all():
        raise ValueError('At least two finite jackknife samples for two players/models are required.')
    centered = points-points.mean(axis=0)
    covariance = (count-1)/count*np.einsum('gsi,gsj->sij', centered, centered).mean(axis=0)
    first, second, cross = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    difference_variance = first+second-2*cross
    if difference_variance <= 1e-12*max(1., first+second):
        weight = .5
    else:
        weight = float(np.clip((second-cross)/difference_variance, 0., 1.))
    return weight, covariance


def predict_detail(evidence, fit, ratings, calibration_cases):
    weighted_fit = marginal_fit(evidence)
    weighted_calibration = [{'fit': marginal_fit(case['evidence'])} for case in calibration_cases]
    jackknife = [
        _raw_components(fit, weighted_fit, calibration_cases[:index]+calibration_cases[index+1:],
                         weighted_calibration[:index]+weighted_calibration[index+1:])
        for index in range(len(calibration_cases))]
    weight, covariance = simplex_gls_weight(jackknife)
    final_coverage = coverage_predict(evidence, fit, ratings, calibration_cases)[
        'native_coverage_mean_account_5pct_all']
    raw_base = _raw_components(fit, weighted_fit, calibration_cases, weighted_calibration)
    pair = {side: weight*final_coverage[side]+(1-weight)*(.95*raw_base[index, 1]+.05*ratings[side])
            for index, side in enumerate(('White', 'Black'))}
    reference_coverage_raw = .95*raw_base[:, 0]+.05*np.array([ratings[side] for side in ('White', 'Black')])
    return {'prediction': {'jackknife_gls_consensus': project_order(pair, fit)},
            'coverage_weight': weight, 'jackknife_covariance': covariance.tolist(),
            'maximum_isotonic_point_difference': max(abs(final_coverage[side]-reference_coverage_raw[index])
                                                     for index, side in enumerate(('White', 'Black')))}


def predict(evidence, fit, ratings, calibration_cases):
    return predict_detail(evidence, fit, ratings, calibration_cases)['prediction']


def run():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, read_json, rank
    from tests.analysis.edge_quality_likelihood import is_edge
    cases, _ = load_cases(ROOT/'games')
    for case in cases:
        moves = read_json(ROOT/'games/output'/f'{case["game"]}-full'/'analysis.json')['moves']
        case['input']['evidence'] = annotate_probabilities(case['input']['evidence'], moves)
    predictions = []
    for case in cases:
        prediction = predict_detail(**case['input'], calibration_cases=[other['input'] for other in cases if other is not case])
        predictions.append(prediction)
        print(f'{case["game"]}: coverage weight {prediction["coverage_weight"]:.3f}', flush=True)
    rows, diagnostics = [], []
    for case, detailed in zip(cases, predictions, strict=True):
        diagnostics.append({'game': case['game'], **{k:v for k,v in detailed.items() if k != 'prediction'}})
        for method, pair in detailed['prediction'].items():
            for side, value in pair.items():
                rows.append({'game': case['game'], 'side': side, 'method': method, 'estimate': value,
                             'reference': case['references'][side], 'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': 10., 'own_elo_span': 20., 'opponent_elo_max_change': 5.})
    ranking = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/jackknife-consensus.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'ranking': ranking, 'players': rows, 'diagnostics': diagnostics,
                                  'assumptions': __doc__, 'sensitivity_note': 'Analytic5%account/projection bounds.'}, indent=2), encoding='utf-8')
    print(json.dumps(ranking, indent=2))


if __name__ == '__main__':
    run()
