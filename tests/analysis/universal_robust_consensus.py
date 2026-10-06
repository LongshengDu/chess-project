"""Three fixed robust opinion pools, with no commercial-label weight fitting.

The four structural opinions cross two measurements (arithmetic/competitive) and
two reliability models (native-coverage/proper predictive-mixture mean). Compare
coordinatewise median, geometric median of the paired ratings, and the mean of
the least-dispersed three of four opinions. These classical location estimators
address one unusually discrepant model, not reference-calibrated model bias.

All players use the same rule. The already declared10% account contribution is
applied after pooling, followed by the explicit original-accuracy order projection.
Before projection, account changes translate each coordinate by exactly10%; after
projection own-rating sensitivity is at most20Elo for+/-200, opponent at most10.
"""
from __future__ import annotations

from itertools import combinations
import json

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior
from tests.analysis.universal_competitiveness import annotate_probabilities, project_order, weighted_fit
from tests.analysis import universal_native_coverage


ACCOUNT_WEIGHT = .10


def geometric_median(points):
    """Modified Weiszfeld iteration, including the coincident-iterate case."""
    points = np.asarray(points, dtype=float)
    current = points.mean(axis=0)
    for _ in range(1000):
        distances = np.linalg.norm(points-current, axis=1)
        nonzero = distances > 1e-10
        if not np.any(nonzero):
            return current
        weights = 1/distances[nonzero]
        weighted_mean = np.sum(points[nonzero]*weights[:, None], axis=0)/weights.sum()
        coincidences = np.count_nonzero(~nonzero)
        if coincidences:
            residual = np.sum((points[nonzero]-current)*weights[:, None], axis=0)
            norm = np.linalg.norm(residual)
            if norm <= coincidences:
                return current
            fraction = coincidences/norm
            updated = (1-fraction)*weighted_mean+fraction*current
        else:
            updated = weighted_mean
        if np.linalg.norm(updated-current) < 1e-9:
            return updated
        current = updated
    return current


def robust_points(opinions):
    """Return predeclared robust centers; ties do not depend on opinion order."""
    points = np.asarray(opinions, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Four finite White/Black opinion pairs are required.')
    candidates = []
    for selected in combinations(range(4), 3):
        group = points[list(selected)]
        mean = group.mean(axis=0)
        candidates.append((float(np.sum((group-mean)**2)), mean))
    best = min(score for score, _ in candidates)
    tied = [mean for score, mean in candidates if np.isclose(score, best, rtol=1e-12, atol=1e-12)]
    return {'robust_coordinate_median': np.median(points, axis=0),
            'robust_geometric_median': geometric_median(points),
            'robust_trimmed_three_mean': np.mean(tied, axis=0)}


def _proper_points(fit, calibration):
    _, variances, population, between = _population(calibration, ARGS.grid)
    own = fit['diagnostics']['curve']
    accuracy = np.array([fit['players'][side]['average_accuracy'] for side in ('White', 'Black')])
    return posterior(accuracy[:, None], np.asarray(own['shared_accuracy']),
                     own['likelihood']['accuracy_variance'], population,
                     between+variances.mean(), ARGS.grid)['mean']


def predict_detail(evidence, fit, ratings, calibration_cases):
    transformed = weighted_fit(evidence, .5)
    calibration = [{'fit': weighted_fit(case['evidence'], .5)} for case in calibration_cases]
    arithmetic_coverage = universal_native_coverage.predict(evidence, fit, ratings, calibration_cases)[
        'native_coverage_mean_all']
    competitive_coverage = universal_native_coverage.predict(evidence, transformed, ratings, calibration)[
        'native_coverage_mean_all']
    opinions = np.array([[arithmetic_coverage[side] for side in ('White', 'Black')],
                         [competitive_coverage[side] for side in ('White', 'Black')],
                         _proper_points(fit, calibration_cases), _proper_points(transformed, calibration)])
    output, raw = {}, {}
    for name, point in robust_points(opinions).items():
        pair = {side: (1-ACCOUNT_WEIGHT)*float(point[index])+ACCOUNT_WEIGHT*float(ratings[side])
                for index, side in enumerate(('White', 'Black'))}
        raw[name] = pair
        output[name] = project_order(pair, fit)
    return output, raw, opinions


def predict(evidence, fit, ratings, calibration_cases):
    return predict_detail(evidence, fit, ratings, calibration_cases)[0]


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
        predictions.append(predict_detail(**case['input'], calibration_cases=[other['input'] for other in cases if other is not case]))
        print(case['game']+' complete', flush=True)
    rows, diagnostics = [], []
    for case, (prediction, raw, opinions) in zip(cases, predictions, strict=True):
        diagnostics.append({'game': case['game'], 'opinions': opinions.tolist(), 'before_projection': raw})
        for method, pair in prediction.items():
            for side, value in pair.items():
                rows.append({'game': case['game'], 'side': side, 'method': method, 'estimate': value,
                             'reference': case['references'][side], 'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': 20., 'own_elo_span': 40., 'opponent_elo_max_change': 10.})
    ranking = rank(rows)
    rounded_ranking = rank([{**row, 'estimate': round(row['estimate'])} for row in rows])
    output = ROOT/'tests/analysis/output/universal-rating-methods/robust-consensus.json'
    output.write_text(json.dumps({'ranking': ranking, 'rounded_ranking': rounded_ranking,
                                  'players': rows, 'diagnostics': diagnostics, 'assumptions': __doc__,
                                  'sensitivity_note': 'Analytic common-translation andprojection bounds.'}, indent=2), encoding='utf-8')
    print(json.dumps({'ranking': ranking, 'rounded_ranking': rounded_ranking}, indent=2))


if __name__ == '__main__':
    run()
