"""One declared coarse competitiveness sensitivity check, never an optimizer.

Exponents0.25,0.5,0.75 are evaluated once with every other choice held fixed:
weighted observed/synthetic arithmetic accuracy, equal Gaussian/Beta model priors,
posterior mean,5% account blend and optional original-accuracy order projection.
The study is exploratory model sensitivity; choosing a variant by this same
commercial-reference table is not independent out-of-sample validation.
"""
from __future__ import annotations

import json

from tests.analysis.universal_competitiveness import (
    annotate_probabilities, project_order, weighted_fit, _predict_transformed,
)


POWERS = (.25, .5, .75)


def predict(evidence, fit, ratings, calibration_cases):
    output = {}
    for power in POWERS:
        name = f'competitive_power{power:g}_mean_account'
        transformed_fit = weighted_fit(evidence, power)
        calibration = [{'fit': weighted_fit(case['evidence'], power)} for case in calibration_cases]
        prediction = _predict_transformed(transformed_fit, ratings, calibration, 'candidate')['candidate_mean_account']
        output[name] = prediction
        output[name+'_ordered'] = project_order(prediction, fit)
    return output


def run():
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
                base_method = method.removesuffix('_ordered')
                perturbations = {}
                for changed_side in ('White', 'Black'):
                    perturbations[changed_side] = []
                    for shift in (-200, -100, 100, 200):
                        shifted = dict(prediction[base_method])
                        shifted[changed_side] += .05*shift
                        if method.endswith('_ordered'):
                            shifted = project_order(shifted, case['input']['fit'])
                        perturbations[changed_side].append(shifted[side])
                other = 'Black' if side == 'White' else 'White'
                rows.append({'game': case['game'], 'side': side, 'method': method, 'estimate': value,
                             'reference': case['references'][side], 'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': max(abs(p-value) for p in perturbations[side]),
                             'own_elo_span': max(perturbations[side]+[value])-min(perturbations[side]+[value]),
                             'opponent_elo_max_change': max(abs(p-value) for p in perturbations[other])})
    ranking = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/competitive-coarse-sensitivity.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'ranking': ranking, 'players': rows, 'powers': POWERS,
                                  'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(ranking, indent=2))


if __name__ == '__main__':
    run()
