"""Universal information-based opinion weights, with no fitted coefficients.

For each side, square-root competitive weights give Kish effective sample size
n_eff=(sum w)^2/sum(w^2). Average n_eff/n across both sides to obtain one game-
level reliability r. Compare competitive-opinion weights r and r/(1+r): direct
reliability versus relative information against unweighted information normalized
to one. Both are declared fixed statistical assumptions, not optimized weights.

Combine account-free arithmetic native coverage and competitive predictive means,
then apply the existing10% account blend and exactly one original-order constraint.
The weight is independent of actual Elo and commercial estimates. Effective count
does not model serial dependence or bias; this is an exploratory information proxy.
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


def paired_reliability(evidence):
    fractions, counts = [], {}
    for side in ('White', 'Black'):
        probabilities = np.array([row['position_win_probability'] for row in evidence[side]['observations']
                                  if len(row['qualities']['position']) > 1], dtype=float)
        if not len(probabilities) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError('Finite before-position probabilities and non-forced moves are required.')
        weights = np.sqrt(4*probabilities*(1-probabilities))
        if not np.any(weights > 0):
            raise ValueError('At least one move must have positive competitive information.')
        effective = float(weights.sum()**2/np.sum(weights**2))
        fraction = effective/len(weights)
        fractions.append(fraction)
        counts[side] = {'moves': len(weights), 'effective_moves': effective, 'fraction': fraction}
    return float(np.mean(fractions)), counts


def predict_detail(evidence, fit, ratings, calibration_cases):
    reliability, counts = paired_reliability(evidence)
    coverage = universal_native_coverage.predict(evidence, fit, {'White': 0., 'Black': 0.}, calibration_cases)[
        'native_coverage_mean_all']
    transformed = weighted_fit(evidence, .5)
    calibration = [{'fit': weighted_fit(case['evidence'], .5)} for case in calibration_cases]
    _, variances, population, between = _population(calibration, ARGS.grid)
    own = transformed['diagnostics']['curve']
    observed = np.array([transformed['players'][side]['average_accuracy'] for side in ('White', 'Black')])
    means = posterior(observed[:, None], np.asarray(own['shared_accuracy']),
                      own['likelihood']['accuracy_variance'], population,
                      between+variances.mean(), ARGS.grid)['mean']
    competitive = dict(zip(('White', 'Black'), means, strict=True))
    choices = {'effective_sample_direct': reliability,
               'effective_sample_relative': reliability/(1+reliability)}
    output, raw = {}, {}
    for name, weight in choices.items():
        quality = {side: (1-weight)*coverage[side]+weight*competitive[side] for side in ('White', 'Black')}
        pair = {side: (1-ACCOUNT_WEIGHT)*quality[side]+ACCOUNT_WEIGHT*float(ratings[side]) for side in quality}
        raw[name] = pair
        output[name] = project_order(pair, fit)
    return output, {'shared_reliability': reliability, 'counts': counts,
                    'competitive_weights': choices, 'before_projection': raw}


def predict(evidence, fit, ratings, calibration_cases):
    return predict_detail(evidence, fit, ratings, calibration_cases)[0]


def run():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, read_json
    from tests.analysis.experiment_universal_rating import score
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
    for case, (prediction, diagnostic) in zip(cases, predictions, strict=True):
        diagnostics.append({'game': case['game'], **diagnostic})
        for method, pair in prediction.items():
            for side, estimate in pair.items():
                rows.append({'game': case['game'], 'side': side, 'method': method, 'estimate': estimate,
                             'reference': case['references'][side], 'edge': bool(is_edge(case['input']['fit'], side)),
                             'original_estimate': case['input']['fit']['players'][side]['estimate']})
    ranking = score(rows)
    for entry in ranking:
        edges = [row for row in rows if row['method'] == entry['method'] and row['edge']]
        entry['edge_ordering_pairs'] = int(sum(np.sign(first['estimate']-second['estimate']) ==
                                              np.sign(first['reference']-second['reference'])
                                              for first, second in combinations(edges, 2)))
        entry['edge_ordering_total'] = len(edges)*(len(edges)-1)//2
        entry['own_elo_change_bound'] = 20.
        entry['opponent_elo_change_bound'] = 10.
    output = ROOT/'tests/analysis/output/universal-rating-methods/effective-sample-consensus.json'
    output.write_text(json.dumps({'ranking': ranking, 'players': rows, 'diagnostics': diagnostics,
                                  'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(ranking, indent=2))


if __name__ == '__main__':
    run()
