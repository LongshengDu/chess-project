"""Paired nuisance cancellation: new absolute location with old relative contrast.

A common game-context offset changes both player estimates together, while the
White-minus-Black contrast cancels it to first order. Keep the universal consensus
pair center, and use either the original shared-curve contrast or an equal opinion
of original/new contrasts. These are four fixed rules, not fitted combinations:
two previously declared centers crossed with two paired-design contrast choices.

The original contrast uses the production unrounded posterior medians and no
account Elo. This exploits a common-context assumption; it is not a guarantee
that differences are accurate, and preserves known original ordering by design.
"""
from __future__ import annotations

import json

from tests.analysis import universal_native_coverage, universal_competitiveness
from tests.analysis.universal_competitiveness import annotate_probabilities, project_order


CENTERS = {'coverage25': .25, 'coverage50': .5}


def combine_pair(new_pair, fit, old_weight):
    center = (new_pair['White']+new_pair['Black'])/2
    old_difference = (fit['players']['White']['unrounded_estimate']
                      -fit['players']['Black']['unrounded_estimate'])
    new_difference = new_pair['White']-new_pair['Black']
    difference = old_weight*old_difference+(1-old_weight)*new_difference
    return {'White': center+difference/2, 'Black': center-difference/2}


def transform_centers(centers, fit):
    result = {}
    for name, pair in centers.items():
        result['paired_'+name+'_old_contrast'] = combine_pair(pair, fit, 1.)
        result['paired_'+name+'_equal_contrast'] = combine_pair(pair, fit, .5)
    return result


def predict(evidence, fit, ratings, calibration_cases):
    coverage = universal_native_coverage.predict(evidence, fit, ratings, calibration_cases)[
        'native_coverage_mean_account_5pct_all']
    competition = universal_competitiveness.predict(evidence, fit, ratings, calibration_cases)[
        'competitive_sqrt_mean_account']
    centers = {name: project_order({side: weight*coverage[side]+(1-weight)*competition[side]
                                    for side in ('White', 'Black')}, fit)
               for name, weight in CENTERS.items()}
    return transform_centers(centers, fit)


def run():
    """Transform previously saved model points, then score commercial references."""
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, rank, read_json
    from tests.analysis.edge_quality_likelihood import is_edge
    cases, _ = load_cases(ROOT/'games')
    source = read_json(ROOT/'tests/analysis/output/universal-rating-methods/consensus-coarse-sensitivity.json')
    saved_rows = source.get('players', source.get('rows'))
    if saved_rows is None:
        raise ValueError('Coarse consensus saved player predictions are required.')
    lookup = {(r['game'], r['side'], r['method']): r['estimate'] for r in saved_rows}
    predictions = []
    for case in cases:
        centers = {name: {side: lookup[case['game'], side, f'consensus_arithmetic_coverage{int(weight*100)}']
                           for side in ('White', 'Black')} for name, weight in CENTERS.items()}
        predictions.append(transform_centers(centers, case['input']['fit']))
    rows = []
    for case, prediction in zip(cases, predictions, strict=True):
        for method, pair in prediction.items():
            for side, estimate in pair.items():
                rows.append({'game': case['game'], 'side': side, 'method': method, 'estimate': estimate,
                             'reference': case['references'][side], 'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': 5. if method.endswith('_old_contrast') else 7.5,
                             'own_elo_span': 10. if method.endswith('_old_contrast') else 15.,
                             'opponent_elo_max_change': 5.})
    ranking = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/paired-center-contrast.json'
    output.write_text(json.dumps({'ranking': ranking, 'players': rows, 'assumptions': __doc__,
                                  'sensitivity_note': 'Analytic5%account/projection bounds.',
                                  'source': 'Saved coarse-consensus predictions; only estimates consumed.'}, indent=2), encoding='utf-8')
    print(json.dumps(ranking, indent=2))


if __name__ == '__main__':
    run()
