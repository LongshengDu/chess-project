"""Integrate a uniform prior over competitiveness power before measuring quality.

With b=4p(1-p) and alpha uniform on[0,1], E[b**alpha]=(b-1)/log(b).
This logarithmic-mean weight averages the assumptions of no context discount
and full Bernoulli-information discount without selecting an exponent from labels.
It is a fixed prior expectation of measurement weights, not a posterior learned
from commercial references. Normalization occurs after averaging the raw weights.

The predictive Gaussian/Beta posterior mean plus5% account blend is unchanged.
A separately reported equal-opinion combination with unweighted native-coverage
inference averages two distinct reliability assumptions. The original shared-
curve color order is an explicit least-squares decision constraint.
"""
from __future__ import annotations

from hashlib import blake2b
import json

import numpy as np

from tests.analysis.universal_competitiveness import annotate_probabilities, project_order, _predict_transformed
from tests.analysis.universal_quality_alignment import refit
from tests.analysis import universal_native_coverage


_FIT_CACHE = {}


def marginal_weight(probability):
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Before-position probability must lie in[0,1].')
    base = 4*probability*(1-probability)
    result = np.zeros_like(base)
    interior = (base > 0) & (base < 1)
    result[base == 1] = 1.
    logs = np.log(base[interior])
    result[interior] = np.expm1(logs)/logs
    return result


def marginal_fit(evidence):
    fingerprint = blake2b(digest_size=20)
    adapted = {}
    for side in ('White', 'Black'):
        record = evidence[side]
        observations = []
        fingerprint.update(side.encode('ascii'))
        for row in record['observations']:
            weight = float(marginal_weight(row['position_win_probability']))
            observations.append({**row, 'weight': weight})
            quality = np.asarray(row['qualities']['position'], dtype=float)
            fingerprint.update(np.array([len(quality), row['played_index']], dtype=np.int64).tobytes())
            fingerprint.update(np.array([weight], dtype=float).tobytes())
            fingerprint.update(quality.tobytes())
            fingerprint.update(np.asarray(row['maia_probabilities'], dtype=float).tobytes())
        adapted[side] = {**record, 'observations': observations}
    key = fingerprint.digest()
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = refit(adapted, 'position', weighted=True)
    return _FIT_CACHE[key]


def predict(evidence, fit, ratings, calibration_cases):
    transformed = marginal_fit(evidence)
    calibration = [{'fit': marginal_fit(case['evidence'])} for case in calibration_cases]
    candidate = _predict_transformed(transformed, ratings, calibration, 'marginal')['marginal_mean_account']
    coverage = universal_native_coverage.predict(evidence, fit, ratings, calibration_cases)[
        'native_coverage_mean_account_5pct_all']
    consensus = {side: .5*(candidate[side]+coverage[side]) for side in ('White', 'Black')}
    return {'competitive_marginal_mean_account': candidate,
            'competitive_marginal_mean_account_ordered': project_order(candidate, fit),
            'competitive_marginal_coverage_consensus': project_order(consensus, fit)}


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
                rows.append({'game': case['game'], 'side': side, 'method': method, 'estimate': value,
                             'reference': case['references'][side], 'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': 10., 'own_elo_span': 20.,
                             'opponent_elo_max_change': 0. if method == 'competitive_marginal_mean_account' else 5.})
    ranking = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/competitive-marginal.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'ranking': ranking, 'players': rows, 'assumptions': __doc__,
                                  'sensitivity_note': 'Analytic upper bounds from5%blend andorderprojection.'}, indent=2), encoding='utf-8')
    print(json.dumps(ranking, indent=2))


if __name__ == '__main__':
    run()
