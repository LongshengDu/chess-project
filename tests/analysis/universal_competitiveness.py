"""Universal competitive-position weighting of observed and synthetic accuracy.

Before-move Stockfish winning chances p determine w=4p(1-p), the Bernoulli
variance relative to its equal-position maximum. A square-root alternative is
predeclared as less aggressive downweighting. These weights emphasize undecided
positions and reduce easy high-accuracy moves in already decided positions.

The existing project win-percent mapping clips centipawns at+/-1000, so weights
remain strictly positive even for mate scores. Both observed accuracy and every
Maia expected curve use identical normalized weights; sampling variance is the
sum of squared normalized weights times per-position conditional variance.
No scores after the move, future volatility, commercial references or account
ratings determine weights. Actual Elo enters only the fixed5% final blend.
"""
from __future__ import annotations

from hashlib import blake2b
import json

import numpy as np

from analysis.lichess_accuracy import win_percent
from analysis.position_evaluation import centipawns
from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior
from tests.analysis.universal_quality_alignment import refit


VARIANTS = {'competitive': 1., 'competitive_sqrt': .5}
_FIT_CACHE = {}


def annotate_probabilities(evidence, moves):
    """Attach only the before-position probability, matching every side's rows."""
    probabilities = {'White': [], 'Black': []}
    for move in moves:
        probability = win_percent(centipawns(move['position_eval']))
        if probability is None:
            raise ValueError('Saved before-position evaluation is required.')
        probabilities[move['side'].title()].append(probability/100.)
    adapted = {}
    for side, record in evidence.items():
        if len(record['observations']) != len(probabilities[side]):
            raise ValueError('Saved position evaluations must align with every evidence move.')
        adapted[side] = {**record, 'observations': [
            {**row, 'position_win_probability': probability}
            for row, probability in zip(record['observations'], probabilities[side], strict=True)]}
    return adapted


def weighted_fit(evidence, power):
    """Cache fits by inference data only; no account/reference metadata is hashed."""
    fingerprint = blake2b(digest_size=20)
    fingerprint.update(np.array([power], dtype=float).tobytes())
    adapted = {}
    for side in ('White', 'Black'):
        record = evidence[side]
        observations = []
        fingerprint.update(side.encode('ascii'))
        for row in record['observations']:
            probability = float(row['position_win_probability'])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Before-position winning probability must lie in[0,1].')
            weight = (4*probability*(1-probability))**power
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


def _predict_transformed(fit, ratings, calibration, prefix):
    _, variances, population, between = _population(calibration, ARGS.grid)
    population_variance = between+variances.mean()
    own = fit['diagnostics']['curve']
    output = {prefix+'_mean_account': {}, prefix+'_median_account': {}}
    for side in ('White', 'Black'):
        value = posterior(fit['players'][side]['average_accuracy'], np.asarray(own['shared_accuracy']),
                          own['likelihood']['accuracy_variance'], population, population_variance, ARGS.grid)
        for point in ('mean', 'median'):
            output[prefix+'_'+point+'_account'][side] = .95*float(value[point])+.05*float(ratings[side])
    return output


def project_order(pair, fit):
    """Least-squares projection onto the original observed-accuracy ordering.

The one-Elo gap represents displayed rating resolution. This enforces a supplied
ordering constraint; it is not independent evidence that the ordering is correct.
    """
    direction = np.sign(fit['players']['White']['average_accuracy']-fit['players']['Black']['average_accuracy'])
    difference = pair['White']-pair['Black']
    if direction*difference >= 1:
        return dict(pair)
    middle = (pair['White']+pair['Black'])/2
    return {'White': float(middle+.5*direction), 'Black': float(middle-.5*direction)}


def predict(evidence, fit, ratings, calibration_cases):
    """All rows require before-position probabilities from annotate_probabilities."""
    result = {}
    for name, power in VARIANTS.items():
        transformed_fit = weighted_fit(evidence, power)
        calibration = [{'fit': weighted_fit(case['evidence'], power)} for case in calibration_cases]
        result.update(_predict_transformed(transformed_fit, ratings, calibration, name))
    result.update({name+'_ordered': project_order(pair, fit) for name, pair in list(result.items())})
    return result


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
    rows, audit = [], []
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
        for name, power in VARIANTS.items():
            transformed = weighted_fit(case['input']['evidence'], power)
            for side in ('White', 'Black'):
                record = case['input']['evidence'][side]
                probabilities = np.array([row['position_win_probability'] for row in record['observations']
                                          if len(row['qualities']['position']) > 1])
                weights = (4*probabilities*(1-probabilities))**power
                audit.append({'game': case['game'], 'side': side, 'weighting': name,
                              'original_accuracy': case['input']['fit']['players'][side]['average_accuracy'],
                              'weighted_accuracy': transformed['players'][side]['average_accuracy'],
                              'minimum_weight': float(weights.min()), 'maximum_weight': float(weights.max()),
                              'effective_moves': float(weights.sum()**2/np.sum(weights**2))})
    ranking = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/competitive-positions.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'ranking': ranking, 'players': rows, 'measurement_audit': audit,
                                  'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(ranking, indent=2))


if __name__ == '__main__':
    run()
