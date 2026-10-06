"""Replace exponential accuracy by inverse-mapped win-probability retention.

Lichess move accuracy is clamp(A*exp(-k*loss)-B,0,100), with constants copied
exactly from analysis.lichess_accuracy.move_accuracy (including its +1 offset).
For0<accuracy<100 the transformation100+log((accuracy+B)/A)/k recovers100 minus
win-percentage loss exactly. Ataccuracy100 the clamp hides gains and smalllosses;
the minimum consistent nonnegative loss is0. Ataccuracy0 the true loss is only
known to exceed -log(B/A)/k; the experiment uses this minimum consistent loss.
Thus endpoint values are conservative censored estimates, not exact inversions.

Original arithmetic and fixed square-root competitive weighting are tested with
the same full legal Maia distributions and rebuilt leave-game-out population
curves. Both use the existing equal-prior Gaussian/Beta likelihood mixture,
posterior mean and fixed5% account blend. Optional original-accuracy order
projection is reported separately. No inverse constants or weights are fitted.
"""
from __future__ import annotations

import json

import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import posterior
from tests.analysis.universal_competitiveness import annotate_probabilities, project_order, weighted_fit


ACCURACY_AMPLITUDE = 103.1668100711649
ACCURACY_OFFSET = 3.166924740191411 - 1.
ACCURACY_DECAY = .04354415386753951
ZERO_ACCURACY_MINIMUM_LOSS = -np.log(ACCURACY_OFFSET/ACCURACY_AMPLITUDE)/ACCURACY_DECAY
PERFECT_ACCURACY_MAXIMUM_LOSS = -np.log((100+ACCURACY_OFFSET)/ACCURACY_AMPLITUDE)/ACCURACY_DECAY
VARIANTS = {'outcome_arithmetic': 0., 'outcome_competitive_sqrt': .5}


def outcome_quality(accuracy):
    """Invert uncensored accuracy, and use minimum consistent endpoint losses."""
    q = np.asarray(accuracy, dtype=float)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 100)):
        raise ValueError('Lichess move accuracy must lie within0--100.')
    loss = -np.log((q+ACCURACY_OFFSET)/ACCURACY_AMPLITUDE)/ACCURACY_DECAY
    loss = np.where(q >= 100., 0., loss)
    return np.clip(100-loss, 0., 100.)


def _outcome_evidence(evidence):
    return {side: {**record, 'observations': [
        {**row, 'qualities': {view: outcome_quality(values).tolist() for view, values in row['qualities'].items()}}
        for row in record['observations']]} for side, record in evidence.items()}


def predict(evidence, fit, ratings, calibration_cases):
    """Use the same transformed measurement and likelihood for every player."""
    transformed = _outcome_evidence(evidence)
    calibration = [_outcome_evidence(case['evidence']) for case in calibration_cases]
    outputs = {}
    for name, power in VARIANTS.items():
        local = weighted_fit(transformed, power)
        other_fits = [{'fit': weighted_fit(other, power)} for other in calibration]
        _, variances, population, between = _population(other_fits, ARGS.grid)
        own = local['diagnostics']['curve']
        outputs[name+'_mean_account'] = {}
        for side in ('White', 'Black'):
            estimate = posterior(local['players'][side]['average_accuracy'],
                                  np.asarray(own['shared_accuracy']), own['likelihood']['accuracy_variance'],
                                  population, between+variances.mean(), ARGS.grid)
            outputs[name+'_mean_account'][side] = .95*float(estimate['mean'])+.05*float(ratings[side])
    outputs.update({name+'_ordered': project_order(pair, fit) for name, pair in list(outputs.items())})
    return outputs


def main():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, read_json, rank
    from tests.analysis.edge_quality_likelihood import is_edge
    cases, _ = load_cases(ROOT/'games')
    censoring = []
    for case in cases:
        moves = read_json(ROOT/'games/output'/f'{case["game"]}-full'/'analysis.json')['moves']
        case['input']['evidence'] = annotate_probabilities(case['input']['evidence'], moves)
        for side, record in case['input']['evidence'].items():
            rows = [row for row in record['observations'] if len(row['qualities']['position']) > 1]
            played = np.array([row['qualities']['position'][row['played_index']] for row in rows])
            candidate_zero_mass = np.mean([np.asarray(row['maia_probabilities'])
                                          @ (np.asarray(row['qualities']['position']) == 0.) for row in rows], axis=0)
            censoring.append({'game': case['game'], 'side': side, 'played_moves': len(rows),
                              'played_zero_accuracy': int(np.sum(played == 0.)),
                              'played_perfect_accuracy': int(np.sum(played == 100.)),
                              'maximum_mean_maia_zero_accuracy_mass': float(np.max(candidate_zero_mass))})
    predictions = [predict(**case['input'], calibration_cases=[other['input'] for other in cases if other is not case])
                   for case in cases]
    rows = []
    for case, prediction in zip(cases, predictions, strict=True):
        for method, pair in prediction.items():
            for side, value in pair.items():
                sensitivity = {}
                for changed_side in ('White', 'Black'):
                    sensitivity[changed_side] = [value]
                    for shift in (-200., -100., 100., 200.):
                        changed = dict(prediction[method.removesuffix('_ordered')])
                        changed[changed_side] += .05*shift
                        if method.endswith('_ordered'):
                            changed = project_order(changed, case['input']['fit'])
                        sensitivity[changed_side].append(changed[side])
                other = 'Black' if side == 'White' else 'White'
                rows.append({'game': case['game'], 'side': side, 'method': method, 'estimate': value,
                             'reference': case['references'][side], 'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': float(np.max(abs(np.array(sensitivity[side])-value))),
                             'own_elo_span': float(np.ptp(sensitivity[side])),
                             'opponent_elo_max_change': float(np.max(abs(np.array(sensitivity[other])-value)))})
    ranked = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/outcome-quality.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'rankings': ranked, 'rows': rows, 'censoring': censoring,
                                  'zero_accuracy_minimum_loss': float(ZERO_ACCURACY_MINIMUM_LOSS),
                                  'perfect_accuracy_maximum_loss': float(PERFECT_ACCURACY_MAXIMUM_LOSS),
                                  'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(ranked, indent=2))
    print(output)


if __name__ == '__main__':
    main()
