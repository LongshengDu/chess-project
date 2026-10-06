"""Audit consistent parent-search quality and volatility-weighted arithmetic fits.

The three fixed evidence choices are root/unweighted, position/volatility, and
root/volatility. All retain complete legal Maia policies, the existing prior,
accuracy likelihood, sigma scale and curve/tail construction. Volatility uses
the already stored Lichess weight without a newly fitted exponent or threshold.
The unchanged proper and monotone mixture models are applied universally after
rebuilding both the target fit and every leave-game-out calibration fit.

Volatility weights describe the actual observed trajectory. They can emphasize
critical positions but are not independent of future played moves, an explicit
limitation for counterfactual Maia interpretation.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
from scipy.optimize import isotonic_regression

from analysis.player_rating.bayesian_shared_curve import ARGS, SharedCurve, curve_posterior, summarize
from tests.analysis import edge_monotone_quality, edge_predictive_mixture


VARIANTS = {'root_arithmetic': ('root', False),
            'position_volatility': ('position', True),
            'root_volatility': ('root', True)}


def refit(evidence, view, weighted):
    """Keep production code untouched; replace only quality source/aggregation."""
    adapted = {side: {**record, 'observations': [
        {**row, 'qualities': {**row['qualities'], 'position': row['qualities'][view]}}
        for row in record['observations']]} for side, record in evidence.items()}
    fit = summarize(adapted)
    if not weighted:
        return fit
    moments = {}
    for side, record in adapted.items():
        rows = [row for row in record['observations'] if len(row['qualities']['position']) > 1]
        if not rows:
            moments[side] = None
            continue
        weights = np.array([row['weight'] for row in rows], dtype=float)
        weights /= weights.sum()
        means, variances, observed = [], [], []
        for row in rows:
            q = np.asarray(row['qualities']['position'])
            p = np.asarray(row['maia_probabilities'])
            mean = p @ q
            means.append(mean)
            variances.append(np.maximum(0., p @ (q*q) - mean*mean))
            observed.append(q[row['played_index']])
        moments[side] = {'mean': weights @ means, 'variance': weights**2 @ variances,
                         'observed': float(weights @ observed)}
    available = [moment for moment in moments.values() if moment is not None]
    if not available:
        return fit
    expected = np.mean([moment['mean'] for moment in available], axis=0)
    monotone = np.clip(isotonic_regression(expected).x, 0., 100.)
    curve = SharedCurve(monotone)(ARGS.grid)
    variance = float(np.mean([moment['variance'].mean() for moment in available]))
    diagnostic = fit['diagnostics']['curve']
    diagnostic.update(maia_expected_accuracy=expected.tolist(), monotone_expected_accuracy=monotone.tolist(),
                      shared_accuracy=curve.tolist(), shared_mean_variance=variance)
    diagnostic['likelihood'].update(accuracy_variance=variance, accuracy_sigma=float(np.sqrt(variance)))
    for side, moment in moments.items():
        if moment is None:
            continue
        posterior = curve_posterior(moment['observed'], curve, variance)
        fit['players'][side].update(estimate=posterior['estimate'],
                                    unrounded_estimate=posterior['unrounded_estimate'],
                                    average_accuracy=moment['observed'])
        diagnostic['posterior_densities'][side] = posterior['posterior_density']
    return fit


def _predict_refitted(data, calibration, prefix):
    outputs = {prefix+'_baseline_all': {
        side: data['fit']['players'][side]['unrounded_estimate'] for side in ('White', 'Black')}}
    outputs[prefix+'_baseline_account_5pct_all'] = {
        side: .95*value+.05*data['ratings'][side] for side, value in outputs[prefix+'_baseline_all'].items()}
    for module in (edge_predictive_mixture, edge_monotone_quality):
        outputs.update({prefix+'_'+name: pair for name, pair in module.predict(
            data['evidence'], data['fit'], data['ratings'], calibration).items()})
    return outputs


def predict(evidence, fit, ratings, calibration_cases):
    del fit
    output = {}
    for name, (view, weighted) in VARIANTS.items():
        data = {'evidence': evidence, 'fit': refit(evidence, view, weighted), 'ratings': ratings}
        calibration = [{**case, 'fit': refit(case['evidence'], view, weighted)} for case in calibration_cases]
        output.update(_predict_refitted(data, calibration, name))
    return output


def main():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases, rank
    from tests.analysis.edge_quality_likelihood import is_edge
    cases, _ = load_cases(ROOT/'games')
    predictions = [{} for _ in cases]
    audit = []
    for case in cases:
        evidence = case['input']['evidence']
        for side, record in evidence.items():
            observations = [row for row in record['observations'] if len(row['qualities']['position']) > 1]
            played_difference = [row['qualities']['root'][row['played_index']]
                                 -row['qualities']['position'][row['played_index']] for row in observations]
            q_differences = [np.asarray(row['qualities']['root'])-row['qualities']['position'] for row in observations]
            max_alternative_difference = max((abs(difference[j]) for row, difference in zip(observations, q_differences)
                                              for j in range(len(difference)) if j != row['played_index']), default=0.)
            audit.append({'game': case['game'], 'side': side,
                          'root_minus_position_played_mean': float(np.mean(played_difference)),
                          'played_move_mean_absolute_change': float(np.mean(np.abs(played_difference))),
                          'maximum_alternative_change': float(max_alternative_difference),
                          'changed_played_moves': int(np.count_nonzero(np.abs(played_difference) > 1e-8))})
    for name, (view, weighted) in VARIANTS.items():
        transformed = [{**case['input'], 'fit': refit(case['input']['evidence'], view, weighted)} for case in cases]
        for index, data in enumerate(transformed):
            calibration = transformed[:index]+transformed[index+1:]
            predictions[index].update(_predict_refitted(data, calibration, name))
        print(f'Completed {name}.', flush=True)
    rows = []
    for case, prediction in zip(cases, predictions, strict=True):
        for method, pair in prediction.items():
            account = '_account_' in method
            for side, value in pair.items():
                rows.append({'game': case['game'], 'side': side, 'method': method,
                             'estimate': value, 'reference': case['references'][side],
                             'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': 10. if account else 0., 'own_elo_span': 20. if account else 0.,
                             'opponent_elo_max_change': 0.})
    ranked = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/quality-alignment.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'rankings': ranked, 'rows': rows, 'quality_audit': audit,
                                  'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(ranked, indent=2))
    print(output)


if __name__ == '__main__':
    main()
