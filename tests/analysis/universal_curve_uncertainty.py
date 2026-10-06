"""Universal shared latent curve reliability; label-free exploratory estimator.

A common game-level reliability lambda mixes the local and leave-game-out
population curves. Integrating uniform lambda expresses uncertainty about how
strongly the local Maia context predicts human game quality, with no edge switch.
The two players jointly update lambda but retain independent rating priors.

For any fixed lambda, the mean curve is monotone and Gaussian variance constant
in rating. Both players use that same likelihood and prior. Their conditional
posteriors are stochastically ordered by observed accuracy; integrating with one
common lambda posterior preserves the game's White/Black ordering. A separately
labelled 5% actual-rating blend has 10-Elo sensitivity per 200 account Elo.

Three predeclared coordinates test Gaussian accuracy, variance-stabilizing angular
accuracy and regularized log loss. Noise transformations use a delta approximation
over the measured curve; constants/quadrature are not reference calibrated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import betainc, betaincc

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID, prior_density
from tests.analysis.edge_global_quality import _population


ACCOUNT_WEIGHT = .05
COORDINATES = ('accuracy', 'angular', 'log_loss')
RELIABILITY_NODES = 12


def _transform(quality, coordinate):
    quality = np.clip(np.asarray(quality, dtype=float), 0., 100.)
    if coordinate == 'accuracy':
        return quality
    if coordinate == 'angular':
        return np.arcsin(np.sqrt(quality/100.))
    if coordinate == 'log_loss':
        return -np.log((101.-quality)/101.)
    raise ValueError('Unknown quality coordinate.')


def _derivative(quality, coordinate):
    quality = np.clip(np.asarray(quality, dtype=float), .5, 99.5)
    if coordinate == 'accuracy':
        return np.ones_like(quality)
    if coordinate == 'angular':
        return 1/(2*np.sqrt(quality*(100-quality)))
    return 1/(101-quality)


def _shared_reliability(accuracies, local, local_variance, population, population_variance,
                        grid, *, integrate=True, beta=False, context=None, point='median'):
    if integrate:
        nodes, weights = leggauss(RELIABILITY_NODES)
        reliability, weights = (nodes+1)/2, weights/2
    else:
        reliability, weights = np.array([.5]), np.array([1.])
    if context is not None:
        offset, between_variance, measurement_variance = context
        # Student-t(4) random-effect scale mixture. Observed context offset is
        # noisy; heavy tails avoid shrinking unusually hard/easy contexts away.
        ratio = max(measurement_variance, 1e-12)/max(between_variance, 1e-12)
        scale = ratio*reliability/(1-reliability)
        context_variance = max(measurement_variance, 1e-12)/(1-reliability)
        log_prior = (-3*np.log(scale)-2/scale+np.log(ratio)-2*np.log1p(-reliability)
                     -.5*(offset**2/context_variance+np.log(context_variance)))
        weights = weights*np.exp(log_prior-log_prior.max())
        weights /= weights.sum()
    means = reliability[:, None]*local+(1-reliability[:, None])*population
    variances = reliability*local_variance+(1-reliability)*population_variance
    variances = np.maximum(variances, 1e-12)
    if beta:
        probability = np.clip(means/100, 1e-9, 1-1e-9)
        native = (grid >= GRID[0]) & (grid <= GRID[-1])
        concentration = np.maximum(np.mean(probability[:, native]*(1-probability[:, native]), axis=1)
                                   /(variances/10000)-1, 1e-6)
        alpha = probability*concentration[:, None]
        beta_shape = (1-probability)*concentration[:, None]
        lower = np.maximum(0, accuracies[:, None, None]-.005)/100
        upper = np.minimum(100, accuracies[:, None, None]+.005)/100
        lower_cdf = betainc(alpha, beta_shape, lower)
        upper_cdf = betainc(alpha, beta_shape, upper)
        mass = np.where(lower_cdf < .5, upper_cdf-lower_cdf,
                        betaincc(alpha, beta_shape, lower)-betaincc(alpha, beta_shape, upper))
        logs = np.log(np.maximum(mass, 1e-300))
    else:
        logs = -.5*((accuracies[:, None, None]-means[None, :, :])**2/variances[None, :, None]
                    +np.log(2*np.pi*variances)[None, :, None])
    # One shift per side retains each lambda's relative integrated evidence.
    likelihood = np.exp(logs-logs.max(axis=(1, 2), keepdims=True))
    unnormalized = likelihood*prior_density(grid)[None, None, :]
    evidence = trapezoid(unnormalized, grid, axis=2)
    conditional = unnormalized/np.maximum(evidence[:, :, None], 1e-300)
    reliability_mass = weights*np.prod(evidence, axis=0)
    reliability_mass /= reliability_mass.sum()
    marginals = np.sum(conditional*reliability_mass[None, :, None], axis=1)
    cdfs = cumulative_trapezoid(marginals, grid, axis=1, initial=0.)
    medians = [float(np.interp(.5, row/row[-1], grid)) for row in cdfs]
    estimates = medians if point == 'median' else trapezoid(marginals*grid, grid, axis=1).tolist()
    return estimates, float(reliability_mass@reliability)


def predict(evidence, fit, ratings, calibration_cases):
    """Predict both players for every game, using other games only for pooling."""
    del evidence
    grid = ARGS.grid
    curves, calibration_variances, pooled, between = _population(calibration_cases, grid)
    own = fit['diagnostics']['curve']
    local = np.asarray(own['shared_accuracy'])
    local_variance = float(own['likelihood']['accuracy_variance'])
    measured = (grid >= GRID[0]) & (grid <= GRID[-1])
    observations = np.asarray([fit['players'][side]['average_accuracy'] for side in ('White', 'Black')])
    if not np.isfinite(observations).all():
        raise ValueError('Both players need observed average accuracies.')
    result = {}
    for coordinate in COORDINATES:
        transformed_local = _transform(local, coordinate)
        transformed_population_curves = _transform(curves, coordinate)
        transformed_population = transformed_population_curves.mean(axis=0)
        target_variance = local_variance*np.mean(_derivative(local[measured], coordinate)**2)
        within = calibration_variances[:, None]*_derivative(curves[:, measured], coordinate)**2
        between = np.var(transformed_population_curves[:, measured], axis=0, ddof=1)
        population_variance = float(within.mean()+between.mean())
        estimates, _ = _shared_reliability(_transform(observations, coordinate), transformed_local,
                                          target_variance, transformed_population, population_variance, grid)
        name = 'latent_curve_'+coordinate
        result[name] = dict(zip(('White', 'Black'), estimates, strict=True))
        result[name+'_account'] = {
            side: (1-ACCOUNT_WEIGHT)*value+ACCOUNT_WEIGHT*float(ratings[side])
            for side, value in result[name].items()}
    population_variance = float(np.mean(np.var(curves[:, measured], axis=0, ddof=1))
                                +np.mean(calibration_variances))
    estimates, _ = _shared_reliability(observations, local, local_variance, pooled,
                                      population_variance, grid, beta=True)
    result['latent_curve_beta'] = dict(zip(('White', 'Black'), estimates, strict=True))
    result['latent_curve_beta_account'] = {
        side: (1-ACCOUNT_WEIGHT)*value+ACCOUNT_WEIGHT*float(ratings[side])
        for side, value in result['latent_curve_beta'].items()}
    context = (float(np.mean(local[measured]-pooled[measured])),
               float(np.var(curves[:, measured], axis=0, ddof=1).mean()), local_variance)
    for beta in (False, True):
        estimates, _ = _shared_reliability(observations, local, local_variance, pooled,
                                          population_variance, grid, beta=beta, context=context)
        name = 'latent_curve_random_effects'+('_beta' if beta else '')
        result[name] = dict(zip(('White', 'Black'), estimates, strict=True))
        result[name+'_account'] = {
            side: (1-ACCOUNT_WEIGHT)*value+ACCOUNT_WEIGHT*float(ratings[side])
            for side, value in result[name].items()}
    for beta in (False, True):
        for use_context in (False, True):
            estimates, _ = _shared_reliability(observations, local, local_variance, pooled,
                                              population_variance, grid, beta=beta,
                                              context=context if use_context else None, point='mean')
            name = 'latent_curve_mean'+('_beta' if beta else '')+('_random_effects' if use_context else '')
            result[name] = dict(zip(('White', 'Black'), estimates, strict=True))
            result[name+'_account'] = {
                side: (1-ACCOUNT_WEIGHT)*value+ACCOUNT_WEIGHT*float(ratings[side])
                for side, value in result[name].items()}
    return result


def run():
    """Score only after all predictions; output isolated research artifacts."""
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from tests.analysis.experiment_edge_rating import ROOT, load_cases
    cases, _ = load_cases(ROOT/'games')
    predictions = []
    for case in cases:
        calibration = [other['input'] for other in cases if other is not case]
        predictions.append(predict(**case['input'], calibration_cases=calibration))
    ranking, rows = [], []
    for name in predictions[0]:
        for case, prediction in zip(cases, predictions, strict=True):
            for side in ('White', 'Black'):
                player = case['input']['fit']['players'][side]
                curve = case['input']['fit']['diagnostics']['curve']['monotone_expected_accuracy']
                rows.append({'game': case['game'], 'side': side, 'method': name,
                             'estimate': prediction[name][side], 'reference': case['references'][side],
                             'edge': not curve[0] <= player['average_accuracy'] <= curve[-1]})
        selected = [r for r in rows if r['method'] == name]
        errors = [abs(r['estimate']-r['reference']) for r in selected]
        ranking.append({'method': name, 'mae': float(np.mean(errors)), 'max_error': max(errors),
                        'edge_mae': float(np.mean([abs(r['estimate']-r['reference']) for r in selected if r['edge']])),
                        'ordering': sum(np.sign(selected[i]['estimate']-selected[i+1]['estimate']) ==
                                        np.sign(selected[i]['reference']-selected[i+1]['reference'])
                                        for i in range(0, len(selected), 2))})
    ranking.sort(key=lambda r: r['mae'])
    output = ROOT/'tests/analysis/output/universal-rating-methods/latent-curve-uncertainty.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'ranking': ranking, 'players': rows}, indent=2, default=int), encoding='utf-8')
    print(json.dumps(ranking, indent=2, default=int))


if __name__ == '__main__':
    run()
