"""Bounded joint accuracy using Beta marginals and a Gaussian copula.

Both arithmetic and square-root competitive accuracies have moment-matched Beta
marginals. A Gaussian copula retains positive dependence. The Pearson correlation
from the Maia-derived quality moments is used as the latent Gaussian correlation:
this is an explicit approximation, not exact matching of the resulting Pearson
covariance after the nonlinear Beta transforms. No reference-derived adjustment
or optimized correlation coefficient is used.

The local and leave-game-out population likelihoods have equal prior weights.
The common0.01-point accuracy bins supply finite marginal probabilities at100;
the copula density is evaluated at each bin's probability midpoint. This is a
small-bin approximation to the bivariate rectangle probability. The underlying
continuous joint density is normalized on[0,1]^2. Only the posterior mean with a
fixed5% account blend is reported, before/after the separately disclosed original
arithmetic-accuracy ordering constraint.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import betainc, betaincc, logsumexp, ndtri

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_density
from tests.analysis.edge_bounded_accuracy import beta_parameters, _stable_mass
from tests.analysis.universal_competitiveness import annotate_probabilities, project_order
from tests.analysis.universal_joint_accuracy import _game, _observed, _positive_covariance


ACCURACY_BIN = .0001
ACCOUNT_WEIGHT = .05
TAIL_FLOOR = 1e-300
CORRELATION_EPSILON = 1e-9


def copula_log_density(z_first, z_second, correlation):
    """Gaussian-copula density in normal quantiles, excluding marginal densities."""
    rho = np.clip(correlation, -1+CORRELATION_EPSILON, 1-CORRELATION_EPSILON)
    variance = 1-rho*rho
    return -.5*np.log(variance)-.5*((z_second-rho*z_first)**2/variance-z_second*z_second)


def _marginal(observed, mean, variance):
    alpha, beta = beta_parameters(mean*100, variance*10000)
    lower = max(0., observed-ACCURACY_BIN/2)
    upper = min(1., observed+ACCURACY_BIN/2)
    lower_cdf, upper_cdf = betainc(alpha, beta, lower), betainc(alpha, beta, upper)
    lower_survival, upper_survival = betaincc(alpha, beta, lower), betaincc(alpha, beta, upper)
    mass = _stable_mass(lower_cdf, upper_cdf, lower_survival, upper_survival)
    midpoint_cdf = .5*(lower_cdf+upper_cdf)
    midpoint_survival = .5*(lower_survival+upper_survival)
    normal = np.where(midpoint_cdf <= .5,
                      ndtri(np.maximum(midpoint_cdf, TAIL_FLOOR)),
                      -ndtri(np.maximum(midpoint_survival, TAIL_FLOOR)))
    return np.log(np.maximum(mass, TAIL_FLOOR)), normal


def likelihood(observed, mean, covariance):
    covariance = _positive_covariance(covariance)
    first, z_first = _marginal(observed[0], mean[:, 0], covariance[:, 0, 0])
    second, z_second = _marginal(observed[1], mean[:, 1], covariance[:, 1, 1])
    rho = covariance[:, 0, 1]/np.sqrt(covariance[:, 0, 0]*covariance[:, 1, 1])
    return first+second+copula_log_density(z_first, z_second, rho)


def predict(evidence, fit, ratings, calibration_cases):
    local_mean, constant_covariance = _game(evidence)
    local_covariance = np.broadcast_to(constant_covariance, (len(ARGS.grid), 2, 2))
    calibration = [_game(case['evidence']) for case in calibration_cases]
    means = np.stack([item[0] for item in calibration])
    population_mean = means.mean(axis=0)
    centered = means-population_mean
    between = np.einsum('gri,grj->rij', centered, centered)/max(1, len(calibration)-1)
    population_covariance = between+np.mean([item[1] for item in calibration], axis=0)
    pair = {}
    for side in ('White', 'Black'):
        observed = _observed(evidence[side])
        components = [likelihood(observed, mean, covariance) for mean, covariance in
                      ((local_mean, local_covariance), (population_mean, population_covariance))]
        logs = logsumexp(components, axis=0)-np.log(2.)
        density = np.exp(logs-logs.max())*prior_density(ARGS.grid)
        density /= trapezoid(density, ARGS.grid)
        point = float(trapezoid(density*ARGS.grid, ARGS.grid))
        pair[side] = (1-ACCOUNT_WEIGHT)*point+ACCOUNT_WEIGHT*float(ratings[side])
    return {'joint_beta_copula_mean_account': pair,
            'joint_beta_copula_mean_account_ordered': project_order(pair, fit)}


def main():
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
                sensitivity = {}
                for changed_side in ('White', 'Black'):
                    sensitivity[changed_side] = [value]
                    for shift in (-200., -100., 100., 200.):
                        changed = dict(prediction['joint_beta_copula_mean_account'])
                        changed[changed_side] += ACCOUNT_WEIGHT*shift
                        if method.endswith('_ordered'):
                            changed = project_order(changed, case['input']['fit'])
                        sensitivity[changed_side].append(changed[side])
                other = 'Black' if side == 'White' else 'White'
                rows.append({'game': case['game'], 'side': side, 'method': method,
                             'estimate': value, 'reference': case['references'][side],
                             'edge': bool(is_edge(case['input']['fit'], side)),
                             'own_elo_max_change': float(np.max(abs(np.array(sensitivity[side])-value))),
                             'own_elo_span': float(np.ptp(sensitivity[side])),
                             'opponent_elo_max_change': float(np.max(abs(np.array(sensitivity[other])-value)))})
    ranked = rank(rows)
    output = ROOT/'tests/analysis/output/universal-rating-methods/joint-copula.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'rankings': ranked, 'rows': rows, 'assumptions': __doc__}, indent=2), encoding='utf-8')
    print(json.dumps(ranked, indent=2))
    print(output)


if __name__ == '__main__':
    main()
