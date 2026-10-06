"""Leave-game-out empirical Bayes from unlabeled observed chess accuracies.

This experiment learns a population prior from OTHER games' observed accuracies,
not just their Maia policies. It is unsupervised training on this corpus and must
not be presented as independent external validation. Commercial/account ratings
never enter prior learning. The target game is excluded from all training inputs.

Single-Gaussian and ordered two-Gaussian population priors are fitted by penalized
EM. Each component is normalized after multiplication by the unchanged rating
taper. Two pseudoobservations anchor the original prior; the mixture assigns one
to each half of that prior, which also supplies one pseudo-count per component.
Both forecasts use the same proper Gaussian/Beta accuracy likelihood and posterior
mean, optionally followed by the declared5% account-rating blend.
"""
from __future__ import annotations

from functools import lru_cache
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from analysis.player_rating.bayesian_shared_curve import ARGS, prior_weights
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.edge_global_quality import _population
from tests.analysis.edge_predictive_mixture import gaussian_accuracy_mass
from tests.analysis.universal_accuracy_likelihood import _numeric_fit


PSEUDO_OBSERVATIONS = 2.
EM_MAX_ITERATIONS = 200
EM_TOLERANCE = 1e-7
ACCOUNT_WEIGHT = .05
BASE_METHODS = ('empirical_gaussian_mean_all', 'empirical_two_gaussian_mean_all')
METHODS = tuple(name for base in BASE_METHODS for name in (base, base[:-4]+'_account_5pct_all'))


def measurement_likelihood(fit, calibration_cases):
    """Two observed-accuracy likelihoods; population contexts exclude this game."""
    grid = ARGS.grid
    _, variances, pooled, between = _population(calibration_cases, grid)
    diagnostic = fit['diagnostics']['curve']
    if not np.array_equal(np.asarray(diagnostic['fine_ratings']), grid):
        raise ValueError('The fit must use the current production rating grid.')
    accuracy = np.array([fit['players'][side]['average_accuracy'] for side in ('White', 'Black')], dtype=float)
    if not np.isfinite(accuracy).all():
        raise ValueError('Observed accuracies are required by empirical population learning.')
    local = gaussian_accuracy_mass(accuracy[:, None], np.asarray(diagnostic['shared_accuracy']),
                                   diagnostic['likelihood']['accuracy_variance'])
    population = beta_accuracy_mass(accuracy[:, None], pooled, between+variances.mean())
    return .5*(local+population)


def _component_logs(parameters, x, log_taper):
    mean, log_sigma = parameters
    logs = log_taper-.5*((x-mean)/np.exp(log_sigma))**2
    return logs-logsumexp(logs)


def _m_step(counts, initial, x, log_taper):
    """Optimize the tapered component, including its parameter-dependent normalizer."""
    total = float(counts.sum())
    first, second = float(counts@x/total), float(counts@(x*x)/total)

    def objective(parameters):
        mean, log_sigma = parameters
        variance = np.exp(2*log_sigma)
        raw = log_taper-.5*(x-mean)**2/variance
        log_normalizer = logsumexp(raw)
        probability = np.exp(raw-log_normalizer)
        expected = float(probability@x)
        expected_square = float(probability@((x-mean)**2))
        target_square = second-2*mean*first+mean*mean
        value = total*(.5*target_square/variance+log_normalizer)
        gradient = np.array([total*(expected-first)/variance,
                             total*(expected_square-target_square)/variance])
        return value, gradient

    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      bounds=((0., 3.2), (np.log(.005), np.log(10.))),
                      options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError('The empirical-prior component update failed numerically.')
    return result.x


def fit_population(likelihoods, components):
    """Penalized EM for one/two tapered Gaussian components, without rating labels."""
    if components not in (1, 2):
        raise ValueError('Only the two predeclared population shapes are supported.')
    likelihoods = np.asarray(likelihoods, dtype=float)
    grid = ARGS.grid
    if (likelihoods.ndim != 2 or likelihoods.shape[1] != len(grid) or not len(likelihoods)
            or not np.isfinite(likelihoods).all() or np.any(likelihoods < 0)):
        raise ValueError('Finite nonnegative observation likelihoods are required.')
    taper = prior_weights(grid)
    supported = taper > 0
    x = grid[supported]/1000.
    log_taper = np.log(taper[supported])
    base = taper[supported]/taper[supported].sum()
    center = float(base@x)
    sigma = float(np.sqrt(base@((x-center)**2)))
    if components == 1:
        anchors = PSEUDO_OBSERVATIONS*base[None, :]
        parameters = np.array([[center, np.log(sigma)]])
    else:
        split = x <= center
        anchors = np.stack((np.where(split, base, 0.), np.where(~split, base, 0.)))
        anchors /= anchors.sum(axis=1, keepdims=True)
        initial_means = np.interp([.25, .75], np.cumsum(base), x)
        parameters = np.column_stack((initial_means, np.full(2, np.log(sigma))))
    mixture_weights = np.full(components, 1/components)
    log_likelihood = np.log(np.maximum(likelihoods[:, supported], 1e-300))
    previous = -np.inf
    trace = []
    converged = False
    for iteration in range(1, EM_MAX_ITERATIONS+1):
        component_logs = np.array([_component_logs(parameter, x, log_taper) for parameter in parameters])
        joint = log_likelihood[:, None, :]+np.log(mixture_weights)[None, :, None]+component_logs[None, :, :]
        evidence = logsumexp(joint, axis=(1, 2))
        counts = np.exp(joint-evidence[:, None, None]).sum(axis=0)+anchors
        parameters = np.array([_m_step(count, initial, x, log_taper)
                               for count, initial in zip(counts, parameters, strict=True)])
        mixture_weights = counts.sum(axis=1)/counts.sum()
        order = np.argsort(parameters[:, 0])
        parameters, mixture_weights = parameters[order], mixture_weights[order]
        component_logs = np.array([_component_logs(parameter, x, log_taper) for parameter in parameters])
        log_prior = logsumexp(np.log(mixture_weights)[:, None]+component_logs, axis=0)
        objective = float(logsumexp(log_likelihood+log_prior, axis=1).sum()
                          +np.sum(anchors*(np.log(mixture_weights)[:, None]+component_logs)))
        trace.append(objective)
        if np.isfinite(previous) and abs(objective-previous) <= EM_TOLERANCE*(1+abs(previous)):
            converged = True
            break
        previous = objective
    mass = np.zeros_like(grid)
    mass[supported] = np.exp(log_prior)
    center = float(mass@grid)
    return {'prior_mass': mass, 'components': components, 'component_means': (parameters[:, 0]*1000).tolist(),
            'component_sigmas': (np.exp(parameters[:, 1])*1000).tolist(), 'component_weights': mixture_weights.tolist(),
            'prior_mean': center, 'prior_sd': float(np.sqrt(mass@((grid-center)**2))),
            'iterations': iteration, 'converged': converged, 'objective_trace': trace,
            'training_observations': len(likelihoods), 'pseudo_observations': PSEUDO_OBSERVATIONS}


def _training_fit(fit):
    result = _numeric_fit(fit)
    result['players'] = {side: {'average_accuracy': fit['players'][side]['average_accuracy']}
                          for side in ('White', 'Black')}
    return result


@lru_cache(maxsize=64)
def _cached_learning(calibration_json):
    cases = [{'fit': json.loads(value)} for value in calibration_json]
    if len(cases) < 2:
        raise ValueError('At least two independent calibration games are required.')
    likelihoods = np.vstack([measurement_likelihood(case['fit'], [other for other in cases if other is not case])
                             for case in cases])
    return {components: fit_population(likelihoods, components) for components in (1, 2)}


def learn_population(calibration_cases):
    """Only other-game curves, uncertainties and observed accuracies enter training."""
    return _cached_learning(tuple(json.dumps(_training_fit(case['fit']), sort_keys=True) for case in calibration_cases))


def predict(evidence, fit, ratings, calibration_cases):
    """Forecast both target players using the same learned population prior."""
    del evidence
    learned = learn_population(calibration_cases)
    likelihood = measurement_likelihood(fit, calibration_cases)
    output = {}
    for name, components in zip(BASE_METHODS, (1, 2), strict=True):
        posterior_mass = likelihood*learned[components]['prior_mass']
        posterior_mass /= posterior_mass.sum(axis=1, keepdims=True)
        estimates = posterior_mass@ARGS.grid
        output[name] = {}
        output[name[:-4]+'_account_5pct_all'] = {}
        for side, estimate in zip(('White', 'Black'), estimates, strict=True):
            account = ratings.get(side)
            if account is None or not np.isfinite(account) or not 0 <= account <= 3200:
                raise ValueError('Account ratings must be finite and lie in[0,3200].')
            output[name][side] = float(estimate)
            output[name[:-4]+'_account_5pct_all'][side] = float((1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*account)
    return output
