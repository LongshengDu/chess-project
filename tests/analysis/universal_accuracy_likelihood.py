"""Universal two-player likelihoods with a shared latent accuracy context.

Every player is estimated under the same rule. The game-level context is inferred
jointly from both observed arithmetic accuracies, instead of selecting independent
conditional-strength/global-quality models for the two players. Four fixed models
compare a binary target/population context, a complete game-context ensemble, a
random logit-accuracy offset, and a random offset/positive-slope transformation.

Random-effect covariances come only from leave-game-out Maia curve geometry. They
describe differences between position contexts, not externally measured Maia
prediction errors. Beta measurement probabilities and priors are reused unchanged.
The optional5% account blend is an explicit stable decision rule, not fitted data.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import expit, logit

from analysis.player_rating.bayesian_shared_curve import ARGS, GRID, prior_density
from tests.analysis.edge_bounded_accuracy import beta_accuracy_mass
from tests.analysis.edge_global_quality import _population


ACCOUNT_WEIGHT = .05
TARGET_CONTEXT_PROBABILITY = .5
BASE_METHODS = (
    'joint_accuracy_target_population_all',
    'joint_accuracy_context_ensemble_all',
    'joint_accuracy_random_offset_all',
    'joint_accuracy_random_affine_all',
)
METHODS = tuple(name for base in BASE_METHODS for name in (base, base[:-4]+'_account_5pct_all'))


def joint_posterior(accuracies, means, variances, context_weights, grid):
    """Marginalize a single shared context using both players' integrated evidence."""
    means = np.asarray(means, dtype=float)
    variances = np.broadcast_to(variances, means.shape)
    weights = np.asarray(context_weights, dtype=float)
    if (means.ndim != 2 or weights.shape != (len(means),) or np.any(weights < 0)
            or not np.isfinite(weights).all() or weights.sum() <= 0):
        raise ValueError('Finite nonnegative weights must identify the accuracy contexts.')
    weights = weights/weights.sum()
    masses = np.stack([beta_accuracy_mass(accuracy, means, variances) for accuracy in accuracies])
    unnormalized = masses*prior_density(grid)
    evidence = trapezoid(unnormalized, grid, axis=-1)
    valid = (weights > 0) & np.all(evidence > 0, axis=0)
    if not np.any(valid):
        raise ValueError('The shared-context model has no finite joint evidence.')
    logs = np.log(weights[valid])+np.log(evidence[:, valid]).sum(axis=0)
    posterior_context = np.exp(logs-logs.max())
    posterior_context /= posterior_context.sum()
    conditional = unnormalized[:, valid]/evidence[:, valid, None]
    density = np.sum(conditional*posterior_context[None, :, None], axis=1)
    cdf = cumulative_trapezoid(density, grid, axis=-1, initial=0.)
    estimates = [float(np.interp(.5, row, grid)) for row in cdf]
    return {'estimates': estimates, 'density': density,
            'posterior_context': posterior_context, 'retained_contexts': valid}


def _random_effects(calibration_curves, grid):
    """Fit context geometry on native600--2600 support, without played outcomes."""
    measured = np.array([np.interp(GRID, grid, curve) for curve in calibration_curves])
    bounded_logits = logit(np.clip(measured/100., 1e-9, 1-1e-9))
    basis = np.column_stack((np.ones(len(GRID)), (GRID-1600.)/1000.))
    coefficients = np.linalg.lstsq(basis, bounded_logits.T, rcond=None)[0].T
    geometry = np.column_stack((coefficients[:, 0], np.log(np.maximum(coefficients[:, 1], 1e-9))))
    covariance = np.cov(geometry, rowvar=False, ddof=1) if len(geometry) > 1 else np.zeros((2, 2))
    return np.asarray(covariance, dtype=float)


def context_models(fit, calibration_cases):
    """Build four context collections; no account/reference ratings are inspected."""
    grid = ARGS.grid
    curves, conditional_variances, pooled, between = _population(calibration_cases, grid)
    diagnostic = fit['diagnostics']['curve']
    own = np.asarray(diagnostic['shared_accuracy'], dtype=float)
    if not np.array_equal(np.asarray(diagnostic['fine_ratings']), grid):
        raise ValueError('The target fit must use the production rating grid.')
    own_variance = float(diagnostic['likelihood']['accuracy_variance'])
    covariance = _random_effects(curves, grid)
    own_logit = logit(np.clip(own/100., 1e-9, 1-1e-9))
    anchor = float(np.interp(1600., grid, own_logit))

    offset_nodes, offset_weights = hermgauss(7)
    offsets = np.sqrt(2*max(0., covariance[0, 0]))*offset_nodes
    offset_means = 100.*expit(own_logit[None, :]+offsets[:, None])

    nodes, weights = hermgauss(5)
    pairs = np.array(list(product(range(5), repeat=2)))
    standard = np.sqrt(2)*nodes[pairs]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    effects = standard @ (eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.)))).T
    affine_means = 100.*expit(anchor+effects[:, :1]+np.exp(effects[:, 1:2])*(own_logit[None, :]-anchor))

    return {
        BASE_METHODS[0]: (np.stack((own, pooled)),
                          np.stack((np.full_like(grid, own_variance), between+conditional_variances.mean())),
                          np.array([TARGET_CONTEXT_PROBABILITY, 1-TARGET_CONTEXT_PROBABILITY])),
        BASE_METHODS[1]: (np.vstack((own[None, :], curves)),
                          np.r_[own_variance, conditional_variances][:, None],
                          np.r_[TARGET_CONTEXT_PROBABILITY,
                                np.full(len(curves), (1-TARGET_CONTEXT_PROBABILITY)/len(curves))]),
        BASE_METHODS[2]: (offset_means, own_variance, offset_weights/np.sqrt(np.pi)),
        BASE_METHODS[3]: (affine_means, own_variance, np.prod(weights[pairs], axis=1)/np.pi),
    }


@lru_cache(maxsize=64)
def _cached_fit(target_json, calibration_json):
    """Reuse latent-context computations when only supplied account ratings change."""
    import json
    target = json.loads(target_json)
    calibration = [{'fit': json.loads(value)} for value in calibration_json]
    accuracies = [target['players'][side]['average_accuracy'] for side in ('White', 'Black')]
    if any(accuracy is None for accuracy in accuracies):
        return {name: tuple(target['players'][side]['estimate'] for side in ('White', 'Black'))
                for name in BASE_METHODS}
    return {name: tuple(joint_posterior(accuracies, means, variance, weights, ARGS.grid)['estimates'])
            for name, (means, variance, weights) in context_models(target, calibration).items()}


def _numeric_fit(fit):
    """Strip names, labels and irrelevant metadata before memoizing calculations."""
    diagnostic = fit['diagnostics']['curve']
    return {'diagnostics': {'curve': {key: diagnostic[key] for key in
                                      ('fine_ratings', 'shared_accuracy', 'monotone_expected_accuracy', 'likelihood')}}}


def predict(evidence, fit, ratings, calibration_cases):
    """Universal shared-context fits, with optional analytically stable account blend."""
    import json
    del evidence
    if not calibration_cases:
        raise ValueError('Independent calibration cases are required.')
    numeric_target = _numeric_fit(fit)
    numeric_target['players'] = {side: {key: fit['players'][side][key]
                                       for key in ('average_accuracy', 'estimate')}
                                 for side in ('White', 'Black')}
    target = json.dumps(numeric_target, sort_keys=True, separators=(',', ':'))
    calibration = tuple(json.dumps(_numeric_fit(case['fit']), sort_keys=True, separators=(',', ':'))
                        for case in calibration_cases)
    fitted = _cached_fit(target, calibration)
    outputs = {}
    for name, pair in fitted.items():
        base, account_variant = {}, {}
        for side, estimate in zip(('White', 'Black'), pair, strict=True):
            base[side] = estimate
            account = ratings.get(side)
            if estimate is None:
                account_variant[side] = None
            elif account is None or not np.isfinite(account) or not 0 <= account <= 3200:
                raise ValueError('Account ratings must be finite and lie in[0,3200].')
            else:
                account_variant[side] = (1-ACCOUNT_WEIGHT)*estimate+ACCOUNT_WEIGHT*account
        outputs[name] = base
        outputs[name[:-4]+'_account_5pct_all'] = account_variant
    return outputs
