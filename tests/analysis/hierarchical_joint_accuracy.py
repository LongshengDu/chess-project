"""Research-only affine rating decision from two correlated accuracy features.

The features are arithmetic accuracy on non-forced decisions and full Lichess
game accuracy. Their conditional moments must come from the same policy draws.
Context covariance T and mean conditional covariance V produce K=T(T+V)^+,
M=P+K(C-P), and residual covariance V+T-T(T+V)^+T. A common translated
account prior gives the best affine squared-error point using both features.

Pseudoinverses handle redundant features; they are numerical rank decisions,
not learned regularization coefficients. Matrix shrinkage need not preserve
component monotonicity or arithmetic player ordering. A separately declared
center/contrast action keeps the original arithmetic gap by design. Neither
decision returns a posterior density or a calibrated rating interval.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import isotonic_regression

from analysis.player_rating.bayesian_shared_curve import SharedCurve
from analysis.player_rating.shared_curve_affine import CURVE_ARGS, _account_anchor, translated_prior
from analysis.player_rating.parameters import RATINGS


SIDES = ('White', 'Black')
METHODS = ('hierarchical_joint_accuracy', 'hierarchical_joint_center_arithmetic_contrast')
PINV_RCOND = 1e-12
PSD_TOLERANCE = 1e-10


def describe():
    return {
        METHODS[0]: 'Joint arithmetic and expected Lichess accuracy, with conditional/context covariance and a common minimum-MSE affine decision.',
        METHODS[1]: 'Joint-feature native pair center plus original arithmetic affine gap; a separate ordering-preserving point decision.',
    }


def positive_semidefinite(value, name='covariance'):
    """Symmetrize rounding noise and remove only numerically negative eigenvalues."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite 2 by 2 covariance matrix.')
    tolerance = PSD_TOLERANCE*max(1., float(np.max(np.abs(matrix))))
    if np.max(np.abs(matrix-matrix.T)) > tolerance:
        raise ValueError(f'{name} must be symmetric within numerical tolerance.')
    matrix = (matrix+matrix.T)/2.
    values, vectors = np.linalg.eigh(matrix)
    if values[0] < -tolerance:
        raise ValueError(f'{name} is materially non-positive-semidefinite.')
    if values[0] < 0:
        matrix = (vectors*np.maximum(values, 0.)) @ vectors.T
    return (matrix+matrix.T)/2.


def matrix_shrinkage(local, population, between_covariance, measurement_covariance):
    """Normal-normal-motivated matrix analogue of scalar curve shrinkage."""
    local, population = np.asarray(local, dtype=float), np.asarray(population, dtype=float)
    if (local.ndim != 2 or local.shape[1] != 2 or not len(local)
            or local.shape != population.shape or not np.isfinite([local, population]).all()):
        raise ValueError('Matching finite rating by two-feature curves are required.')
    between = positive_semidefinite(between_covariance, 'Between-context covariance')
    measurement = positive_semidefinite(measurement_covariance, 'Measurement covariance')
    inverse = np.linalg.pinv(between+measurement, rcond=PINV_RCOND, hermitian=True)
    gain = between @ inverse
    model = population+(local-population) @ gain.T
    remaining = positive_semidefinite(between-between @ inverse @ between, 'Remaining covariance')
    noise = positive_semidefinite(measurement+remaining, 'Residual covariance')
    return {'model_curve': model, 'gain': gain, 'remaining_covariance': remaining,
            'residual_covariance': noise, 'between_covariance': between,
            'measurement_covariance': measurement}


def joint_affine_moments(model_curve, residual_covariance, prior):
    """Integrate the vector-feature affine normal equations under one prior."""
    grid = CURVE_ARGS.grid
    curve = np.asarray(model_curve, dtype=float)
    if curve.shape != (len(grid), 2) or not np.isfinite(curve).all():
        raise ValueError('A finite native grid by two-feature model curve is required.')
    noise = positive_semidefinite(residual_covariance, 'Residual covariance')
    density = np.asarray(prior['prior_density'], dtype=float)
    if density.shape != grid.shape or not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError('The common prior must have a finite nonnegative density on the native grid.')
    # Center before integration so a constant feature has exactly zero signal;
    # otherwise roundoff-sized variance can acquire a spurious pseudoinverse.
    reference = curve[0]
    deviations = curve-reference
    deviation_mean = trapezoid(deviations*density[:, None], grid, axis=0)
    mean = reference+deviation_mean
    centered = deviations-deviation_mean
    covariance = trapezoid((grid-prior['prior_mean'])[:, None]*centered*density[:, None], grid, axis=0)
    model_variance = trapezoid(centered[:, :, None]*centered[:, None, :]*density[:, None, None], grid, axis=0)
    total = positive_semidefinite(model_variance+noise, 'Total feature covariance')
    coefficient = covariance @ np.linalg.pinv(total, rcond=PINV_RCOND, hermitian=True)
    return {'accuracy_mean': mean, 'accuracy_covariance': total,
            'model_accuracy_covariance': positive_semidefinite(model_variance),
            'rating_accuracy_covariance': covariance, 'affine_coefficients': coefficient,
            'affine_intercept': float(prior['prior_mean']-coefficient @ mean),
            'covariance_rank': int(np.linalg.matrix_rank(total, tol=PINV_RCOND*np.linalg.norm(total, 2)))}


def _context(measured):
    moments, observed = [], dict.fromkeys(SIDES)
    for side in SIDES:
        entry = measured['sides'][side]
        joint = entry.get('joint') if entry else None
        if joint is None:
            continue
        mean = np.asarray(joint['mean'], dtype=float)
        covariance = np.asarray(joint['covariance'], dtype=float)
        value = np.asarray(joint['observed'], dtype=float)
        if (mean.shape != (len(RATINGS), 2) or covariance.shape != (len(RATINGS), 2, 2)
                or value.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(value).all()
                or np.any((mean < -1e-9) | (mean > 100+1e-9))
                or np.any((value < -1e-9) | (value > 100+1e-9))):
            raise ValueError('Joint moments require bounded two-feature observations and complete native-grid means/covariances.')
        covariance = np.stack([positive_semidefinite(matrix) for matrix in covariance])
        moments.append((np.clip(mean, 0., 100.), covariance))
        observed[side] = np.clip(value, 0., 100.)
    if not moments:
        return None, observed
    raw = np.mean([moment[0] for moment in moments], axis=0)
    knots = np.column_stack([isotonic_regression(raw[:, feature]).x for feature in range(2)])
    covariance = positive_semidefinite(np.mean([moment[1] for moment in moments], axis=(0, 1)))
    curve = np.column_stack([SharedCurve(knots[:, feature])(CURVE_ARGS.grid) for feature in range(2)])
    return {'knots': knots, 'curve': curve, 'measurement_covariance': covariance,
            'largest_isotonic_adjustment': float(np.max(np.abs(knots-raw)))}, observed


def fit_joint(measured, actual, records, fingerprint):
    """Fit one target with consistently measured, target-excluded contexts.

    ``records`` maps label-free context fingerprints to measurement results.
    Both scalar features are isotonic-projected and extended before the matrix
    combination. The population and T use those same projected native knots.
    Supplied account values enter only the common translated rating prior.
    """
    local, observed = _context(measured)
    empty = {'players': dict.fromkeys(SIDES), 'unbounded_players': dict.fromkeys(SIDES),
             'observed': observed, 'identifiable': False}
    if local is None or not np.any(np.ptp(local['curve'], axis=0) > 1e-10):
        return empty
    retained = [_context(value)[0] for key, value in records.items() if key != fingerprint]
    retained = [context for context in retained if context is not None]
    if not retained:
        raise ValueError('No joint population context remains after target exclusion.')
    knots = np.stack([context['knots'] for context in retained])
    centered = knots-knots.mean(axis=0)
    between = (np.einsum('jki,jkl->il', centered, centered)/(len(RATINGS)*(len(retained)-1))
               if len(retained) > 1 else np.zeros((2, 2)))
    population = np.mean([context['curve'] for context in retained], axis=0)
    hierarchy = matrix_shrinkage(local['curve'], population, between, local['measurement_covariance'])
    anchor, _ = _account_anchor(actual)
    prior = translated_prior(anchor)
    affine = joint_affine_moments(hierarchy['model_curve'], hierarchy['residual_covariance'], prior)
    unbounded = {side: (float(prior['prior_mean']+affine['affine_coefficients'] @
                             (value-affine['accuracy_mean'])) if value is not None else None)
                 for side, value in observed.items()}
    points = {side: float(np.clip(value, *CURVE_ARGS.rating_range)) if value is not None else None
              for side, value in unbounded.items()}
    return {'players': points, 'unbounded_players': unbounded, 'observed': observed, 'identifiable': True,
            'hierarchy': {key: value for key, value in hierarchy.items() if key != 'model_curve'} |
                         {'contexts_used': len(retained), 'contexts_excluded': int(fingerprint in records),
                          'largest_isotonic_adjustment': local['largest_isotonic_adjustment']},
            'affine': {**{key: value for key, value in prior.items() if not isinstance(value, np.ndarray)}, **affine},
            'curve': {'grid': CURVE_ARGS.grid, 'local': local['curve'], 'population': population,
                      'model': hierarchy['model_curve']}}
