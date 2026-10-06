"""Prespecified common-account priors for an arithmetic shared-curve likelihood.

These experimental rules do not use population calibration or reference labels.
The same prior and likelihood variance apply to both players, giving a monotone
posterior-mean map of accuracy without a final pair-order projection.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid

from tests.analysis.simple_curve_account import prepare


SIDES = ('White', 'Black')
ELO_LOGISTIC_SCALE = 400./math.log(10.)
ELO_LOGISTIC_SD = ELO_LOGISTIC_SCALE*math.pi/math.sqrt(3.)
METHODS = (
    'curve_anchor_logistic_sd',
    'curve_anchor_local_noise_sd',
    'curve_anchor_cauchy',
)


def describe():
    return {
        'family': 'Common-account regularization of the arithmetic shared curve',
        'declared_before_scoring': True,
        'likelihood': 'The original Gaussian arithmetic-accuracy likelihood; shared conditional Maia mean variance; sigma scale1.',
        'point': 'Posterior mean under the existing fourth-power tapered rating prior multiplied by the account prior.',
        'account_anchor': 'Arithmetic mean of supplied actual ratings, clipped to0--3200; no account prior if neither is supplied.',
        'curve_anchor_logistic_sd': 'Gaussian account prior with SD400*pi/(sqrt(3)*log(10)), the SD of a logistic performance-noise distribution on the conventional Elo scale.',
        'curve_anchor_local_noise_sd': 'Gaussian account prior with SD=sigma/|C_prime(anchor)|; zero local slope gives the uninformative account-prior limit.',
        'curve_anchor_cauchy': 'Cauchy account prior with scale400/log(10), a robust heavy-tailed account-location model on the conventional Elo logistic scale.',
        'fixed_logistic_sd': ELO_LOGISTIC_SD,
        'fixed_cauchy_scale': ELO_LOGISTIC_SCALE,
        'assumptions': 'The Elo-scale noise interpretation is a prior modeling assumption, not a calibrated distribution of single-game strength. The local-noise prior gives account and accuracy evidence comparable local precision.',
        'ordering': 'Shared monotone C, fixed Gaussian accuracy variance and a common prior imply monotone likelihood ratio in observed accuracy, hence a nondecreasing posterior mean.',
        'account_sensitivity': 'Not bounded to25Elo; account sensitivity is measured rather than forced.',
        'missing_evidence': 'No decisions or a wholly unidentifiable shared curve return missing estimates.',
    }


def point(accuracy, prepared, method):
    """Evaluate a fixed common rating map, without labels or side-specific priors."""
    if method not in METHODS:
        raise ValueError('Unknown common-account candidate.')
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError('Observed arithmetic accuracy must lie in0--100.')
    grid, prior = prepared['grid'], prepared['prior']
    variance = max(float(prepared['curve']['likelihood']['accuracy_variance']), 1e-12)
    curve = np.asarray(prepared['curve']['shared_accuracy'], dtype=float)
    active = prior > 0
    log_mass = -.5*(accuracy-curve)**2/variance
    log_mass[active] += np.log(prior[active])
    if prepared['has_account']:
        distance = grid-prepared['anchor']
        if method == 'curve_anchor_logistic_sd':
            log_mass -= .5*(distance/ELO_LOGISTIC_SD)**2
        elif method == 'curve_anchor_local_noise_sd':
            # Precision=slope^2/variance avoids division by a vanishing slope.
            log_mass -= .5*distance**2*prepared['slope']**2/variance
        else:
            log_mass -= np.log1p((distance/ELO_LOGISTIC_SCALE)**2)
    density = np.zeros_like(grid)
    density[active] = np.exp(log_mass[active]-log_mass[active].max())
    density /= trapezoid(density, grid)
    return float(trapezoid(grid*density, grid))


def predict(evidence, actual_ratings):
    """Return three method->White/Black dictionaries, leaving inputs unchanged."""
    prepared = prepare(evidence, actual_ratings)
    result = {method: dict.fromkeys(SIDES) for method in METHODS}
    if not prepared['curve']['identifiable']:
        return result
    for side, player in zip(SIDES, prepared['curve']['players'], strict=True):
        accuracy = player['average_accuracy']
        if accuracy is not None:
            for method in METHODS:
                result[method][side] = point(accuracy, prepared, method)
    return result
