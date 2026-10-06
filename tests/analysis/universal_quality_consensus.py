"""Equal-weight consensus across distinct universal quality measurement models.

The moment-based Gaussian/Beta model, full-policy convolution and shared latent
context model encode different approximations. Fixed equal weights reduce reliance
on any one approximation; there is no training of weights against reference Elo.
Point consensus is not a newly calibrated posterior or credible interval.
"""
from __future__ import annotations

from tests.analysis import universal_posterior_decision, universal_predictive_distribution, universal_curve_uncertainty


def predict(evidence, fit, ratings, calibration_cases):
    inputs = (evidence, fit, ratings, calibration_cases)
    moments = universal_posterior_decision.predict(*inputs)['mixture_mean_account']
    discrete = universal_predictive_distribution.predict(*inputs)['policy_predictive_mixture_median_account']
    hierarchical = universal_curve_uncertainty.predict(*inputs)['latent_curve_mean_account']
    models = {'consensus_moment_discrete': (moments, discrete),
              'consensus_moment_hierarchical': (moments, hierarchical),
              'consensus_three_quality_models': (moments, discrete, hierarchical)}
    return {name: {side: sum(model[side] for model in components)/len(components)
                   for side in ('White', 'Black')} for name, components in models.items()}
