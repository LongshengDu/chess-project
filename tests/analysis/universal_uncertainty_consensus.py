"""Equal pooling of shared and side-specific measurement uncertainty.

Both component rules measure the same shared-curve strength, but either pool
White/Black measurement variance or retain each side's conditional Maia variance.
Equal point pooling expresses uncertainty about that exchangeability assumption;
it is a constrained decision ensemble, not a newly calibrated posterior.
Component weights and the 10% account contribution are common to every player.
The account contribution and original color-order constraint are applied once,
after component pooling. References and game identities never enter inference.
"""
from tests.analysis.universal_account_sensitivity import unprojected_quality
from tests.analysis.universal_side_uncertainty import quality_estimates
from tests.analysis.universal_competitiveness import project_order


def predict(evidence, fit, ratings, calibration_cases):
    pooled = unprojected_quality(evidence, fit, calibration_cases)
    individual = quality_estimates(evidence, fit, calibration_cases, ('average',))['side_uncertainty_average']
    estimates = {side: .9*(pooled[side]+individual[side])/2+.1*ratings[side]
                 for side in ('White', 'Black')}
    return {'uncertainty_consensus': project_order(estimates, fit)}
