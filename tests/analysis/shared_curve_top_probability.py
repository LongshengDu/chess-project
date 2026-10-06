"""Compatibility defaults for the retired top-probability experiment.

All selection, moments, prior and posterior calculations use the production
estimator and its current prior. The experiment's original 68% probability
cutoff and 0.5 accuracy-sigma multiplier are preserved.
"""
from dataclasses import dataclass

from analysis.player_rating.bayesian_shared_curve import (
    Args as CurveArgs, Rating, prior_density, prior_weights, top_probability_policy,
)


@dataclass(frozen=True)
class Args(CurveArgs):
    top_probability: float = .68
    accuracy_sigma_scale: float = .5


class TopProbabilityRating(Rating):
    """Run the common estimator with the historical experiment defaults."""

    def __init__(self, args=None):
        super().__init__(args or Args())

    @property
    def id(self):
        return Rating.__module__.rsplit('.', 1)[-1]
