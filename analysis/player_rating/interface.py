"""Abstract interface implemented by analysis/player_rating/<METHOD>.py rating estimators."""
from __future__ import annotations

from abc import ABC, abstractmethod


class PlayerRating(ABC):
    """A stateless rating calculation over the common numeric evidence contract.

    Each method module exports a concrete ``Rating(PlayerRating)`` class.
    Inference may use only the current game's evidence and declared inputs.
    Benchmark games and their derived moments must never become population,
    calibration, training, normalization or prior assets for another game.
    Implement ``fit``; optionally override ``name``, ``version`` and ``parameters``. Method
    identity comes from the module filename, so no registration or duplicate ID
    is needed. Engines, caches and result validation belong to the caller.
    """

    version = 1

    @property
    def id(self) -> str:
        return type(self).__module__.rsplit('.', 1)[-1]

    @property
    def name(self) -> str:
        return self.id.replace('_', ' ').title()

    @property
    def parameters(self) -> dict:
        """Finite JSON settings used by the method, included in saved-fit identity."""
        return {}

    @abstractmethod
    def fit(self, evidence: dict) -> dict:
        """Return player summaries, rating_range, central_interval and prior metadata.

        Point-only methods declare central_interval, interval and uncertainty as
        None instead of borrowing intervals from a different estimator.
        Actual ratings, rating grids and returned estimates all use native
        Lichess Blitz coordinates. The service normalizes declared input scales
        and converts returned decisions; estimators must not convert them again.

        Evidence and output contracts are documented in analysis/README.md under
        "Player-rating estimator interface".
        The caller validates evidence first and the returned result afterward.
        """
        raise NotImplementedError
