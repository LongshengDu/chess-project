"""Withdrawn population schema, retained only for historical synthetic tests.

The former asset was derived from benchmark games and violated their test-only
role. Loading a calibration asset is disabled, including explicit paths. The
current estimators use only the game being analyzed and declared inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from analysis.player_rating.bayesian_shared_curve import SharedCurve
from analysis.player_rating.parameters import RATINGS

DEFAULT_PATH = Path(__file__).with_name('data') / 'maia_accuracy_calibration.json'
WITHDRAWN_REASON = (
    'Benchmark-derived population calibration is withdrawn: test games must '
    'never supply training data or precomputed population assets. Use '
    'shared_curve_affine or bayesian_shared_curve, which use only the current game.'
)
SIDES = ('White', 'Black')
MEASUREMENTS = ('arithmetic', 'competitive')


def evidence_fingerprint(evidence):
    """Hash numeric context only; omit played indices, ratings and all metadata.

    Ordered positions, legal-candidate qualities, full policies and competitive
    weights identify each side. Sorting side encodings makes color exchange
    invariant. Competitive weights are rounded to twelve decimal places only
    for identity, avoiding complement-rounding differences for p versus 1-p.
    Numeric moments retain their original precision. Different engine evidence
    from the same PGN is not guaranteed to match this exact-context identity.
    """
    context = []
    for side in SIDES:
        positions = []
        for row in evidence[side]['observations']:
            q = np.asarray(row['qualities']['position'], dtype=float)
            p = np.asarray(row['maia_probabilities'], dtype=float)
            probability = float(row['position_win_probability'])
            if (q.ndim != 1 or not len(q) or p.shape != (len(RATINGS), len(q))
                    or not np.isfinite(q).all() or np.any((q < 0) | (q > 100))
                    or not np.isfinite(p).all() or np.any(p < 0)
                    or not np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=0)
                    or not np.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError('Calibration fingerprint requires complete finite context evidence.')
            positions.append([q.tolist(), p.tolist(), round(4*probability*(1-probability), 12)])
        context.append(json.dumps(positions, separators=(',', ':'), allow_nan=False))
    encoded = json.dumps(sorted(context), separators=(',', ':'), allow_nan=False).encode('utf-8')
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CalibrationCurve:
    """Measured monotone expected-accuracy knots and scalar mean variance."""

    knots: tuple[float, ...]
    variance: float


@dataclass(frozen=True)
class CalibrationRecord:
    """One anonymous paired-game context, with both accuracy measurements."""

    fingerprint: str
    arithmetic: CalibrationCurve
    competitive: CalibrationCurve


@dataclass(frozen=True)
class CalibrationCorpus:
    """Validated immutable asset; content hash identifies the complete source."""

    content_hash: str
    records: tuple[CalibrationRecord, ...]
    excluded_count: int = 0

    def for_evidence(self, evidence):
        """Return a copy excluding all exact matches of the target game context."""
        fingerprint = evidence_fingerprint(evidence)
        retained = tuple(row for row in self.records if row.fingerprint != fingerprint)
        if not retained:
            raise ValueError('Target exclusion leaves no calibration context.')
        return replace(self, records=retained,
                       excluded_count=self.excluded_count + len(self.records)-len(retained))

    def population(self, grid, measurement='arithmetic'):
        """Equal-context mean and within-plus-between population variance.

        Between-context variance uses the sample divisor J-1 when J > 1.
        Every game receives equal mass regardless of its number of moves.
        """
        if measurement not in MEASUREMENTS:
            raise ValueError('Unknown calibration accuracy measurement.')
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
            raise ValueError('A finite one-dimensional rating grid is required.')
        if not self.records:
            raise ValueError('At least one calibration context is required.')
        moments = tuple(getattr(row, measurement) for row in self.records)
        curves = np.stack([SharedCurve(row.knots)(grid) for row in moments])
        variance = np.var(curves, axis=0, ddof=int(len(curves) > 1))
        variance += np.mean([row.variance for row in moments])
        return curves.mean(axis=0), variance


def _curve(data):
    if not isinstance(data, dict) or set(data) != {'knots', 'variance'}:
        raise ValueError('Calibration curves require only knots and variance.')
    knots = np.asarray(data['knots'], dtype=float)
    # SharedCurve validates dimensions, bounds, finiteness and monotonicity.
    SharedCurve(knots)
    variance = data['variance']
    if (isinstance(variance, bool) or not isinstance(variance, (int, float))
            or not np.isfinite(variance) or not 0 <= variance <= 2500):
        raise ValueError('Calibration variance must be finite and in [0, 2500].')
    return CalibrationCurve(tuple(map(float, knots)), float(variance))


@lru_cache(maxsize=4)
def _read_content(content):
    data = json.loads(content)
    if (not isinstance(data, dict)
            or set(data) != {'schema_version', 'rating_grid', 'provenance', 'contexts'}
            or type(data['schema_version']) is not int or data['schema_version'] != 1
            or data['rating_grid'] != list(RATINGS)):
        raise ValueError('Incompatible Maia calibration asset schema or rating grid.')
    provenance = data['provenance']
    if (not isinstance(provenance, dict)
            or provenance.get('independent_human_validation') is not False
            or provenance.get('calibration_labels') != 'none'):
        raise ValueError('Calibration must explicitly declare reference-free exploratory provenance.')
    if not isinstance(data['contexts'], list) or not data['contexts']:
        raise ValueError('Calibration asset must contain at least one context.')
    if (type(provenance.get('context_count')) is not int
            or provenance['context_count'] != len(data['contexts'])):
        raise ValueError('Calibration provenance must report its exact context count.')
    records = []
    for row in data['contexts']:
        if (not isinstance(row, dict)
                or set(row) != {'fingerprint', *MEASUREMENTS}
                or not isinstance(row['fingerprint'], str)
                or not re.fullmatch('[0-9a-f]{64}', row['fingerprint'])):
            raise ValueError('Calibration contexts require anonymous SHA-256 fingerprints and moments only.')
        records.append(CalibrationRecord(row['fingerprint'], _curve(row['arithmetic']), _curve(row['competitive'])))
    if len({row.fingerprint for row in records}) != len(records):
        raise ValueError('Calibration contexts must have unique fingerprints.')
    return CalibrationCorpus(sha256(content).hexdigest(), tuple(records))


def load_calibration(path=None):
    """Reject the withdrawn asset before any file can be read."""
    raise ValueError(WITHDRAWN_REASON)
