"""Research-only Lichess level with an arithmetic within-game contrast.

Two predeclared variants combine either the plug-in or policy-expectation
Lichess pair center with the arithmetic estimator's pair difference. This is
a constrained point decision, not the posterior mean or minimum-MSE action of
a single measurement model. It deliberately preserves the arithmetic ordering.
Both inputs and the common translation are on the native Lichess Blitz scale;
any nonlinear display-scale conversion must happen after this operation.
"""
from __future__ import annotations

import math
from numbers import Real


SIDES = ('White', 'Black')
METHODS = {
    'lichess_plugin_center_arithmetic_contrast': 'lichess_plugin',
    'lichess_policy_center_arithmetic_contrast': 'lichess_policy_expectation',
}


def describe():
    return {
        name: ('Shared native rating level from ' +
               ('plug-in Lichess accuracy' if source == 'lichess_plugin' else
                'expected policy-draw Lichess accuracy') +
               '; retain the arithmetic estimator\'s within-game rating gap. '
               'A point-decision constraint, not a single-model posterior.')
        for name, source in METHODS.items()
    }


def _points(values):
    if not isinstance(values, dict) or set(values) != set(SIDES):
        raise ValueError('Native point estimates must use exactly White and Black keys.')
    result = {}
    for side in SIDES:
        value = values[side]
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not math.isfinite(value)):
            raise ValueError('Unbounded native affine predictions must be finite, or None.')
        result[side] = float(value) if value is not None else None
    return result


def preserve_arithmetic_contrast(arithmetic_points, lichess_points, rating_range=(0., 3200.)):
    """Replace a complete native pair's center and keep its signed difference.

    Inputs must be the unbounded affine predictions before native clipping.
    With arithmetic center c_A, alternative center c_L and original gap d_A,
    return clip(c_L+d_A/2), clip(c_L-d_A/2). For incomplete pairs return the
    clipped new-metric inputs and record that no fusion was performed.
    Clipping may reduce the gap; it cannot reverse its sign. Inputs are untouched.
    """
    if (len(rating_range) != 2 or any(isinstance(x, bool) or not isinstance(x, Real)
                                    or not math.isfinite(x) for x in rating_range)
            or rating_range[0] >= rating_range[1]):
        raise ValueError('The native rating support needs two increasing finite endpoints.')
    bounds = tuple(map(float, rating_range))
    arithmetic = _points(arithmetic_points)
    lichess = _points(lichess_points)
    complete = all(value is not None for pair in (arithmetic, lichess) for value in pair.values())
    if not complete:
        points = {side: min(bounds[1], max(bounds[0], value)) if value is not None else None
                  for side, value in lichess.items()}
        return {'players': points, 'diagnostics': {
            'applied': False, 'reason': 'Both complete native pairs are required; new-metric fallback.',
            'arithmetic_center': None, 'lichess_center': None, 'common_shift': None,
            'arithmetic_contrast': None, 'unclipped': lichess.copy(),
            'clipped': {side: points[side] != lichess[side] for side in SIDES},
            'rating_range': list(bounds),
        }}
    arithmetic_center = (arithmetic['White']+arithmetic['Black'])/2.
    lichess_center = (lichess['White']+lichess['Black'])/2.
    contrast = arithmetic['White']-arithmetic['Black']
    shift = lichess_center-arithmetic_center
    unclipped = {'White': lichess_center+contrast/2., 'Black': lichess_center-contrast/2.}
    points = {side: min(bounds[1], max(bounds[0], value)) for side, value in unclipped.items()}
    return {'players': points, 'diagnostics': {
        'applied': True, 'arithmetic_center': arithmetic_center, 'lichess_center': lichess_center,
        'common_shift': shift, 'arithmetic_contrast': contrast, 'unclipped': unclipped,
        'clipped': {side: points[side] != unclipped[side] for side in SIDES},
        'rating_range': list(bounds),
    }}
