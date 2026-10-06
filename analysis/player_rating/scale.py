"""Rating coordinates around a native Lichess Blitz estimator.

Changing a display scale must not change the statistical model. Actual ratings
are normalized before fitting; points, quantiles and plot coordinates are then
mapped forward. A displayed mean is the converted native decision, not a new
posterior mean under a nonlinear coordinate change. Numeric chess evidence and
all Maia probability anchors remain in Lichess Blitz.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from analysis import elo_convert
from analysis.player_rating.context import attach_context, saved_ratings

VERSION = 1
NATIVE_SCALE = 'lb'
SIDES = ('White', 'Black')


def scale_code(value):
    return elo_convert.normalize_scale(value['scale'] if isinstance(value, dict) else value)


def rating_context(headers=None, override=None):
    """Resolve a PGN scale; missing metadata has an explicitly labeled native fallback.

    Recognized but unsupported time classes are never relabeled Blitz or Rapid.
    Numeric-only callers can explicitly pass a scale without supplying a PGN.
    """
    headers = headers or {}
    if override is not None:
        if isinstance(override, dict):
            code = scale_code(override)
            return {**override, 'scale': code, 'name': elo_convert.NAMES[code]}
        return elo_convert.resolve_scale(headers, override=override)
    missing = lambda value: value is None or str(value).strip() in ('', '?', '-')
    if missing(headers.get('Site')) or missing(headers.get('TimeControl')):
        return {'scale': NATIVE_SCALE, 'name': elo_convert.NAMES[NATIVE_SCALE],
                'platform': 'lichess', 'time_class': 'blitz', 'estimated_seconds': None,
                'source': 'assumed_native', 'assumption': 'Site or TimeControl is missing; numeric Maia scale retained.',
                'site': headers.get('Site'), 'time_control': headers.get('TimeControl')}
    return elo_convert.resolve_scale(headers)


def from_native(value, rating_scale):
    """Map native scalar/array coordinates without premature rounding or clipping."""
    if value is None:
        return None
    code = scale_code(rating_scale)
    if np.isscalar(value):
        return float(elo_convert.convert(value, NATIVE_SCALE, code, extrapolate=True))
    array = np.asarray(value, dtype=float)
    return np.array([elo_convert.convert(float(item), NATIVE_SCALE, code, extrapolate=True)
                     for item in array.flat], dtype=float).reshape(array.shape)


def native_rating(value, rating_scale):
    if value is None:
        return None
    return float(elo_convert.convert(value, scale_code(rating_scale), NATIVE_SCALE, extrapolate=True))


def native_jacobian(native_grid, rating_scale):
    """d(display rating)/d(native rating), for probability-density pushforwards."""
    code = scale_code(rating_scale)
    if np.isscalar(native_grid):
        return elo_convert.derivative(native_grid, code, extrapolate=True)
    grid = np.asarray(native_grid, dtype=float)
    return np.array([elo_convert.derivative(float(item), code, extrapolate=True)
                     for item in grid.flat]).reshape(grid.shape)


def normalize_evidence(evidence, rating_scale):
    """Preserve complete cached move evidence, replacing only actual-rating inputs."""
    ratings = {side: native_rating(evidence[side].get('actual_rating'), rating_scale) for side in SIDES}
    return attach_context(evidence, ratings)


def context_signature(ratings, context):
    payload = {'actual_ratings': {side: None if ratings.get(side) is None else float(ratings[side]) for side in SIDES},
               'scale': rating_context(override=context), 'conversion_version': elo_convert.MODEL_VERSION,
               'coordinate_version': VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def display_fit(result, context, source_ratings):
    """Map the native fit's decisions and interval endpoints to the declared scale.

    Canonical diagnostics/parameters remain explicitly labeled, avoiding a
    misleading mixture of transformed Elo and native Maia anchors. Renderers
    map their axes and densities through the same conversion functions.
    """
    context = rating_context(override=context)
    code = context['scale']
    converted = deepcopy(result)
    support = list(result['rating_range'])
    mapped_support = from_native(support, code).tolist()
    out_of_model = lambda value: value is not None and not elo_convert.LB_MIN <= value <= elo_convert.LB_MAX
    metadata = {**context, 'native_scale': NATIVE_SCALE, 'native_name': elo_convert.NAMES[NATIVE_SCALE],
                'conversion_version': elo_convert.MODEL_VERSION, 'coordinate_version': VERSION,
                'conversion_native_range': [elo_convert.LB_MIN, elo_convert.LB_MAX],
                'mapped_rating_range': mapped_support,
                'coordinate_rule': 'C_display(F(r)) = C_native(r); displayed decision = F(native decision).',
                'curve_support_extrapolated': code != NATIVE_SCALE and any(out_of_model(x) for x in support),
                'actual_ratings': dict(source_ratings),
                'native_actual_ratings': {side: native_rating(source_ratings.get(side), code) for side in SIDES},
                'actual_conversion_extrapolated': {}, 'estimate_conversion_extrapolated': {}}
    for side, player in converted['players'].items():
        native = result['players'][side]
        point = native.get('unrounded_estimate', native['estimate'])
        interval = native.get('interval')
        player.update(canonical_estimate=point, canonical_interval=deepcopy(interval))
        metadata['actual_conversion_extrapolated'][side] = code != NATIVE_SCALE and out_of_model(metadata['native_actual_ratings'][side])
        metadata['estimate_conversion_extrapolated'][side] = code != NATIVE_SCALE and out_of_model(point)
        if code == NATIVE_SCALE:
            continue  # Preserve native rounding and method-specific presentation.
        displayed = from_native(point, code)
        player['unrounded_estimate'] = displayed
        player['estimate'] = None if displayed is None else math.floor(displayed+.5)
        if interval is not None:
            low, high = from_native(interval, code)
            player['interval'] = [math.floor(low), math.ceil(high)]
            player['uncertainty'] = (None if displayed is None else
                int(math.ceil(max(player['estimate']-math.floor(low), math.ceil(high)-player['estimate'])/10)*10))
        # at_rating_limit and interval_touches_limit express native support,
        # and monotone coordinate conversion preserves those statements.
    converted['rating_scale'] = metadata
    converted['canonical_rating_range'] = support
    converted['parameters_rating_scale'] = NATIVE_SCALE
    if 'diagnostics' in converted:
        converted['diagnostics']['rating_scale'] = NATIVE_SCALE
    if code != NATIVE_SCALE:
        converted['rating_range'] = [math.floor(mapped_support[0]), math.ceil(mapped_support[1])]
        prior = converted['prior']
        prior['native_prior'] = deepcopy(result['prior'])
        for key in ('minimum', 'maximum', 'account_anchor'):
            if prior.get(key) is not None:
                prior[key] = from_native(prior[key], code)
        for key in ('flat_range', 'truncated_to'):
            if prior.get(key) is not None:
                prior[key] = from_native(prior[key], code).tolist()
        # A nonlinear change of scale does not preserve a constant shift.
        # Its native value remains available in native_prior above.
        prior.pop('shift', None)
        prior['kind'] = 'pushforward_'+str(prior.get('kind', 'native_prior'))
        prior['rating_scale'] = code
        prior['density_rule'] = 'p_display(F(r)) = p_native(r) / F_prime(r)'
        converted['interval_scope'] = ('Native Lichess Blitz calculation: '+result['interval_scope']+
            f' Points and interval endpoints are converted to {metadata["name"]}; conversion-model uncertainty is not included.')
    return converted


def analysis_scale(analysis):
    """The current actual-rating scale, resolved afresh from PGN context."""
    return rating_context(analysis.get('headers'), analysis.get('rating_scale_override'))['scale']


def native_actual_ratings(analysis):
    ratings = saved_ratings(analysis)
    selected = analysis.get('selected_player', {})
    side = str(selected.get('side', '')).title()
    if side in SIDES and ratings[side] is None:
        ratings[side] = selected.get('actual_elo')
    code = analysis_scale(analysis)
    native = {side: native_rating(value, code) for side, value in ratings.items()}
    if any(value is not None and not 0 <= value <= 4000 for value in native.values()):
        raise ValueError('Actual ratings must convert to finite Lichess Blitz ratings within [0, 4000].')
    return native


def native_player_rating(analysis, side, *, fitted=False):
    """Native Maia input from a source-scale actual or saved played-level result."""
    side = side.title()
    if side not in SIDES:
        raise ValueError('Player side must be White or Black.')
    if fitted:
        player = analysis.get('played_elo', {}).get(side.lower(), {})
        if player.get('canonical_estimate') is not None:
            return player['canonical_estimate']
        point = player.get('unrounded_estimate', player.get('estimate'))
        if point is not None:
            # Historical fits predate scale handling and were native outputs.
            return native_rating(point, analysis.get('played_elo_scale') or NATIVE_SCALE)
    return native_actual_ratings(analysis)[side]
