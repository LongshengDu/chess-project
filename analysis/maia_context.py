"""Declared account ratings converted only for Maia policy conditioning."""
from __future__ import annotations

import math

from analysis import elo_convert

NATIVE_SCALE = 'lb'
SIDES = ('White', 'Black')


def rating_context(headers=None, override=None):
    """Resolve the account scale, labeling missing PGN metadata explicitly."""
    headers = headers or {}
    if override is not None:
        if isinstance(override, dict):
            code = elo_convert.normalize_scale(override['scale'])
            return {**override, 'scale': code, 'name': elo_convert.NAMES[code]}
        return elo_convert.resolve_scale(headers, override=override)
    missing = lambda value: value is None or str(value).strip() in ('', '?', '-')
    if missing(headers.get('Site')) or missing(headers.get('TimeControl')):
        return {'scale': NATIVE_SCALE, 'name': elo_convert.NAMES[NATIVE_SCALE],
                'platform': 'lichess', 'time_class': 'blitz', 'estimated_seconds': None,
                'source': 'assumed_native', 'assumption': 'Site or TimeControl is missing; numeric Maia scale retained.',
                'site': headers.get('Site'), 'time_control': headers.get('TimeControl')}
    return elo_convert.resolve_scale(headers)


def analysis_scale(analysis):
    return elo_convert.normalize_scale(analysis['game']['rating_scale'])


def native_actual_ratings(analysis):
    """Read declared Elo only; analysis measurements never supply an account Elo."""
    ratings = {side: analysis['game'][side.lower()]['elo'] for side in SIDES}
    if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value))
           for value in ratings.values()):
        raise ValueError('Player Elo must be a finite number or null.')
    code = analysis_scale(analysis)
    result = {side: None if value is None else
              float(elo_convert.convert(value, code, NATIVE_SCALE, extrapolate=True))
              for side, value in ratings.items()}
    if any(value is not None and (not math.isfinite(value) or not 0 <= value <= 4000)
           for value in result.values()):
        raise ValueError('Actual ratings must convert to finite Lichess Blitz ratings within [0, 4000].')
    return result


def native_player_rating(analysis, side):
    side = side.title()
    if side not in SIDES:
        raise ValueError('Player side must be White or Black.')
    return native_actual_ratings(analysis)[side]
