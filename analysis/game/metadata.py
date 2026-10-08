"""Authoritative game context for analysis and coaching, separate from source PGN."""
from __future__ import annotations

from analysis.maia_context import SIDES, native_actual_ratings, rating_context
from analysis.elo_convert import SCALE_IDENTIFIERS
from analysis.game.opening import opening_name
from analysis.position_evaluation import header_elo

TEXT_HEADERS = {
    'result': 'Result', 'date': 'Date', 'eco': 'ECO',
}
TEXT_FIELDS = (*TEXT_HEADERS, 'opening')
RESULTS = ('1-0', '0-1', '1/2-1/2')


def _text(value):
    text = str(value).strip() if value is not None else ''
    return None if text in ('', '?', '-', '*', '????.??.??') else text


def game_metadata(headers, actual_elo=None, rating_scale=None, *, game=None):
    """Resolve effective ratings and useful PGN context without editing headers."""
    if actual_elo is not None and (type(actual_elo) is not int or not 100 <= actual_elo <= 4000):
        raise ValueError('Actual Elo must be an integer from 100 to 4000.')
    context = {side.lower(): {'name': _text(headers.get(side)),
                          'elo': actual_elo if actual_elo is not None else header_elo(headers, side)}
            for side in SIDES}
    context['rating_scale'] = SCALE_IDENTIFIERS[rating_context(headers, rating_scale)['scale']]
    context.update({field: _text(headers.get(tag)) for field, tag in TEXT_HEADERS.items()})
    context['opening'] = opening_name(context['eco'], game)
    if context['result'] not in RESULTS:
        context['result'] = None
    validate_game_metadata(context)
    return context


def validate_game_metadata(game):
    """Require complete effective context; never reconstruct it from raw headers."""
    if not isinstance(game, dict) or game.get('rating_scale') not in SCALE_IDENTIFIERS.values():
        raise ValueError('Game metadata requires a supported rating_scale.')
    if 'side' in game:
        raise ValueError('Game metadata must be independent of the coaching target.')
    if set(game) != {'white', 'black', 'rating_scale', *TEXT_FIELDS}:
        raise ValueError('Game metadata has missing or unsupported fields.')
    for side in ('white', 'black'):
        if not isinstance(game.get(side), dict):
            raise ValueError(f'Game metadata requires {side} name and Elo.')
        name = game[side]['name']
        if name is not None and (not isinstance(name, str) or _text(name) != name):
            raise ValueError('Player names must be nonempty text or null.')
    for field in TEXT_FIELDS:
        value = game[field]
        if value is not None and (not isinstance(value, str) or _text(value) != value):
            raise ValueError(f'Game {field} must be nonempty text or null.')
    if game['result'] is not None and game['result'] not in RESULTS:
        raise ValueError('Game result must be 1-0, 0-1, 1/2-1/2 or null.')
    native_actual_ratings({'game': game})
