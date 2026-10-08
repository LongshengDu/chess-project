"""Bounded comparisons of prepared Maia accuracy measurements for coaching."""
from __future__ import annotations

import math
from statistics import mean

import chess

from analysis.accuracy.evidence import RATINGS

SIDES = ('white', 'black')
STAGES = ('opening', 'middlegame', 'endgame')


def _metric(value, field, ply=None):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
        location = f' at ply {ply}' if ply is not None else ''
        raise ValueError(f'Missing or invalid {field}{location}; refresh the saved analysis first.')
    return value


def _rounded(value):
    return round(value, 2) if value is not None else None


class AccuracyComparison:
    """Read saved move measurements without engines, raw positions, or cache access.

    Comparing both colors at the same Maia anchor describes their experienced
    positions under equal-rating conditioning. Higher expected accuracy suggests
    more forgiving choices for that model, not objectively easier chess or cause.
    """

    def __init__(self, analysis):
        self.analysis = analysis

    @staticmethod
    def _ratings(ratings):
        ratings = [1600] if ratings is None else ratings
        if (not isinstance(ratings, (list, tuple)) or not 1 <= len(ratings) <= 6
                or any(type(r) is not int or r not in RATINGS for r in ratings)
                or len(set(ratings)) != len(ratings)):
            raise ValueError('Choose 1–6 distinct Maia anchors from 600 to 2600 in steps of 100.')
        return ratings

    def _select(self, ratings, stage, from_ply, to_ply, side=None):
        if stage is not None and stage not in STAGES:
            raise ValueError('Stage must be opening, middlegame, endgame, or null.')
        if side is not None and side not in SIDES:
            raise ValueError('Side must be white, black, or null.')
        if type(from_ply) is not int or from_ply < 1:
            raise ValueError('from_ply must be a positive game-relative ply.')
        if to_ply is not None and (type(to_ply) is not int or to_ply < from_ply):
            raise ValueError('to_ply must be an integer at least from_ply, or null.')
        moves = self.analysis.get('moves')
        if not isinstance(moves, list):
            raise ValueError('Saved analysis must contain a moves list.')
        try:
            board = chess.Board(self.analysis['start_fen'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Saved analysis must contain a valid start_fen.') from exc
        end = len(moves) if to_ply is None else to_ply
        rows, forced = [], dict.fromkeys(SIDES, 0)
        for ply, row in enumerate(moves, 1):
            turn = 'white' if board.turn else 'black'
            try:
                move = chess.Move.from_uci(row['played']['move'])
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f'Missing or invalid played move at ply {ply}.') from exc
            if (row.get('side') != turn or row.get('ply', ply) != ply
                    or move not in board.legal_moves):
                raise ValueError(f'Saved move side, ply, or legality does not match its position at ply {ply}.')
            if row.get('stage') not in STAGES:
                raise ValueError(f'Missing or invalid stage at ply {ply}; refresh the saved analysis first.')
            selected = (from_ply <= ply <= end and (side is None or side == turn)
                        and (stage is None or stage == row['stage']))
            if selected and board.legal_moves.count() == 1:
                forced[turn] += 1
            elif selected:
                accuracy = _metric(row.get('accuracy'), 'played accuracy', ply)
                maia = row.get('maia')
                values = {}
                for rating in ratings:
                    record = maia.get(str(rating)) if isinstance(maia, dict) else None
                    if not isinstance(record, dict):
                        raise ValueError(f'Missing prepared Maia {rating} data at ply {ply}; refresh the saved analysis first.')
                    values[str(rating)] = {field: _metric(record.get(field), field, ply)
                                          for field in ('expected_accuracy', 'absolute_deviation')}
                label = row.get('label') or f'{board.fullmove_number}{"." if board.turn else "..."} {board.san(move)}'
                rows.append({'ply': ply, 'move_number': board.fullmove_number, 'label': label, 'side': turn,
                             'stage': row['stage'], 'accuracy': accuracy, 'maia': values})
            board.push(move)
        return rows, forced, {'stage': stage, 'from_ply': from_ply, 'to_ply': end}

    def compare_positions(self, ratings=None, stage=None, from_ply=1, to_ply=None):
        """Compare actual encountered positions at common saved Maia anchors.

        Forced rows are counted but excluded from every arithmetic measurement.
        Shared measurements weight each nonempty side equally. Full-game Lichess
        accuracy is separate context and is never recalculated for a subset.
        """
        ratings = self._ratings(ratings)
        rows, forced, scope = self._select(ratings, stage, from_ply, to_ply)
        players, raw_means = {}, {}
        for side in SIDES:
            selected = [row for row in rows if row['side'] == side]
            average = mean(row['accuracy'] for row in selected) if selected else None
            raw_means[side] = {}
            for rating in ratings:
                key = str(rating)
                expected = mean(row['maia'][key]['expected_accuracy'] for row in selected) if selected else None
                deviation = mean(row['maia'][key]['absolute_deviation'] for row in selected) if selected else None
                raw_means[side][key] = {'expected_accuracy': expected, 'absolute_deviation': deviation,
                                       'actual_minus_expected': average-expected if selected else None}
            players[side] = {'positions': len(selected), 'forced_moves_excluded': forced[side],
                             'average_accuracy': _rounded(average),
                             'maia': {key: {field: _rounded(value) for field, value in metrics.items()}
                                      for key, metrics in raw_means[side].items()}}
        comparison = {}
        for rating in ratings:
            key = str(rating)
            white, black = (raw_means[side][key]['expected_accuracy'] for side in SIDES)
            delta = white-black if white is not None and black is not None else None
            available = [raw_means[side][key] for side in SIDES if players[side]['positions']]
            comparison[key] = {
                'shared_expected_accuracy': _rounded(mean(p['expected_accuracy'] for p in available)) if available else None,
                'shared_absolute_deviation': _rounded(mean(p['absolute_deviation'] for p in available)) if available else None,
                'white_minus_black_expected_accuracy': _rounded(delta),
                'higher_expected_accuracy_side': (None if delta is None else
                                                  'equal' if math.isclose(delta, 0., abs_tol=1e-9) else
                                                  'white' if delta > 0 else 'black'),
            }
        performance = self.analysis.get('performance', {}).get('players', {})
        lichess = {side: performance.get(side, {}).get('accuracy') for side in SIDES}
        for side, value in lichess.items():
            if value is not None:
                lichess[side] = _rounded(_metric(value, f'{side} full-game Lichess accuracy'))
        return {'rating_scale': 'lb', 'conditioning': 'equal_rating', 'scope': scope,
                'players': players, 'comparison': comparison,
                'full_game_lichess_accuracy': lichess}

    def by_move(self, maia_elo=1600, side=None, stage=None, from_ply=1, to_ply=None,
                order='chronological', limit=12):
        """Read a bounded set of position expectations; sort difficulty proxies only.

        Hardest/easiest mean lowest/highest expected accuracy, not judgments of
        the played move. move_number is the PGN fullmove number; both colors
        share it, but each accuracy belongs to that color's own pre-move board.
        Chronological pagination preserves separate game-relative plies.
        """
        self._ratings([maia_elo])
        if order not in ('chronological', 'hardest', 'easiest'):
            raise ValueError('Order must be chronological, hardest, or easiest.')
        if type(limit) is not int or not 1 <= limit <= 40:
            raise ValueError('Limit must be an integer from 1 to 40.')
        rows, forced, scope = self._select([maia_elo], stage, from_ply, to_ply, side)
        key = str(maia_elo)
        if order != 'chronological':
            direction = 1 if order == 'hardest' else -1
            rows.sort(key=lambda row: (direction*row['maia'][key]['expected_accuracy'], row['ply']))
        selected = rows[:limit]
        result = [{**{field: row[field] for field in ('ply', 'move_number', 'label', 'side', 'stage')},
                   'accuracy': _rounded(row['accuracy']),
                   **{field: _rounded(value) for field, value in row['maia'][key].items()},
                   'actual_minus_expected': _rounded(row['accuracy']-row['maia'][key]['expected_accuracy'])}
                  for row in selected]
        truncated = len(rows) > len(selected)
        return {'rating_scale': 'lb', 'conditioning': 'equal_rating', 'maia_elo': maia_elo,
                'scope': {'side': side, **scope}, 'order': order,
                'positions_available': len(rows), 'forced_moves_excluded': forced,
                'returned': len(selected), 'truncated': truncated,
                'next_from_ply': rows[limit]['ply'] if truncated and order == 'chronological' else None,
                'rows': result}
