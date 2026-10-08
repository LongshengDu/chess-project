"""Resolve ECO opening names from the local Lichess opening database."""
from __future__ import annotations

import csv
from functools import lru_cache
from io import StringIO
from pathlib import Path
import re

import chess.pgn


OPENINGS_DIR = Path(__file__).resolve().parents[2] / 'deps' / 'lichess-openings'


def _common_name(names):
    """Use an actual database name shared by every variation, never a guessed label."""
    for name in sorted(set(names), key=lambda value: (-len(value), value)):
        if all(other == name or other.startswith((name + ':', name + ',')) for other in names):
            return name
    return None


@lru_cache(maxsize=5)
def _volume(letter):
    path = OPENINGS_DIR / f'{letter.lower()}.tsv'
    if not path.is_file():
        raise FileNotFoundError('Lichess opening data is missing. Run '
                                'git submodule update --init deps/lichess-openings.')
    with path.open(encoding='utf-8', newline='') as source:
        return tuple(csv.DictReader(source, delimiter='\t'))


@lru_cache(maxsize=500)
def _positions(eco):
    positions = {}
    for entry in _volume(eco[0]):
        if entry['eco'] != eco:
            continue
        line = chess.pgn.read_game(StringIO(entry['pgn']))
        if line is None or line.errors:
            raise ValueError(f'Invalid Lichess opening line for {eco}.')
        positions.setdefault(line.end().board().epd(), set()).add(entry['name'])
    return positions


def opening_name(eco, game=None):
    """Match the latest named position for this ECO, or its unambiguous code name.

    Position matching recognizes transpositions. A code-only lookup returns a
    common named ancestor when available; ECO alone cannot identify a variation.
    Missing/unknown codes and games with no matching position return None.
    """
    if not isinstance(eco, str) or not re.fullmatch(r'[A-E]\d{2}', eco):
        return None
    if game is None:
        return _common_name([row['name'] for row in _volume(eco[0]) if row['eco'] == eco])
    positions = _positions(eco)
    board = game.board()
    name = _common_name(positions.get(board.epd(), ()))
    for move in game.mainline_moves():
        board.push(move)
        candidates = positions.get(board.epd())
        if candidates:
            name = _common_name(candidates)
    return name
