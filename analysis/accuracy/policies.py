"""Validate and attach native full-history policies from prepared predictions."""
from __future__ import annotations

from analysis.accuracy.evidence import RATINGS, normalized_policies


def prepare_policies(game, records, predictions):
    """Attach complete policies already inferred by the shared game pipeline."""
    moves = [move.uci() for move in game.mainline_moves()]
    if [row['move'] for row in records] != moves:
        raise ValueError("Maia policies do not match this game's mainline.")
    if len(predictions) != len(records):
        raise ValueError('Incomplete equal-opponent Maia position batch.')
    board = game.board()
    for row, entries in zip(records, predictions, strict=True):
        if row['side'] != ('White' if board.turn else 'Black'):
            raise ValueError('Maia record has the wrong player side.')
        legal = {move.uci() for move in board.legal_moves}
        if not legal:
            raise ValueError('A played position must have at least one legal move.')
        if len(legal) == 1:
            row['policies'] = {rating: {next(iter(legal)): 1.} for rating in RATINGS}
        else:
            if len(entries) != len(RATINGS):
                raise ValueError('Incomplete equal-opponent Maia rating policies.')
            row['policies'] = normalized_policies(
                {rating: entry['policy'] for rating, entry in zip(RATINGS, entries, strict=True)}, legal)
        board.push_uci(row['move'])
