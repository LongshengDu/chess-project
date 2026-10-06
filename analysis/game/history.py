"""Reconstruct legal positions from complete game and compact branch histories."""
import chess


def replay(start_fen, history):
    if not isinstance(history, list) or len(history) > 2000:
        raise ValueError('History must be a complete list of at most 2000 UCI moves from the game start.')
    board = chess.Board(start_fen)
    for index, uci in enumerate(history, 1):
        if board.is_game_over(claim_draw=False):
            raise ValueError(f'Move {index} follows a terminal position.')
        try:
            board.push_uci(uci)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Illegal UCI move at history ply {index}: {uci}') from exc
    return board


def history_at(analysis, ply, line=None):
    """Rebuild full Maia history from a compact game-ply/branch reference."""
    if type(ply) is not int or not 1 <= ply <= len(analysis['moves']) + 1:
        raise ValueError('Ply must be 1 through N+1, before the referenced game move.')
    line = [] if line is None else line
    if not isinstance(line, list) or len(line) > 40 or any(not isinstance(m, str) for m in line):
        raise ValueError('Line must contain at most 40 legal UCI moves from the referenced position.')
    history = [row['played']['move'] for row in analysis['moves'][:ply-1]] + line
    replay(analysis['start_fen'], history)
    return history
