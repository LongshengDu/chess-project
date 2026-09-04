from __future__ import annotations

import chess

PIECES = [
    (chess.QUEEN, "queen", 9),
    (chess.ROOK, "rook", 5),
    (chess.BISHOP, "bishop", 3),
    (chess.KNIGHT, "knight", 3),
    (chess.PAWN, "pawn", 1),
]


def move_sounds(board: chess.Board, captured: bool) -> list[str]:
    sounds = ["capture" if captured else "move"]
    if board.is_checkmate():
        sounds.append("checkmate")
    elif board.is_check():
        sounds.append("check")
    return sounds


def material(board: chess.Board) -> dict:
    result = {
        "white": {"pieces": [], "score": 0},
        "black": {"pieces": [], "score": 0},
    }
    score = 0
    for piece_type, role, value in PIECES:
        difference = len(board.pieces(piece_type, chess.WHITE)) - len(
            board.pieces(piece_type, chess.BLACK)
        )
        side = "white" if difference > 0 else "black"
        result[side]["pieces"].extend([role] * abs(difference))
        score += difference * value
    result["white"]["score"] = max(score, 0)
    result["black"]["score"] = max(-score, 0)
    return result
