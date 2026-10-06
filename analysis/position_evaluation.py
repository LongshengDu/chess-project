"""Chess evaluation units, material stages, and rating metadata helpers."""
import chess

RATINGS = tuple(range(1000, 2601, 100))
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def stage(board):
    material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.piece_type != chess.PAWN)
    return 'endgame' if material <= 20 else 'opening' if board.fullmove_number <= 10 else 'middlegame'


def header_elo(headers, side):
    try:
        value = int(headers.get(side.title() + 'Elo', ''))
        return value if 100 <= value <= 4000 else None
    except (TypeError, ValueError):
        return None


def clip_elo(rating):
    return max(600, min(3000, round(rating)))


def eval_value(score):
    """White-perspective pawns, or explicit mate notation; never fake centipawns."""
    if score['mate'] is not None:
        return '#-0' if score['mate'] == 0 and score.get('winner') == 'black' else f"#{score['mate']}"
    return round(score['cp'] / 100, 2)


def eval_loss(best, played, side):
    if not isinstance(best, (int, float)) or not isinstance(played, (int, float)):
        return None
    return round((best-played) * (1 if side == 'white' else -1), 2)


def centipawns(value):
    """Convert the coach's White-relative pawns, retaining mate and missing scores."""
    return round(value*100) if isinstance(value, (int, float)) else value
