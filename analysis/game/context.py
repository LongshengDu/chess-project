"""Bounded, factual game history for explaining how a critical position arose.

No engine searches or flag calculation. Geometric attacks are not proof that a
piece is lost, a square is safe, or a positional feature caused the outcome.
"""
from __future__ import annotations

import chess

from analysis.position_evaluation import PIECE_VALUES
from analysis.game.history import history_at, replay


def position_facts(board):
    facts = {}
    occupied = board.occupied
    open_files = [chess.FILE_NAMES[f] for f in range(8)
                  if not (board.pawns & chess.BB_FILES[f])]
    for color, side in ((chess.WHITE, 'white'), (chess.BLACK, 'black')):
        king = board.king(color)
        pawns = board.pieces(chess.PAWN, color)
        counts = [len(pawns & chess.SquareSet(chess.BB_FILES[f])) for f in range(8)]
        home_rank = 0 if color else 7
        home_minors = [chess.square_name(chess.square(f, home_rank))
                       for f, piece in ((1, chess.KNIGHT), (2, chess.BISHOP), (5, chess.BISHOP), (6, chess.KNIGHT))
                       if board.piece_at(chess.square(f, home_rank)) == chess.Piece(piece, color)]
        zone = chess.SquareSet(chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]) if king is not None else []
        facts[side] = {
            'king': chess.square_name(king) if king is not None else None,
            'material_points': sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == color),
            'pawn_squares': [chess.square_name(sq) for sq in pawns],
            'king_adjacent_pawns': [chess.square_name(sq) for sq in pawns if king is not None and
                                    chess.square_distance(sq, king) == 1],
            'castling_rights': [name for name, possible in [('kingside', board.has_kingside_castling_rights(color)),
                                                           ('queenside', board.has_queenside_castling_rights(color))] if possible],
            'minor_home_squares': home_minors,
            'doubled_pawn_files': [f'{chess.FILE_NAMES[f]}:{n}' for f, n in enumerate(counts) if n > 1],
            'isolated_pawns': [chess.square_name(sq) for sq in pawns if all(
                counts[f] == 0 for f in (chess.square_file(sq)-1, chess.square_file(sq)+1) if 0 <= f < 8)],
            'semi_open_files': [chess.FILE_NAMES[f] for f in range(8) if counts[f] == 0 and
                               board.pieces_mask(chess.PAWN, not color) & chess.BB_FILES[f]],
            'pinned_pieces': [chess.square_name(sq) for sq in chess.SquareSet(board.occupied_co[color])
                              if board.is_pinned(color, sq)],
            'attacked_undefended_pieces': [chess.square_name(sq) for sq in chess.SquareSet(board.occupied_co[color])
                if board.piece_type_at(sq) not in (chess.PAWN, chess.KING)
                and board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq)],
            'king_zone_attacks': [f'{chess.square_name(attacker)}>{chess.square_name(sq)}' for sq in zone
                                 for attacker in board.attackers(not color, sq)],
            'bishops_blocked_by_own_pawns': [f'{chess.square_name(bishop)}>{chess.square_name(sq)}'
                for bishop in board.pieces(chess.BISHOP, color)
                for sq in chess.SquareSet(chess.BB_KING_ATTACKS[bishop])
                if abs(chess.square_file(sq)-chess.square_file(bishop)) == 1
                and abs(chess.square_rank(sq)-chess.square_rank(bishop)) == 1
                and occupied & chess.BB_SQUARES[sq] and board.piece_at(sq) == chess.Piece(chess.PAWN, color)],
        }
    return {'open_files': open_files, **facts}


def fact_changes(before, after):
    changes = {}
    for key, value in after.items():
        old = before[key]
        if isinstance(value, dict):
            diff = fact_changes(old, value)
        elif isinstance(value, list):
            diff = {name: entries for name, entries in (
                ('added', [v for v in value if v not in old]),
                ('removed', [v for v in old if v not in value])) if entries}
        else:
            diff = {'from': old, 'to': value} if old != value else {}
        if diff:
            changes[key] = diff
    return changes


def leadup_context(analysis, plies, lookback_plies=8, *, side):
    """Deduplicate overlapping actual-game windows; targets are BEFORE their moves."""
    if type(lookback_plies) is not int or not 1 <= lookback_plies <= 16:
        raise ValueError('lookback_plies must be 1–16.')
    windows, positions, moves = [], {}, {}
    rows = analysis['moves']
    if side not in ('white', 'black'):
        raise ValueError('Coaching side must be white or black.')
    selected_side = side
    for ply in dict.fromkeys(plies):
        history_at(analysis, ply)  # Validate the reference, including N+1.
        start = max(1, ply-lookback_plies)
        older = [r['ply'] for r in rows[:start-1] if r['side'] == selected_side and r.get('flags')]
        windows.append({'target_ply': ply, 'from_ply': start, 'to_ply_exclusive': ply,
                        'earlier_flagged_plies': older[-3:]})
        board = replay(analysis['start_fen'], history_at(analysis, start))
        before = position_facts(board)
        positions[str(start)] = {'fen': board.fen(), 'facts': before}
        for row in rows[start-1:ply-1]:
            move = chess.Move.from_uci(row['played']['move'])
            if row['fen'] != board.fen():
                raise ValueError('Saved move FEN does not match the actual game history.')
            board.push(move)
            after = position_facts(board)
            moves[row['ply']] = {'ply': row['ply'], 'label': row['label'], 'side': row['side'],
                'before_eval': row['position_eval'], 'after_eval': row['played']['eval'],
                'loss': row['played']['loss'], 'flags': row.get('flags', []),
                'changes': fact_changes(before, after)}
            before = after
        positions[str(ply)] = {'fen': board.fen(), 'facts': before}
    return {'windows': windows, 'positions': positions, 'moves': [moves[k] for k in sorted(moves)]}
