"""Complete position evidence shared by the saved analysis and Maia web panels."""


def stockfish_result(board, scan):
    """Convert White-oriented scan lines to the frontend's native engine contract."""
    lines = scan['lines']
    return {**scan.get('search', {}),
            'depth': scan.get('search', {}).get('depth', max((line['depth'] for line in lines), default=0)),
            'complete': True, 'coverage_complete': True,
            'is_checkmate': board.is_checkmate(),
            'best_move': scan['best_move'], 'engine_moves': scan['engine_moves'],
            'cp_vec': {line['uci']: line['cp'] if line['mate'] is None else
                       (10000 if line['mate'] >= 0 else -10000) for line in lines},
            'mate_vec': {'': 0} if board.is_checkmate() else {line['uci']: line['mate'] * (1 if board.turn else -1)
                         for line in lines if line['mate'] is not None},
            'root_move_depth_vec': {line['uci']: line['depth'] for line in lines}}
