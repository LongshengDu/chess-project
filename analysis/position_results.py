"""Complete position evidence shared by the saved analysis and Maia web panels."""

import chess


def stockfish_scan(board, result):
    """Expose a cached/native stream as White-oriented scored analysis lines."""
    lines = []
    for uci, cp in result['cp_vec'].items():
        move = chess.Move.from_uci(uci)
        child = board.copy(stack=True)
        san = child.san(move)
        child.push(move)
        mate = result['mate_vec'].get(uci)
        mate = mate * (1 if board.turn else -1) if mate is not None else None
        lines.append({'uci': uci, 'san': san, 'cp': cp if mate is None else None, 'mate': mate,
            'depth': result['root_move_depth_vec'][uci], 'pv_uci': [uci], 'pv_san': [san],
            'fen_after_pv': child.fen()})
    search = {key: value for key, value in result.items()
              if key not in ('cp_vec', 'mate_vec', 'root_move_depth_vec', 'is_checkmate')}
    return {'lines': lines, 'best_move': result['best_move'],
            'engine_moves': result['engine_moves'], 'search': search}


def stockfish_result(board, scan):
    """Convert White-oriented scan lines to the frontend's native engine contract."""
    lines = scan['lines']
    return {**scan.get('search', {}),
            'depth': scan.get('search', {}).get('depth', max((line['depth'] for line in lines), default=0)),
            'complete': scan.get('search', {}).get('complete', True),
            'coverage_complete': scan.get('search', {}).get('coverage_complete', True),
            'is_checkmate': board.is_checkmate(),
            'best_move': scan['best_move'], 'engine_moves': scan['engine_moves'],
            'cp_vec': {line['uci']: line['cp'] if line['mate'] is None else
                       (10000 if line['mate'] >= 0 else -10000) for line in lines},
            'mate_vec': {'': 0} if board.is_checkmate() else {line['uci']: line['mate'] * (1 if board.turn else -1)
                         for line in lines if line['mate'] is not None},
            'root_move_depth_vec': {line['uci']: line['depth'] for line in lines}}
