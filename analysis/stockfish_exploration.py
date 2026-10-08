"""Shared raw Stockfish continuation searches and caller-specific presentation."""
from __future__ import annotations

import math

import chess
import chess.engine

from analysis.settings import CONFIG
from analysis.stockfish_search import SearchControl, SearchDeadline
from engine.uci import seconds_to_milliseconds


def validate_exploration_seconds(seconds, maximum):
    if (isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum) or maximum < .05):
        raise ValueError('ANALYSIS.STOCKFISH_EXPLORATION.MAX_SEARCH_SECONDS must be finite and at least 0.05.')
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not .05 <= seconds <= maximum):
        raise ValueError(f'Exploration seconds must be between 0.05 and {maximum:g}')


def exploration_board(board, move=None):
    """Return the board actually searched, preserving its complete history."""
    root = board.copy(stack=True)
    if move is not None:
        candidate = chess.Move.from_uci(move)
        if candidate not in root.legal_moves:
            raise ValueError('Exploration move must be legal')
        root.push(candidate)
    return root


def validate_search_lines(board, result):
    """Accept only exact scores with legal, complete stored UCI continuations."""
    if not isinstance(result, dict) or set(result) != {'lines'} or not isinstance(result['lines'], list):
        raise ValueError('Invalid cached Stockfish continuation measurement.')
    terminal = board.is_game_over(claim_draw=False)
    if terminal != (not result['lines']):
        raise ValueError('Stockfish continuation lines do not match the terminal state.')
    roots = set()
    for line in result['lines']:
        if not isinstance(line, dict) or set(line) != {'cp', 'mate', 'depth', 'pv_uci'}:
            raise ValueError('Invalid Stockfish continuation fields.')
        cp, mate = line['cp'], line['mate']
        if not ((type(cp) in (int, float) and math.isfinite(cp) and mate is None)
                or (cp is None and type(mate) is int)):
            raise ValueError('Stockfish continuation requires one finite exact score.')
        if type(line['depth']) is not int or line['depth'] < 0:
            raise ValueError('Stockfish continuation depth must be nonnegative.')
        pv = line['pv_uci']
        if not isinstance(pv, list) or not pv or any(not isinstance(uci, str) for uci in pv):
            raise ValueError('Stockfish continuation needs a legal UCI line.')
        if pv[0] in roots:
            raise ValueError('Stockfish returned duplicate root continuations.')
        roots.add(pv[0])
        walk = board.copy(stack=True)
        for uci in pv:
            move = chess.Move.from_uci(uci)
            if move not in walk.legal_moves:
                raise ValueError('Stockfish returned an illegal continuation.')
            walk.push(move)


def search_lines(engine, board, *, seconds, depth, multipv=1, root_moves=None, control=None):
    """Store every engine-returned PV ply, with no SAN/FEN/display duplication."""
    if board.is_game_over(claim_draw=False):
        return {'lines': []}
    control = control or SearchControl()
    infos = {}
    with SearchDeadline(engine, seconds, 'Stockfish exceeded its time budget and was stopped.'):
        with engine.analysis(board, chess.engine.Limit(time=seconds, depth=depth),
                multipv=multipv, root_moves=root_moves) as search:
            control.attach(search)
            try:
                for info in search:
                    rank = info.get('multipv', 1)
                    if (info.get('pv') and 'score' in info and 'depth' in info
                            and not info.get('upperbound') and not info.get('lowerbound')
                            and (rank not in infos or info['depth'] >= infos[rank]['depth'])):
                        infos[rank] = info.copy()
            finally:
                control.attach(None)
    if set(infos) != set(range(1, multipv + 1)):
        raise RuntimeError('Stockfish returned incomplete exact scored lines; increase the search time.')
    result = {'lines': []}
    for _, info in sorted(infos.items()):
        score = info['score'].white()
        result['lines'].append({'cp': score.score(), 'mate': score.mate(),
            'depth': info['depth'], 'pv_uci': [move.uci() for move in info['pv']]})
    validate_search_lines(board, result)
    return result


def analysis_result(board, measurement, *, movetime_ms, depth, pv_plies):
    """Format a coach response without changing its shared raw measurement."""
    validate_search_lines(board, measurement)
    terminal = board.is_game_over(claim_draw=False)
    lines = []
    for raw in measurement['lines']:
        walk, sans = board.copy(stack=True), []
        ucis = raw['pv_uci'][:pv_plies]
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            sans.append(walk.san(move))
            walk.push(move)
        lines.append({**raw, 'uci': ucis[0], 'san': sans[0], 'pv_uci': ucis,
                      'pv_san': sans, 'fen_after_pv': walk.fen()})
    result = {'fen': board.fen(), 'terminal': terminal, 'lines': lines,
              'score_perspective': 'white', 'movetime_ms': movetime_ms, 'depth_ceiling': depth}
    if terminal:
        outcome = board.outcome(claim_draw=False)
        result.update(result=board.result(claim_draw=False), evaluation={
            'cp': None if board.is_checkmate() else 0, 'mate': 0 if board.is_checkmate() else None,
            'winner': None if outcome.winner is None else 'white' if outcome.winner else 'black'})
    else:
        result['evaluation'] = {key: lines[0][key] for key in ('cp', 'mate')}
    return result


def continuation_result(board, measurement, *, move=None, seconds, depth):
    """Format the web continuation, including the optional move before the search."""
    root = exploration_board(board, move)
    result = analysis_result(root, measurement, movetime_ms=seconds_to_milliseconds(seconds),
                             depth=depth, pv_plies=16)
    pv = ([move] if move is not None else []) + (result['lines'][0]['pv_uci'] if result['lines'] else [])
    walk, sans = board.copy(stack=True), []
    for uci in pv:
        step = chess.Move.from_uci(uci)
        sans.append(walk.san(step))
        walk.push(step)
    return {'fen': board.fen(), 'move': move, 'pv_uci': pv, 'pv_san': sans,
            'white_cp': result['evaluation']['cp'], 'white_mate': result['evaluation']['mate'],
            'depth': result['lines'][0]['depth'] if result['lines'] else 0,
            'target_depth': depth, 'time_limit_seconds': seconds,
            'terminal': result['terminal'], 'bound': False}


def explore(engine, board, *, move=None, seconds=None, depth=None):
    """Direct continuation interface, sharing search and presentation with callers."""
    settings = CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION']
    seconds = settings['DEFAULT_SEARCH_SECONDS'] if seconds is None else seconds
    depth = settings['MAX_DEPTH'] if depth is None else depth
    validate_exploration_seconds(seconds, settings['MAX_SEARCH_SECONDS'])
    root = exploration_board(board, move)
    raw = search_lines(engine, root, seconds=seconds_to_milliseconds(seconds) / 1000, depth=depth)
    return continuation_result(board, raw, move=move, seconds=seconds, depth=depth)
