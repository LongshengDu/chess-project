"""Local Maia play sessions using the existing model and shared game storage."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import math
import random
import re
import threading
import time
import uuid

import chess
import chess.pgn
from werkzeug.exceptions import Conflict, NotFound

from analysis.game.study import PgnAnalysisApi
from backend.settings import CONFIG


def time_control(value):
    if value == 'unlimited':
        return None, 0
    if not isinstance(value, str) or not re.fullmatch(r'\d{1,2}\+\d{1,2}', value):
        raise ValueError('Time control must be unlimited or minutes+increment seconds.')
    minutes, increment = map(int, value.split('+'))
    if not 0 <= minutes <= 60 or not 0 <= increment <= 30 or minutes + increment == 0:
        raise ValueError('Choose 0–60 minutes and 0–30 increment seconds, with a positive time.')
    return minutes * 60000, increment * 1000


def termination(board):
    outcome = board.outcome(claim_draw=False)
    if outcome is None and (board.is_repetition(3) or board.is_fifty_moves()):
        return {'type': 'rules', 'winner': None, 'result': '1/2-1/2'}
    if outcome is None:
        return None
    return {'type': 'rules', 'winner': None if outcome.winner is None else 'white' if outcome.winner else 'black',
            'result': outcome.result()}


def sample_move(policy, board, *, temperature, sampled):
    legal = sorted(move.uci() for move in board.legal_moves)
    if not isinstance(policy, dict) or set(policy) != set(legal):
        raise ValueError('Maia returned an incomplete legal move policy.')
    values = [policy[move] for move in legal]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or not any(values):
        raise ValueError('Maia returned invalid move probabilities.')
    if not sampled or temperature == 0:
        return legal[max(range(len(legal)), key=lambda i: values[i])]
    logs = [math.log(value) if value > 0 else -math.inf for value in values]
    peak = max(logs)
    return random.choices(legal, weights=[math.exp((value - peak) / temperature) for value in logs], k=1)[0]


class PlatformPlay:
    def __init__(self, platform):
        self.platform = platform
        self.repository = platform.repository
        self._locks = {}
        self._locks_guard = threading.Lock()

    @contextmanager
    def _locked(self, game_id):
        if not isinstance(game_id, str) or not re.fullmatch(r'[0-9a-f]{32}', game_id):
            raise NotFound('Local play game not found')
        with self._locks_guard:
            lock = self._locks.setdefault(game_id, threading.RLock())
        with lock:
            yield

    def _load(self, game_id):
        if not isinstance(game_id, str) or not re.fullmatch(r'[0-9a-f]{32}', game_id):
            raise NotFound('Local play game not found')
        state = self.repository.get_play(game_id)
        if state is None:
            raise NotFound('Local play game not found')
        return state

    def get(self, game_id):
        return self._response(self._load(game_id))

    def save_for_analysis(self, game_id):
        with self._locked(game_id):
            self._save(self._load(game_id))
        return {'game_id': game_id}

    @staticmethod
    def _clocks(state):
        base, increment = time_control(state['time_control'])
        if base is None:
            return {'white_ms': None, 'black_ms': None}
        # A zero-base increment game needs its first increment before clocks
        # start; the opening two plies remain free, as in the play controller.
        initial = base or increment
        remaining = {'white_ms': initial, 'black_ms': initial}
        board = chess.Board(state['start_fen'])
        for ply, move in enumerate(state['moves']):
            key = 'white_ms' if board.turn else 'black_ms'
            if ply >= 2:
                elapsed = state['move_times'][ply] or 0
                remaining[key] = max(0, remaining[key] - elapsed + increment)
            board.push_uci(move)
        if state['termination'] and state['termination']['type'] == 'time':
            remaining['white_ms' if board.turn else 'black_ms'] = 0
        return remaining

    def _response(self, state, **extra):
        return {'game_id': state['game_id'], 'opponent_elo': state['maia_rating'],
                'player_elo': None, 'moves': state['moves'], 'move_times': state['move_times'],
                'start_fen': state['start_fen'], 'termination': state['termination'],
                'clocks': self._clocks(state), **extra}

    def _snapshot(self, state):
        game = chess.pgn.Game()
        game.setup(chess.Board(state['start_fen']))
        human = state['player_color'].title()
        computer = 'Black' if human == 'White' else 'White'
        game.headers.update(Event='Local Maia game', Site='lichess.org', Date=state['date'],
                            Result=state['termination']['result'] if state['termination'] else '*')
        game.headers[human] = 'You'
        game.headers[computer] = f"Maia3 ({state['maia_rating']})"
        game.headers.update(WhiteElo=str(state['maia_rating']), BlackElo=str(state['maia_rating']))
        game.headers['MaiaModel'] = state['model']
        base, increment = time_control(state['time_control'])
        game.headers['TimeControl'] = '300' if base is None else f'{base // 1000}+{increment // 1000}'
        if state['termination']:
            game.headers['Termination'] = {'resign': 'resignation', 'time': 'time forfeit', 'rules': 'normal'}[state['termination']['type']]
        node = game
        for move in state['moves']:
            node = node.add_variation(chess.Move.from_uci(move))
        pgn = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
        snapshot = PgnAnalysisApi(self.platform.maia, self.platform.stockfish).import_pgn(pgn)
        snapshot.update(move_times=state['move_times'], time_control=state['time_control'],
                        termination=state['termination'], clocks=self._clocks(state))
        return snapshot

    def _save(self, state, *, invalidate=False):
        self.repository.save_play(state, self._snapshot(state), invalidate=invalidate)

    def start(self, data):
        color = data.get('player_color')
        if color not in ('white', 'black'):
            raise ValueError('Player color must be white or black.')
        rating = data.get('maia_rating', CONFIG['MAIA']['PLAYER_RATING'])
        if type(rating) is not int or not 600 <= rating <= 2600:
            raise ValueError('Maia rating must be an integer from 600 to 2600.')
        sampled = data.get('sample_moves', True)
        if type(sampled) is not bool:
            raise ValueError('sample_moves must be a boolean.')
        control = data.get('time_control', 'unlimited')
        time_control(control)
        start_fen = data.get('start_fen')
        board = self.platform.board(chess.STARTING_FEN if start_fen is None else start_fen)
        temperature = CONFIG['MAIA']['TEMPERATURE']
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('MAIA.TEMPERATURE must be finite and nonnegative.')
        request_id = data.get('request_id')
        if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id)):
            raise ValueError('request_id must be a short unique identifier.')
        config = {'player_color': color, 'maia_rating': rating, 'sample_moves': sampled,
                  'time_control': control, 'start_fen': board.fen(en_passant='fen'),
                  'temperature': temperature, 'model': CONFIG['MAIA']['MODEL']}
        state = {**config, 'game_id': uuid.uuid4().hex, 'date': datetime.now(timezone.utc).strftime('%Y.%m.%d'),
                 'moves': [], 'move_times': [], 'replies': {}, 'pending_reply_ply': None,
                 'termination': termination(board)}
        saved = self.repository.create_play(state, self._snapshot(state), request_id)
        if any(saved[key] != value for key, value in config.items()):
            raise Conflict('Start request was already used with different settings')
        return self._response(saved)

    def _history(self, state, data):
        supplied = data.get('start_fen')
        if supplied is not None and self.platform.board(supplied).fen(en_passant='fen') != state['start_fen']:
            raise ValueError('Starting FEN does not match this game.')
        moves = data.get('moves')
        if not isinstance(moves, list) or len(moves) > 2000 or any(not isinstance(move, str) for move in moves):
            raise ValueError('Supply the complete legal UCI move history.')
        current = state['moves']
        common = min(len(current), len(moves))
        if moves[:common] != current[:common]:
            raise Conflict('Move history conflicts with the saved game')
        board = chess.Board(state['start_fen'])
        for ply, move in enumerate(moves):
            if termination(board):
                raise ValueError('A move follows a finished position.')
            if ply >= len(current):
                if state['termination']:
                    raise Conflict('The game is already finished')
                if ('white' if board.turn else 'black') != state['player_color']:
                    raise ValueError('Maia moves must be requested from the local engine.')
            try:
                board.push_uci(move)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Illegal UCI move at ply {ply + 1}.') from exc
        if len(moves) > len(current):
            state['moves'] = list(moves)
            state['move_times'].extend([None] * (len(moves) - len(current)))
            state['termination'] = termination(board)
        pending = state.get('pending_reply_ply')
        acknowledged = pending is not None and len(moves) > pending
        if acknowledged:
            state['pending_reply_ply'] = None
        return board, len(moves) < len(current), len(moves) > len(current), acknowledged

    @staticmethod
    def _cancel_pending_reply(state, ply):
        # A selected reply is durable for retries, but is not played until a
        # subsequent client history includes it. Only that last undelivered
        # reply may be removed when the mover's clock expires in transit.
        if state.get('pending_reply_ply') != ply or len(state['moves']) != ply + 1:
            return False
        if state['replies'].get(str(ply)) != state['moves'][-1]:
            return False
        state['moves'].pop()
        state['move_times'].pop()
        state['replies'].pop(str(ply))
        state['pending_reply_ply'] = None
        state['termination'] = None
        return True

    @staticmethod
    def _forfeit(state, board, status, expected=None):
        loser = state['player_color'] if status == 'resign' else 'white' if board.turn else 'black'
        if status == 'time' and state['time_control'] == 'unlimited':
            raise ValueError('Unlimited games cannot end on time.')
        winner = 'black' if loser == 'white' else 'white'
        if status == 'time' and board.has_insufficient_material(winner == 'white'):
            winner = None
        if expected not in (None, 'none', winner):
            raise ValueError('Winner conflicts with the game termination.')
        state['termination'] = {'type': status, 'winner': winner,
            'result': '1-0' if winner == 'white' else '0-1' if winner == 'black' else '1/2-1/2'}

    def move(self, data):
        started = time.monotonic()
        for key in ('initial_clock', 'current_clock'):
            value = data.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 86400):
                raise ValueError('Clocks must be finite, nonnegative seconds.')
        game_id = data.get('game_id')
        with self._locked(game_id):
            state = self._load(game_id)
            board, stale, changed, acknowledged = self._history(state, data)
            ply = len(data['moves'])
            cached = state['replies'].get(str(ply))
            remaining = data.get('current_clock') if state['time_control'] != 'unlimited' and ply >= 2 else None
            expired = lambda: remaining is not None and time.monotonic() - started >= remaining
            if cached is not None:
                if expired() and self._cancel_pending_reply(state, ply):
                    self._forfeit(state, board, 'time')
                    self._save(state, invalidate=True)
                    return self._response(state, top_move=None, move_delay=0)
                return self._response(state, top_move=cached, move_delay=0, cached=True)
            if stale:
                raise Conflict('This position has already advanced')
            if changed or acknowledged:
                self._save(state, invalidate=changed)
            if state['termination']:
                return self._response(state, top_move=None, move_delay=0)
            if ('white' if board.turn else 'black') == state['player_color']:
                raise ValueError('It is the human player’s turn.')
            if expired():
                self._forfeit(state, board, 'time')
                self._save(state)
                return self._response(state, top_move=None, move_delay=0)
            predictions = self.platform.maia_predictions(
                [board.copy(stack=True)], [state['maia_rating']], [state['maia_rating']])
            if expired():
                self._forfeit(state, board, 'time')
                self._save(state)
                return self._response(state, top_move=None, move_delay=0)
            if not isinstance(predictions, list) or len(predictions) != 1 or not isinstance(predictions[0], dict):
                raise ValueError('Maia returned an invalid prediction batch.')
            selected = sample_move(predictions[0].get('policy'), board,
                                   temperature=state['temperature'], sampled=state['sample_moves'])
            state['replies'][str(ply)] = selected
            state['moves'].append(selected)
            state['move_times'].append(None)
            state['pending_reply_ply'] = ply
            board.push_uci(selected)
            state['termination'] = termination(board)
            self._save(state, invalidate=True)
            return self._response(state, top_move=selected, move_delay=0, cached=False)

    def submit(self, game_id, data):
        with self._locked(game_id):
            state = self._load(game_id)
            board, stale, changed, _ = self._history(state, data)
            times = data.get('move_times')
            if not isinstance(times, list) or len(times) != len(data['moves']) or any(
                    value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 86400000) for value in times):
                raise ValueError('Move times must be finite, nonnegative milliseconds or null for unknown times, one per ply.')
            status = data.get('game_over_state', 'not_over')
            if status not in ('not_over', 'rules', 'resign', 'time'):
                raise ValueError('Invalid game-over state.')
            if status == 'time' and stale and state['time_control'] != 'unlimited' and len(times) >= 2:
                if self._cancel_pending_reply(state, len(times)):
                    stale, changed = False, True
            for ply, value in enumerate(times):
                if state['move_times'][ply] is None:
                    state['move_times'][ply] = value
            if not stale and state['termination'] is None:
                if status == 'rules':
                    raise ValueError('The position is not finished under chess rules.')
                if status in ('resign', 'time'):
                    self._forfeit(state, board, status, data.get('winner'))
            self._save(state, invalidate=changed)
            return self._response(state, ok=True, stale=stale)

    def stats(self):
        states = self.repository.play_states()
        finished = [state for state in states if state['termination']]
        return {'play_games_played': len(finished),
                'play_won': sum(state['termination']['winner'] == state['player_color'] for state in finished),
                'play_drawn': sum(state['termination']['winner'] is None for state in finished),
                'play_elo': None, 'hand_games_played': 0, 'hand_won': 0, 'hand_drawn': 0, 'hand_elo': None,
                'brain_games_played': 0, 'brain_won': 0, 'brain_drawn': 0, 'brain_elo': None}
