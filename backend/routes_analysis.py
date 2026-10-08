"""HTTP controller for imported studies and native analysis requests."""
from __future__ import annotations

import json
import math
import uuid

import chess
from flask import Blueprint, Response, abort, current_app, request, stream_with_context

from analysis.stockfish_exploration import (continuation_result, exploration_board,
                                           search_lines, validate_exploration_seconds)
from analysis.game.study import PgnAnalysisApi
from analysis.stockfish_search import BOUNDED_POLICY_VERSION, SearchControl
from analysis.cache.requests import stockfish_exploration_request
from analysis.cache.policy import promote_request
from backend.settings import CONFIG
from engine.stockfish import StockfishScorer
from engine.uci import seconds_to_milliseconds


class AnalysisRoutes:
    def __init__(self, platform):
        self.platform = platform
        self.repository = platform.repository

    def register(self, server):
        routes = Blueprint('analysis', __name__, url_prefix='/api/platform')
        for path, methods, handler in (
            ('/explore', ['POST'], self.explore),
            ('/config', ['GET'], self.config),
            ('/bootstrap', ['GET'], self.bootstrap),
            ('/games', ['POST'], self.store),
            ('/games', ['GET'], self.games),
            ('/games/<game_id>', ['GET'], self.game),
            ('/games/<game_id>', ['PATCH'], self.metadata),
            ('/games/<game_id>', ['DELETE'], self.delete),
            ('/games/<game_id>/analysis', ['GET', 'POST'], self.cache),
            ('/games/<game_id>/analyze', ['POST'], self.analyze),
            ('/games/<game_id>/analyze/cancel', ['POST'], self.cancel_game_analysis),
            ('/maia', ['POST'], self.maia),
            ('/stockfish', ['POST'], self.stockfish),
            ('/stockfish/cancel', ['POST'], self.cancel),
        ):
            routes.add_url_rule(path, view_func=handler, methods=methods)
        server.register_blueprint(routes)

    def explore(self):
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Expected exploration options')
        board = self.platform.history_board(data.get('fen'), data.get('start_fen'), data.get('moves'))
        config = CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION']
        seconds = data.get('seconds', config['DEFAULT_SEARCH_SECONDS'])
        depth = data.get('depth', config['MAX_DEPTH'])
        validate_exploration_seconds(seconds, config['MAX_SEARCH_SECONDS'])
        if type(depth) is not int or not 1 <= depth <= config['MAX_DEPTH']:
            raise ValueError(f"Exploration depth must be an integer from 1 to {config['MAX_DEPTH']}")
        move = data.get('move')
        if move is not None and (not isinstance(move, str) or move not in {m.uci() for m in board.legal_moves}):
            raise ValueError('Exploration move must be legal')
        root = exploration_board(board, move)
        milliseconds = seconds_to_milliseconds(seconds)
        cache_request = stockfish_exploration_request(self.platform.stockfish_signature,
                                                       milliseconds, depth, 1, None)
        persistent = self.platform.persistent_cache_enabled
        cached = self.platform.cache.select_record(root, 'stockfish', cache_request) if persistent else None
        raw = cached['result'] if cached else None
        if cached is not None:
            cache_request = cached['request']
        if raw is None:
            if persistent:
                cache_request = promote_request(self.platform.cache.records(root, 'stockfish'), cache_request)
            milliseconds, depth = cache_request['movetime_ms'], cache_request['depth']
            if root.is_game_over(claim_draw=False):
                raw = {'lines': []}
            else:
                with self.platform.search_engine() as engine:
                    raw = search_lines(engine, root, seconds=milliseconds / 1000, depth=depth)
            if persistent:
                self.platform.cache.put(root, 'stockfish', cache_request, raw)
        return continuation_result(board, raw, move=move,
                                   seconds=cache_request['movetime_ms'] / 1000, depth=cache_request['depth'])

    def config(self):
        platform = self.platform
        presets = {depth: platform.analysis_limits(depth) for depth in platform.analysis_presets()}
        return {'strategy': platform.strategy, 'policy_version': BOUNDED_POLICY_VERSION,
                'analysis_workers': platform.stockfish.analysis_pool.workers if isinstance(platform.stockfish, StockfishScorer) else 1,
                'maia_batch_size': CONFIG['MAIA']['BATCH_SIZE'],
                'default_depth': max(presets),
                'budgets': {str(depth): limits.verify_ms / 1000 for depth, limits in presets.items()},
                'max_budgets': {str(depth): limits.max_ms / 1000 for depth, limits in presets.items()}}

    def store(self):
        return self._store(request.get_json())

    def _store(self, data):
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object')
        parser = PgnAnalysisApi(self.platform.maia, self.platform.stockfish)
        pgn, fen = data.get('pgn'), data.get('fen')
        if bool(pgn) == bool(fen):
            raise ValueError('Provide either PGN or FEN')
        snapshot = parser.import_pgn(str(pgn)) if pgn else parser.import_fen(str(fen))
        name = str(data.get('name') or snapshot['title']).strip()[:200]
        return {'game_id': self.repository.create(snapshot, name)}

    def bootstrap(self):
        game_id = self.repository.latest_id()
        return {'game_id': game_id} if game_id else self._store({'name': 'Custom Analysis', 'fen': chess.STARTING_FEN})

    def games(self):
        return self.repository.list_games(request.args.get('type', 'custom'),
                                          max(1, request.args.get('page', 1, type=int)))

    def game(self, game_id):
        game = self.repository.get(game_id)
        if game is None:
            abort(404, 'Saved analysis not found')
        return game

    def metadata(self, game_id):
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object')
        if not self.repository.update_metadata(game_id, data):
            abort(404)
        return {'ok': True}

    def delete(self, game_id):
        self.repository.delete(game_id)
        return {'ok': True}

    @staticmethod
    def _number_map(values, name, *, probabilities=False):
        if not isinstance(values, dict) or any(
                type(value) not in (int, float) or not math.isfinite(value)
                or (probabilities and not 0 <= value <= 1) for value in values.values()):
            raise ValueError(f'{name} must be an object of finite numbers'
                             + (' from 0 to 1' if probabilities else ''))

    @staticmethod
    def _validate_positions(positions):
        if not isinstance(positions, list) or any(not isinstance(position, dict) for position in positions):
            raise ValueError('Expected an array of position analysis objects')
        for position in positions:
            maia = position.get('maia', {})
            if not isinstance(maia, dict):
                raise ValueError('Maia analysis must be an object keyed by rating')
            for evaluation in maia.values():
                if not isinstance(evaluation, dict):
                    raise ValueError('Each Maia rating evaluation must be an object')
                AnalysisRoutes._number_map(evaluation.get('policy'), 'Maia policy', probabilities=True)
            evaluation = position.get('stockfish', {})
            if not isinstance(evaluation, dict):
                raise ValueError('Stockfish analysis must be an object')
            if 'stockfish' in position:
                AnalysisRoutes._number_map(evaluation.get('cp_vec'), 'Stockfish cp_vec')
            terminal_score = evaluation.get('terminal_cp')
            terminal = (evaluation.get('cp_vec') == {} and type(terminal_score) in (int, float)
                        and math.isfinite(terminal_score))
            for key in ('depth', 'target_depth'):
                minimum = 0 if key == 'depth' and terminal else 1
                if key in evaluation and (type(evaluation[key]) is not int or evaluation[key] < minimum):
                    raise ValueError('Saved search depths must be positive integers (zero for terminal results)')
            for key in ('budget_seconds', 'max_budget_seconds'):
                if key in evaluation:
                    seconds = evaluation[key]
                    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError('Saved search budgets must be finite, positive seconds')

    def cache(self, game_id):
        if request.method == 'POST':
            positions = request.get_json()
            self._validate_positions(positions)
            if not self.platform.full_games.save_positions(game_id, positions):
                abort(404)
            return {'ok': True}
        positions = self.repository.load_analysis(game_id)
        stale = False
        for position in positions:
            evaluation = position.get('stockfish', {})
            if evaluation.get('cp_vec') == {} and type(evaluation.get('terminal_cp')) in (int, float):
                continue
            if evaluation.get('strategy') == 'bounded' or 'max_budget_seconds' in evaluation:
                depth = evaluation.get('target_depth', evaluation.get('depth'))
                try:
                    # The browser checks its selected preset. Preserve complete
                    # observations here even when another preset is the default.
                    valid = (type(depth) is int and depth > 0
                             and evaluation.get('strategy') == self.platform.strategy
                             and evaluation.get('complete') is True
                             and evaluation.get('coverage_complete') is True
                             and evaluation.get('budget_seconds', 0) > 0
                             and (evaluation.get('strategy') != 'bounded' or
                                  evaluation.get('policy_version') == BOUNDED_POLICY_VERSION))
                except (ValueError, TypeError):
                    valid = False
                if not valid:
                    position.pop('stockfish', None)
                    stale = True
            if self.platform.strategy == 'exhaustive' and evaluation.get('strategy') in ('staged', 'bounded'):
                position.pop('stockfish', None)
                stale = True
        return {'positions': positions, 'analysis': None if stale else self.repository.load_full_analysis(game_id)}

    def analyze(self, game_id):
        try:
            run = self.platform.full_games.prepare(game_id, request.get_json())
        except LookupError:
            abort(404, 'Saved game not found')
        response = Response(stream_with_context(self.platform.full_games.stream(run)),
                            mimetype='application/x-ndjson',
                            headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
        response.call_on_close(lambda: self.platform.full_games.release(run))
        return response

    def cancel_game_analysis(self, game_id):
        data = request.get_json()
        if not isinstance(data, dict) or not isinstance(data.get('run_id'), str):
            raise ValueError('Provide a game-analysis run identifier')
        return {'cancelled': self.platform.full_games.cancel(game_id, data['run_id'])}

    def maia(self):
        return self.platform.evaluate_maia(request.get_json())

    def stockfish(self):
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object')
        board = self.platform.history_board(data.get('fen'), data.get('start_fen'), data.get('moves'))
        limits = self.platform.analysis_limits(data.get('depth'), data.get('seconds'))
        options = data.get('options', {})
        if not isinstance(options, dict):
            raise ValueError('Expected search options')
        legal = {move.uci() for move in board.legal_moves}
        for name in ('maiaCandidateMoves', 'forcedCandidateMoves'):
            moves = options.get(name, [])
            if not isinstance(moves, list) or len(moves) > 218 or any(
                    not isinstance(move, str) or move not in legal for move in moves):
                raise ValueError('Search candidates must be legal UCI moves')
        search_id = data.get('search_id', uuid.uuid4().hex)
        if not isinstance(search_id, str) or not 1 <= len(search_id) <= 64:
            raise ValueError('Invalid search identifier')
        control = SearchControl()
        platform = self.platform
        with platform.controls_lock:
            if search_id in platform.search_controls:
                raise ValueError('Search identifier already active')
            platform.search_controls[search_id] = control

        def stream():
            try:
                yield from platform.stream_stockfish(board, limits.depth, options, control,
                                                    seconds=limits.verify_ms / 1000)
            except Exception as error:
                current_app.logger.exception('Native Stockfish analysis failed')
                yield json.dumps({'error': str(error)}) + '\n'
            finally:
                control.cancel()
                with platform.controls_lock:
                    platform.search_controls.pop(search_id, None)

        return Response(stream_with_context(stream()), mimetype='application/x-ndjson',
                        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    def cancel(self):
        data = request.get_json()
        if not isinstance(data, dict) or not isinstance(data.get('search_id'), str):
            raise ValueError('Expected a search identifier')
        with self.platform.controls_lock:
            control = self.platform.search_controls.get(data['search_id'])
            if control:
                control.cancel()
        return {'ok': True}
