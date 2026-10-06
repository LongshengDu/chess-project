"""Opt-in profiler controls, separate from normal analysis routes."""
from flask import Blueprint, request

from engine.stockfish import StockfishScorer


class ProfilerRoutes:
    def __init__(self, platform):
        self.platform = platform

    def register(self, server):
        routes = Blueprint('profiler', __name__, url_prefix='/api/platform/profiler')
        for path, methods, handler in (
            ('', ['GET'], self.status),
            ('/start', ['POST'], self.start),
            ('/position', ['POST'], self.position),
            ('/finish', ['POST'], self.finish),
        ):
            routes.add_url_rule(path, view_func=handler, methods=methods)
        server.register_blueprint(routes)

    def status(self):
        return self.platform.profiler.status()

    def start(self):
        data = request.get_json()
        if (not isinstance(data, dict) or type(data.get('total_positions')) is not int
                or data['total_positions'] < 1):
            raise ValueError('Expected profiler configuration')
        platform = self.platform
        if 'target_depth' not in data:
            raise ValueError('Expected a profiler target depth')
        limits = platform.analysis_limits(data['target_depth'], data.get('seconds'))
        # Clear request caches at the button boundary; retain normal intra-run hash.
        with platform.maia_lock, platform.stockfish_lock:
            if platform.profiler.active:
                raise ValueError('Profiling is already running')
            platform.maia_cache.clear()
            with platform.stockfish_cache_lock:
                platform.stockfish_cache.clear()
            if isinstance(platform.stockfish, StockfishScorer):
                platform.stockfish.analysis_pool.clear_hash()
            if platform.stockfish._engine is not None:
                platform.stockfish._engine.configure({'Clear Hash': None})
            return platform.profiler.start({**data, 'pipeline': 'shared',
                'position_budget_seconds': limits.verify_ms / 1000 if platform.strategy == 'bounded' else None,
                'max_search_seconds': limits.max_ms / 1000},
                platform.maia, platform.stockfish, platform.strategy)

    def position(self):
        data = self._run_data()
        if type(data.get('ply')) is not int or data['ply'] < 0:
            raise ValueError('Expected a nonnegative position ply')
        self.platform.profiler.position(data)
        return {'ok': True}

    def finish(self):
        self.platform.profiler.finish(self._run_data())
        return {'ok': True}

    @staticmethod
    def _run_data():
        data = request.get_json()
        if not isinstance(data, dict) or not isinstance(data.get('run_id'), str):
            raise ValueError('Expected a profiler run identifier')
        return data
