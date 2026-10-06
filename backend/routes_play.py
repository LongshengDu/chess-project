"""HTTP controller for the local Play Maia page."""
from flask import Blueprint, request

from backend.settings import CONFIG


class PlayRoutes:
    def __init__(self, play):
        self.play = play

    def register(self, server):
        routes = Blueprint('play', __name__, url_prefix='/api/platform/play')
        for path, methods, handler in (
            ('/config', ['GET'], self.config),
            ('/games', ['POST'], self.start),
            ('/move', ['POST'], self.move),
            ('/games/<game_id>', ['GET'], self.game),
            ('/games/<game_id>/moves', ['POST'], self.submit),
            ('/games/<game_id>/analysis', ['POST'], self.analysis),
            ('/stats', ['GET'], self.stats),
        ):
            routes.add_url_rule(path, view_func=handler, methods=methods)
        server.register_blueprint(routes)

    @staticmethod
    def _body():
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object')
        return data

    def config(self):
        return {'player_rating': CONFIG['MAIA']['PLAYER_RATING'], 'model': CONFIG['MAIA']['MODEL'],
                'temperature': CONFIG['MAIA']['TEMPERATURE']}

    def start(self):
        return self.play.start(self._body())

    def move(self):
        return self.play.move(self._body())

    def game(self, game_id):
        return self.play.get(game_id)

    def submit(self, game_id):
        return self.play.submit(game_id, self._body())

    def analysis(self, game_id):
        return self.play.save_for_analysis(game_id)

    def stats(self):
        return self.play.stats()
