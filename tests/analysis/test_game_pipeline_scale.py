"""Native Maia scheduling and persisted rating-scale context through the pipeline."""
import io
import unittest
from unittest.mock import patch

import chess.pgn

from analysis import elo_convert
from analysis.game.pipeline import analyze_game
from analysis.player_rating.parameters import RATINGS as FIT_RATINGS
from analysis.position_evaluation import RATINGS
from analysis.settings import CONFIG
from tests.coach.fixtures import FakeEngines


class RatingSpecificEngines(FakeEngines):
    """Make each rating's leading legal move distinct and record SF priorities."""

    def __init__(self, start_fen):
        super().__init__(start_fen)
        self.jobs = []

    def human_pairs(self, history, ratings, opponents):
        self.pair_calls.append((history.copy(), list(ratings), list(opponents)))
        legal = sorted(move.uci() for move in self.board(history).legal_moves)
        result = []
        for rating in ratings:
            focus = legal[((rating-600)//100) % len(legal)]
            policy = {move: (1. if len(legal) == 1 else .6 if move == focus else .4/(len(legal)-1))
                      for move in legal}
            result.append({'policy': policy, 'value': .5})
        return result

    def initial_analysis(self, history, played, human_candidates):
        self.jobs.append((history.copy(), played, list(human_candidates)))
        return super().initial_analysis(history, played, human_candidates)


class PipelineScaleTests(unittest.TestCase):
    def game(self):
        return chess.pgn.read_game(io.StringIO(
            '[Site "Chess.com"]\n[TimeControl "600+0"]\n[WhiteElo "1500"]\n'
            '[BlackElo "1400"]\n\n1. e4 e5 *'))

    def engine(self, game):
        engine = RatingSpecificEngines(game.board().fen())
        self.addCleanup(engine._temp.cleanup)
        return engine

    def test_priority_moves_use_native_header_rating_and_maia_anchors_remain_native(self):
        game = self.game()
        engine = self.engine(game)
        with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'):
            analysis = analyze_game(game, engine, 'white', 1700, progress=lambda _: None)
        self.assertEqual(analysis['played_elo_scale']['scale'], 'cr')
        self.assertNotIn('rating_scale_override', analysis)
        for history, own, other in engine.pair_calls:
            self.assertEqual(own, list(FIT_RATINGS))
            self.assertEqual(other, list(FIT_RATINGS))
        for index, side in enumerate(('White', 'Black')):
            history, _, priority = engine.jobs[index]
            native = elo_convert.convert(int(game.headers[side+'Elo']), 'cr', 'lb', extrapolate=True)
            nearest = min(RATINGS, key=lambda rating: abs(rating-native))
            legal = sorted(move.uci() for move in engine.board(history).legal_moves)
            expected = legal[((nearest-600)//100) % len(legal)]
            self.assertEqual(priority[0], expected)
            # Distinguish the source-scale bug from native-coordinate priority.
            wrong = min(RATINGS, key=lambda rating: abs(rating-int(game.headers[side+'Elo'])))
            self.assertNotEqual(priority[0], legal[((wrong-600)//100) % len(legal)])
        self.assertEqual(analysis['rating_account_overrides'], {'White': 1700})
        self.assertEqual(analysis['selected_player'], {'side': 'white', 'actual_elo': 1700})
        self.assertEqual(analysis['played_elo_scale']['actual_ratings'], {'White': 1700., 'Black': 1400.})

    def test_explicit_scale_is_persisted_and_controls_priority_without_changing_headers(self):
        game = self.game()
        engine = self.engine(game)
        with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'):
            analysis = analyze_game(game, engine, progress=lambda _: None, rating_scale='lb')
        self.assertEqual(analysis['rating_scale_override'], 'lb')
        self.assertEqual(analysis['played_elo_scale']['scale'], 'lb')
        self.assertEqual(analysis['played_elo_scale']['source'], 'override')
        self.assertEqual(analysis['headers'], dict(game.headers))
        legal = sorted(move.uci() for move in game.board().legal_moves)
        self.assertEqual(engine.jobs[0][2][0], legal[(1500-600)//100])

    def test_unsupported_scale_fails_before_any_analysis_engine_calls(self):
        game = self.game()
        game.headers.update(Site='lichess.org', TimeControl='60+0')
        engine = self.engine(game)
        with self.assertRaises(elo_convert.UnsupportedRatingScale):
            analyze_game(game, engine, progress=lambda _: None)
        self.assertFalse(engine.pair_calls or engine.sf_calls or engine.jobs)

    def test_unreachable_header_or_selected_rating_floor_fails_before_inference(self):
        for source in ('header', 'selected'):
            with self.subTest(source=source):
                game = self.game()
                if source == 'header':
                    game.headers['BlackElo'] = '100'
                engine = self.engine(game)
                actual = 100 if source == 'selected' else 1500
                with self.assertRaisesRegex(ValueError, 'lower asymptote'):
                    analyze_game(game, engine, 'white', actual, progress=lambda _: None)
                self.assertFalse(engine.pair_calls or engine.sf_calls or engine.jobs)


if __name__ == '__main__':
    unittest.main()
