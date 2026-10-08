"""Native Maia scheduling and persisted rating-scale context through the pipeline."""
import io
import unittest

import chess.pgn

from analysis import elo_convert
from analysis.game.pipeline import analyze_game
from analysis.accuracy.evidence import RATINGS as MAIA_RATINGS
from analysis.position_evaluation import RATINGS
from analysis.maia_context import analysis_scale, native_actual_ratings
from tests.coach.fixtures import FakeAnalysisSession


class RatingSpecificAnalysisSession(FakeAnalysisSession):
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

    def session(self, game):
        session = RatingSpecificAnalysisSession(game.board().fen())
        self.addCleanup(session._temp.cleanup)
        return session

    def test_priority_moves_use_effective_native_ratings_and_maia_anchors_remain_native(self):
        game = self.game()
        session = self.session(game)
        analysis = analyze_game(game, session, actual_elo=1700, progress=lambda _: None)
        self.assertEqual(analysis_scale(analysis), 'cr')
        self.assertEqual(analysis['game']['rating_scale'], 'chess_com_rapid')
        for history, own, other in session.pair_calls:
            self.assertEqual(own, list(MAIA_RATINGS))
            self.assertEqual(other, list(MAIA_RATINGS))
        for index, side in enumerate(('White', 'Black')):
            history, _, priority = session.jobs[index]
            native = elo_convert.convert(1700, 'cr', 'lb', extrapolate=True)
            nearest = min(RATINGS, key=lambda rating: abs(rating-native))
            legal = sorted(move.uci() for move in session.board(history).legal_moves)
            expected = legal[((nearest-600)//100) % len(legal)]
            self.assertEqual(priority[0], expected)
            # Distinguish the source-scale bug from native-coordinate priority.
            wrong = min(RATINGS, key=lambda rating: abs(rating-1700))
            self.assertNotEqual(priority[0], legal[((wrong-600)//100) % len(legal)])
        for side in ('white', 'black'):
            self.assertEqual(analysis['game'][side], {'name': None, 'elo': 1700})
        self.assertEqual(analysis['coaching'], {})
        self.assertAlmostEqual(native_actual_ratings(analysis)['White'], elo_convert.convert(1700, 'cr', 'lb', extrapolate=True))
        self.assertEqual(native_actual_ratings(analysis)['White'], native_actual_ratings(analysis)['Black'])
        self.assertEqual(analysis['accuracy_curve']['rating_scale'], 'lb')

    def test_explicit_scale_is_persisted_and_controls_priority_without_changing_headers(self):
        game = self.game()
        session = self.session(game)
        analysis = analyze_game(game, session, progress=lambda _: None, rating_scale='lb')
        self.assertEqual(analysis['game']['rating_scale'], 'lichess_blitz')
        self.assertEqual(analysis_scale(analysis), 'lb')
        self.assertEqual(analysis['headers'], dict(game.headers))
        legal = sorted(move.uci() for move in game.board().legal_moves)
        self.assertEqual(session.jobs[0][2][0], legal[(1500-600)//100])

    def test_unsupported_scale_fails_before_any_analysis_engine_calls(self):
        game = self.game()
        game.headers.update(Site='lichess.org', TimeControl='60+0')
        session = self.session(game)
        with self.assertRaises(elo_convert.UnsupportedRatingScale):
            analyze_game(game, session, progress=lambda _: None)
        self.assertFalse(session.pair_calls or session.sf_calls or session.jobs)

    def test_unreachable_effective_rating_floor_fails_before_inference(self):
        for source in ('header', 'override'):
            with self.subTest(source=source):
                game = self.game()
                if source == 'header':
                    game.headers['BlackElo'] = '100'
                session = self.session(game)
                actual = 100 if source == 'override' else None
                with self.assertRaisesRegex(ValueError, 'lower asymptote'):
                    analyze_game(game, session, actual_elo=actual, progress=lambda _: None)
                self.assertFalse(session.pair_calls or session.sf_calls or session.jobs)

    def test_explicit_pair_overrides_unusable_pgn_ratings_and_scale(self):
        game = self.game()
        game.headers.update(Site='?', TimeControl='?', WhiteElo='?', BlackElo='100')
        session = self.session(game)
        result = analyze_game(game, session, actual_elo=1600, rating_scale='lb', progress=lambda _: None)
        self.assertEqual(result['coaching'], {})
        self.assertEqual(native_actual_ratings(result), {'White': 1600., 'Black': 1600.})
        self.assertEqual(result['headers'], dict(game.headers))
        for history, _, priority in session.jobs[:-1]:
            legal = sorted(move.uci() for move in session.board(history).legal_moves)
            self.assertEqual(priority[0], legal[((1600-600)//100) % len(legal)])


if __name__ == '__main__':
    unittest.main()
