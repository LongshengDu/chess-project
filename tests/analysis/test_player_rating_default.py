"""Production selection and cached migration into current-game affine fitting."""
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from analysis.player_rating.service import evidence_cache_path, fit_game, get_estimator, refresh_saved_rating, store_elo_fit
from analysis.settings import CONFIG
from tests.analysis.test_player_rating_bayesian_shared_curve import game_records


class DefaultRatingTests(unittest.TestCase):
    def test_new_default_and_original_method_are_both_loadable(self):
        self.assertEqual(CONFIG['ANALYSIS']['PLAYER_RATING']['METHOD'], 'shared_curve_affine')
        self.assertEqual(get_estimator().id, 'shared_curve_affine')
        self.assertEqual(get_estimator('bayesian_shared_curve').id, 'bayesian_shared_curve')
        for method in ('hierarchical_affine', 'uncertainty_ensemble'):
            with self.assertRaisesRegex(ValueError, 'Unknown rating method'):
                get_estimator(method)
        with self.assertRaisesRegex(ValueError, 'withdrawn'):
            get_estimator('arithmetic_coverage')

    def test_old_cache_migrates_without_engine_work_and_refits_for_account_change(self):
        game, records = game_records('1. e4 e5 2. Nf3 Nc6 *')
        game.headers.update(WhiteElo='1600', BlackElo='1700')
        board, best_moves = game.board(), {}
        for move, row in zip(game.mainline_moves(), records, strict=True):
            choose = max if board.turn else min
            best_moves[board.fen()] = choose(row['scores'], key=row['scores'].get)
            board.push(move)

        def evaluate(board, own, other):
            legal = [move.uci() for move in board.legal_moves]
            best = best_moves[board.fen()]
            policies = []
            for rating in own:
                probability = .1 + .85 * (rating - 600) / 2000
                policies.append({'policy': {
                    move: probability if move == best else (1-probability)/(len(legal)-1)
                    for move in legal}})
            return policies

        evaluate_counted = Mock(side_effect=evaluate)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve'):
                old = fit_game(game, records, evaluate_counted, directory, 'fixture')
            key = old['rating_fit']['evidence_key']
            cache = evidence_cache_path(directory, key)
            legacy = json.loads(cache.read_text(encoding='utf-8'))
            for side in legacy.values():
                for row in side['observations']:
                    row.pop('position_win_probability')
            cache.write_text(json.dumps(legacy), encoding='utf-8')
            analysis = {'headers': dict(game.headers), 'moves': [
                {'side': 'white' if i % 2 == 0 else 'black', 'position_eval': row['position_score']/100.}
                for i, row in enumerate(records)]}
            store_elo_fit(analysis, old)
            count = evaluate_counted.call_count
            refresh_saved_rating(analysis, directory)
            self.assertEqual(analysis['rating_fit']['evidence_key'], key)
            self.assertEqual(analysis['played_elo_method'], 'shared_curve_affine')
            self.assertIsNone(analysis['played_elo_central_interval'])
            self.assertTrue(analysis['played_elo_account_ratings_used'])
            anchor_before = analysis['played_elo_diagnostics']['affine']['account_anchor']
            signature_before = analysis['rating_fit']['context_signature']
            unrounded_before = {side: analysis['played_elo_diagnostics']['components'][side]['unrounded_estimate']
                                for side in ('White', 'Black')}
            analysis['headers']['WhiteElo'] = '1800'
            refresh_saved_rating(analysis, directory)
            self.assertNotEqual(analysis['rating_fit']['context_signature'], signature_before)
            self.assertAlmostEqual(analysis['played_elo_diagnostics']['affine']['account_anchor']-anchor_before, 100.)
            self.assertTrue(any(analysis['played_elo_diagnostics']['components'][side]['unrounded_estimate']
                                != unrounded_before[side] for side in ('White', 'Black')))
            self.assertEqual(evaluate_counted.call_count, count)
            self.assertEqual(json.loads(cache.read_text(encoding='utf-8')), legacy)


if __name__ == '__main__':
    unittest.main()
