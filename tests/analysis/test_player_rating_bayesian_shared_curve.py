"""Shared-curve inference, account-free evidence and legacy-cache separation."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import chess
import chess.pgn
import numpy as np

from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.policies import prepare_rating_policies
from analysis.player_rating.service import fit_game, fit_evidence, rating_signature, selected_method
from analysis.player_rating.evidence import collect_evidence, evidence_compatible
from analysis.settings import CONFIG
from analysis.player_rating.bayesian_shared_curve import Args, Rating, fit_pair, summarize


def record(good_moves, count=20):
    probability = np.linspace(.1, .95, len(RATINGS))
    return {'observations': [
        {'qualities': {'position': [100., 35.]},
         'maia_probabilities': np.c_[probability, 1-probability].tolist(),
         'played_index': 0 if index < good_moves else 1, 'weight': 1.}
        for index in range(count)]}


def game_records(pgn='1. e4 e5 2. Nf3 Nc6 *'):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board, records = game.board(), []
    for move in game.mainline_moves():
        records.append({'move': move.uci(), 'side': 'White' if board.turn else 'Black',
                        'position_score': 0,
                        'scores': {m.uci(): 200 if m == move else 0 for m in board.legal_moves}})
        board.push(move)
    return game, records


def evaluate(board, own, other):
    if own != other:
        raise AssertionError('Shared inference must only request equal ratings.')
    legal = sorted(m.uci() for m in board.legal_moves)
    return [{'policy': {move: 1/len(legal) for move in legal}} for _ in own]


class SharedCurveTests(unittest.TestCase):
    def setUp(self):
        selector = patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='bayesian_shared_curve')
        selector.start()
        self.addCleanup(selector.stop)

    def test_default_method_has_no_trained_model_dependency(self):
        with patch('pathlib.Path.read_text', side_effect=AssertionError('model file read')):
            self.assertEqual(selected_method(), 'bayesian_shared_curve')
            self.assertEqual(rating_signature()['method'], 'bayesian_shared_curve')

    def test_module_rating_class_uses_filename_identity_and_local_arguments(self):
        estimator = Rating()
        self.assertEqual(estimator.id, 'bayesian_shared_curve')
        self.assertEqual(estimator.name, 'Bayesian shared-curve fit')
        self.assertEqual(estimator.version, 8)
        evidence = {'White': record(16), 'Black': record(13)}
        self.assertEqual(estimator.fit(evidence), summarize(evidence))
        self.assertEqual(estimator.parameters['central_interval'], .20)
        self.assertEqual(estimator.parameters['accuracy_sigma_scale'], 1.)
        self.assertEqual(estimator.parameters['top_probability'], 1.)
        self.assertEqual(estimator.parameters['prior_range'], (200., 3000.))
        self.assertEqual(estimator.parameters['flat_prior_range'], (800., 2400.))

    def test_likelihood_arguments_invalidate_fit_signature(self):
        current = rating_signature()
        self.assertEqual(current['method_version'], 8)
        for args in (Args(accuracy_sigma_scale=.25), Args(accuracy_sigma_scale=.5),
                     Args(top_probability=.68), Args(top_probability=.99),
                     Args(prior_range=(0., 3200.)), Args(flat_prior_range=(1000., 2200.))):
            with self.subTest(args=args):
                changed = rating_signature(args=args)
                self.assertNotEqual(changed, current)
                self.assertEqual(changed['method'], current['method'])

    def test_quality_order_and_metadata_invariance(self):
        white, black = record(16), record(13)
        fit = fit_pair(white, black)
        self.assertGreater(fit['white_minus_black'], 0)
        for actual in (100, 1300, 1700, 4000):
            altered = fit_pair(dict(white, actual_elo=actual, reference_elo=2600),
                               dict(black, actual_elo=4100-actual, reference_elo=600))
            self.assertEqual(altered, fit)
        self.assertGreaterEqual(fit_pair(record(17), black)['players'][0]['estimate'],
                                fit['players'][0]['estimate'])

    def test_interval_probability_changes_bounds_not_center(self):
        small = fit_pair(record(16), record(13), args=Args(central_interval=.2))
        large = fit_pair(record(16), record(13), args=Args(central_interval=.9))
        self.assertEqual(small['posterior_densities'], large['posterior_densities'])
        for first, second in zip(small['players'], large['players'], strict=True):
            self.assertEqual(first['estimate'], second['estimate'])
            self.assertGreaterEqual(first['conditional_interval'][0], second['conditional_interval'][0])
            self.assertLessEqual(first['conditional_interval'][1], second['conditional_interval'][1])

    def test_empty_one_sided_and_flat_data_do_not_claim_an_unobserved_rating(self):
        empty = {'observations': []}
        both_empty = fit_pair(empty, empty)
        self.assertEqual([p['estimate'] for p in both_empty['players']], [None, None])
        one = fit_pair(record(16), empty)
        self.assertIsNotNone(one['players'][0]['estimate'])
        self.assertIsNone(one['players'][1]['estimate'])
        forced = {'observations': [{'qualities': {'position': [100.]},
                                   'maia_probabilities': [[1.]]*len(RATINGS), 'played_index': 0}]}
        flat = fit_pair(forced, forced)
        self.assertFalse(flat['identifiable'])
        self.assertEqual(flat['players'][0]['conditional_interval'], [0., 3200.])
        self.assertIsNone(flat['players'][0]['estimate'])

    def test_invalid_distribution_and_played_index_fail(self):
        malformed = record(16)
        malformed['observations'][0]['maia_probabilities'][0][0] = -1
        with self.assertRaisesRegex(ValueError, 'normalized'):
            fit_pair(malformed, record(13))
        malformed = record(16)
        malformed['observations'][0]['played_index'] = True
        with self.assertRaisesRegex(ValueError, 'Played index'):
            fit_pair(malformed, record(13))

    def test_account_free_cached_evidence_reuses_ratings_but_rejects_legacy(self):
        game, rows = game_records()
        callback = Mock(side_effect=evaluate)
        with tempfile.TemporaryDirectory() as directory:
            first = fit_game(game, rows, callback, directory, 'fixture')
            calls = callback.call_count
            game.headers.update(WhiteElo='2700', BlackElo='100', WhiteEloEstimate='1')
            second = fit_game(game, rows, callback, directory, 'fixture')
            self.assertNotEqual(first['rating_fit']['context_signature'], second['rating_fit']['context_signature'])
            self.assertEqual(first['rating_scale']['actual_ratings'], {'White': None, 'Black': None})
            self.assertEqual(second['rating_scale']['actual_ratings'], {'White': 2700., 'Black': 100.})
            self.assertEqual(second['rating_scale']['native_actual_ratings'], {'White': 2700., 'Black': 100.})
            for result in (first, second):
                result['rating_fit'].pop('context_signature')
                result['rating_scale'].pop('actual_ratings')
                result['rating_scale'].pop('native_actual_ratings')
            self.assertEqual(first, second)
            self.assertEqual(callback.call_count, calls)
            evidence = json.loads(next((Path(directory)/'player-rating').glob('rating-*.json')).read_text(encoding='utf-8'))
            self.assertTrue(evidence_compatible(evidence))
            self.assertNotIn('actual_elo', evidence['White'])
            for side in ('White', 'Black'):
                evidence[side].pop('conditioning')
            with self.assertRaisesRegex(ValueError, 'Rerun full analysis'):
                fit_evidence(evidence)
            cache = next((Path(directory)/'player-rating').glob('rating-*.json'))
            cache.write_text(json.dumps(evidence), encoding='utf-8')
            rebuilt = fit_game(game, rows, callback, directory, 'fixture')
            self.assertGreater(callback.call_count, calls)
            self.assertEqual(rebuilt['players'], first['players'])

    def test_history_and_diagonal_graph_policies_are_batched_and_cached(self):
        game, rows = game_records()
        calls = []
        def many(requests):
            calls.append(requests)
            return [evaluate(*request) for request in requests]
        with tempfile.TemporaryDirectory() as directory:
            shell = prepare_rating_policies(game, rows, Mock(side_effect=AssertionError('serial')), directory,
                                      'fixture', evaluate_many=many)
            self.assertIsNone(shell)
            self.assertEqual([len(b.move_stack) for group in calls for b, _, _ in group], [0, 1, 2, 3])
            self.assertTrue(all(sum(len(own) for _, own, _ in group) <= CONFIG['MAIA']['BATCH_SIZE'] for group in calls))
            self.assertTrue(all(set(row['policies']) == set(RATINGS) for row in rows))
            prepare_rating_policies(game, rows, Mock(side_effect=AssertionError('cached')), directory, 'fixture')
        calls.clear()
        canonical = [{k: v for k, v in row.items() if k != 'policies'} for row in rows]
        evidence = collect_evidence(game, canonical, Mock(side_effect=AssertionError('serial')), evaluate_many=many)
        self.assertEqual([len(b.move_stack) for group in calls for b, _, _ in group], [0, 1, 2, 3])
        self.assertTrue(evidence_compatible(evidence))

    def test_complete_diagonal_graph_policies_supply_evidence_without_reinference(self):
        game, rows = game_records()
        with tempfile.TemporaryDirectory() as directory:
            prepare_rating_policies(game, rows, evaluate, directory, 'fixture')
        callback = Mock(side_effect=AssertionError('already have exact diagonal policies'))
        evidence = collect_evidence(game, rows, callback)
        self.assertTrue(evidence_compatible(evidence))
        rows[0]['policies'].pop(600)
        with self.assertRaisesRegex(ValueError, 'exact diagonal'):
            collect_evidence(game, rows, callback)

    def test_corrupt_policy_cache_is_rebuilt(self):
        game, rows = game_records()
        callback = Mock(side_effect=evaluate)
        with tempfile.TemporaryDirectory() as directory:
            prepare_rating_policies(game, rows, callback, directory, 'fixture')
            expected = copy.deepcopy([row['policies'] for row in rows])
            cache = next(Path(directory).glob('diagonal-policies-*.json'))
            for corrupt in ('{unfinished', 'null', '[{}]', json.dumps([{} for _ in rows])):
                calls = callback.call_count
                cache.write_text(corrupt, encoding='utf-8')
                prepare_rating_policies(game, rows, callback, directory, 'fixture')
                self.assertGreater(callback.call_count, calls)
                self.assertEqual([row['policies'] for row in rows], expected)

    def test_empty_fen_game_does_not_infer(self):
        game = chess.pgn.Game()
        game.setup(chess.Board('8/8/8/4k3/8/8/3K4/8 b - - 0 1'))
        evidence = collect_evidence(game, [], Mock(side_effect=AssertionError('no moves')))
        summary = fit_evidence(evidence)
        self.assertEqual([p['estimate'] for p in summary['players'].values()], [None, None])

    def test_frozen_game13_summary_matches_pair_calculation(self):
        path = Path(__file__).with_name('data')/'shared-curve-game13.json'
        if not path.exists():
            self.skipTest('Optional saved game13 test evidence is unavailable.')
        records = json.loads(path.read_text(encoding='utf-8'))['records']
        fit = fit_pair(*records)
        summary = summarize(dict(zip(('White', 'Black'), records)))
        self.assertEqual([p['estimate'] for p in summary['players'].values()], [p['estimate'] for p in fit['players']])
        self.assertEqual(summary['central_interval'], .2)

    def test_paper_evidence_matches_full_probability_gaussian_calculation(self):
        path = Path(__file__).resolve().parents[2] / 'docs/data/bayesian_shared_curve_game10.json'
        evidence = json.loads(path.read_text(encoding='utf-8'))
        result = summarize(evidence)
        self.assertEqual([p['estimate'] for p in result['players'].values()], [2168, 1370])
        self.assertEqual([p['interval'] for p in result['players'].values()], [[2053, 2276], [1249, 1497]])
        diagnostics = result['diagnostics']['curve']
        likelihood = diagnostics['likelihood']
        self.assertEqual(likelihood['kind'], 'gaussian_accuracy')
        self.assertAlmostEqual(likelihood['accuracy_sigma'], 3.5233494203555002)
        self.assertAlmostEqual(likelihood['accuracy_variance'],
                               diagnostics['shared_mean_variance'])
        self.assertEqual(result['prior']['kind'], 'symmetric_fourth_power')


if __name__ == '__main__':
    unittest.main()
