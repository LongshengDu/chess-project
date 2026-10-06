"""Offline paired-method comparisons with strict input and cohort safeguards."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON
from unittest.mock import Mock, patch

import chess
import chess.pgn
import numpy as np

from analysis.cache import write_json
from analysis.player_rating.evidence import collect_evidence
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.service import evidence_cache_path
from tests.analysis import compare_rating_methods as comparison


def prepare_fixture(root, number=16):
    games, cache = root/'games', root/'cache'
    games.mkdir(exist_ok=True)
    pgn = games/f'game{number}.pgn'
    pgn.write_text('[WhiteElo "1600"]\n[BlackElo "1700"]\n'
                   '[WhiteEloEstimate "2055"]\n[BlackEloEstimate "2055"]\n\n1. e4 e5 *\n', encoding='utf-8')
    game = chess.pgn.read_game(io.StringIO(pgn.read_text(encoding='utf-8')))
    board, records, moves = game.board(), [], []
    for index, played in enumerate(game.mainline_moves()):
        legal = sorted(move.uci() for move in board.legal_moves)
        scores = {move: 80-12*i if board.turn else -80+12*i for i, move in enumerate(legal)}
        policies = {}
        for rating in RATINGS:
            probability = np.exp(-np.arange(len(legal))/((3200-rating)/250.))
            probability /= probability.sum()
            policies[rating] = dict(zip(legal, probability.tolist(), strict=True))
        before = 15 if index == 0 else 20
        records.append({'move': played.uci(), 'position_score': before, 'scores': scores, 'policies': policies})
        moves.append({'fen': board.fen(), 'side': 'white' if board.turn else 'black',
                      'position_eval': before/100., 'played': {'move': played.uci(), 'eval': scores[played.uci()]/100.},
                      'candidate_moves': [{'move': move, 'eval': score/100.} for move, score in scores.items()]})
        board.push(played)
    no_engine = Mock(side_effect=AssertionError('No engine inference is permitted.'))
    evidence = collect_evidence(game, records, no_engine)
    no_engine.assert_not_called()
    key = 'a'*64
    write_json(evidence_cache_path(cache, key), evidence)
    analysis = {'headers': dict(game.headers), 'moves': moves, 'rating_fit': {'evidence_key': key},
                'played_elo': {'white': {'estimate': 1600}, 'black': {'estimate': 1700}},
                'played_elo_method': 'bayesian_shared_curve'}
    analysis_path = games/'output'/f'game{number}-full'/'analysis.json'
    write_json(analysis_path, analysis)
    return games, cache, analysis_path


class CompareRatingMethodsTests(unittest.TestCase):
    @unittest.skip(WITHDRAWN_TEST_REASON)
    def test_real_offline_methods_share_evidence_and_keep_their_separate_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            games, cache, path = prepare_fixture(root)
            original = path.read_bytes()
            config_before = deepcopy(comparison.CONFIG['ANALYSIS']['PLAYER_RATING'])
            with patch.object(comparison, '_fit', wraps=comparison._fit) as fitted, \
                    patch.object(comparison, '_reference', wraps=comparison._reference) as reference, \
                    patch('sys.stdout', new=io.StringIO()):
                calls = Mock()
                calls.attach_mock(fitted, 'fit')
                calls.attach_mock(reference, 'reference')
                result = comparison.run(games, root/'results', start=16, end=16, cache=cache)
            self.assertEqual(comparison.CONFIG['ANALYSIS']['PLAYER_RATING'], config_before)
            self.assertEqual(fitted.call_count, 2)
            self.assertEqual(fitted.call_args_list[0].args[0], fitted.call_args_list[1].args[0])
            call_names = [call[0] for call in calls.mock_calls]
            self.assertLess(max(i for i, name in enumerate(call_names) if name == 'fit'),
                            min(i for i, name in enumerate(call_names) if name == 'reference'))
            self.assertEqual((root/'results/original-analysis/game16.json').read_bytes(), original)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(saved['played_elo_method'], 'bayesian_shared_curve')
            for folder, method in ((root/'results/game16/uncertainty_ensemble', 'uncertainty_ensemble'),
                                   (root/'results/game16/bayesian_shared_curve', 'bayesian_shared_curve')):
                self.assertTrue((folder/'analysis.svg').is_file())
                self.assertTrue((folder/'prior.svg').is_file())
                self.assertEqual(json.loads((folder/'fit.json').read_text(encoding='utf-8'))['method_id'], method)
            self.assertEqual(result['games'][0]['calibration']['contexts_excluded'], 0)
            self.assertEqual(result['games'][0]['calibration']['contexts_used'], 16)
            self.assertEqual(result['summary']['original_games_0_15']['uncertainty_ensemble']['players'], 0)
            self.assertEqual(result['summary']['new_games_16_18']['uncertainty_ensemble']['players'], 2)
            for method in comparison.METHODS:
                metrics = result['summary']['all_games'][method]
                self.assertEqual(metrics['ordering']['decisive_games'], 0)
                self.assertEqual(len(metrics['ordering']['reference_ties']), 1)
            self.assertIn('| game16 | 2055 / 2055 |', (root/'results/comparison.md').read_text(encoding='utf-8'))
            with (root/'results/comparison.csv').open(encoding='utf-8', newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)), result['table'])

    def test_missing_later_game_is_detected_before_fitting_or_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            games, cache, path = prepare_fixture(root)
            original = path.read_bytes()
            with patch.object(comparison, '_fit') as fitted, self.assertRaises(FileNotFoundError):
                comparison.load_cases(games, range(16, 18), cache)
            fitted.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((root/'results').exists())

    def test_mismatched_pgn_evidence_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            games, cache, path = prepare_fixture(root)
            pgn = games/'game16.pgn'
            pgn.write_text(pgn.read_text(encoding='utf-8').replace('1. e4 e5', '1. d4 d5'), encoding='utf-8')
            original = path.read_bytes()
            with patch.object(comparison, '_fit') as fitted, self.assertRaisesRegex(ValueError, 'does not match'):
                comparison.load_cases(games, range(16, 17), cache)
            fitted.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((root/'results').exists())

    def test_displayed_errors_and_reference_ties_have_distinct_metrics(self):
        rows = [{'game': game, 'side': side, 'method_id': 'method', 'reference': reference,
                 'estimate': estimate, 'unrounded_estimate': raw}
                for game, side, reference, estimate, raw in (
                    ('game16', 'White', 2000, 2010, 2009.6), ('game16', 'Black', 2000, 1990, 1990.4),
                    ('game17', 'White', 2200, 2100, 2100.2), ('game17', 'Black', 1800, 1900, 1900.2))]
        metrics = comparison.method_metrics(rows, 'method')
        self.assertEqual(metrics['mean_absolute_error'], 55.)
        self.assertAlmostEqual(metrics['unrounded_mean_absolute_error'], 54.8)
        self.assertEqual(metrics['ordering']['strict_matched_including_reference_ties'], 1)
        self.assertEqual(metrics['ordering']['decisive_matched'], 1)
        self.assertEqual(metrics['ordering']['decisive_games'], 1)
        self.assertEqual(metrics['ordering']['reference_ties'][0]['white_minus_black'], 20)


if __name__ == '__main__':
    unittest.main()
