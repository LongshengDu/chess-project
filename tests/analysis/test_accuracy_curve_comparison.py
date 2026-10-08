"""Prepared accuracy comparisons preserve scope, color symmetry and uncertainty."""
import copy
import io
import unittest
from unittest.mock import patch

import chess
import chess.pgn

from analysis.accuracy.comparison import AccuracyComparison
from analysis.accuracy.by_move import expected_accuracy_by_move
from analysis.accuracy.evidence import RATINGS
from analysis.accuracy.service import summarize_moves


def prepared_game(pgn='1. e4 e5 2. Nf3 Nc6 3. Bb5 *', metrics=None):
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game.errors:
        raise AssertionError(game.errors)
    board = game.board()
    result = {'start_fen': board.fen(), 'moves': [],
              'performance': {'players': {'white': {'accuracy': 83.2}, 'black': {'accuracy': 75.}}}}
    metrics = metrics or [(90., 4., 100.), (70., 8., 60.), (80., 6., 90.),
                          (60., 10., 70.), (100., 0., 80.)]
    for index, move in enumerate(game.mainline_moves()):
        expected, deviation, actual = metrics[index % len(metrics)]
        result['moves'].append({
            'ply': index+1, 'label': f'{board.fullmove_number}{"." if board.turn else "..."} {board.san(move)}',
            'side': 'white' if board.turn else 'black', 'played': {'move': move.uci()},
            'stage': 'opening' if index < 2 else 'middlegame' if index < 4 else 'endgame',
            'accuracy': actual,
            'maia': {str(rating): {'expected_accuracy': min(100., expected+(rating-1600)/1000),
                                   'absolute_deviation': deviation}
                     for rating in RATINGS},
        })
        board.push(move)
    return result


class AccuracyComparisonTests(unittest.TestCase):
    def test_side_means_and_equal_side_pooling_use_saved_measurements(self):
        analysis = prepared_game()
        result = AccuracyComparison(analysis).compare_positions()
        self.assertEqual(result['rating_scale'], 'lb')
        self.assertEqual(result['conditioning'], 'equal_rating')
        self.assertEqual(result['scope'], {'stage': None, 'from_ply': 1, 'to_ply': 5})
        self.assertEqual(result['players']['white'], {
            'positions': 3, 'forced_moves_excluded': 0, 'average_accuracy': 90.,
            'maia': {'1600': {'expected_accuracy': 90., 'absolute_deviation': 3.33,
                              'actual_minus_expected': 0.}}})
        self.assertEqual(result['players']['black']['maia']['1600'], {
            'expected_accuracy': 65., 'absolute_deviation': 9., 'actual_minus_expected': 0.})
        self.assertEqual(result['comparison']['1600'], {
            'shared_expected_accuracy': 77.5, 'shared_absolute_deviation': 6.17,
            'white_minus_black_expected_accuracy': 25., 'higher_expected_accuracy_side': 'white'})
        self.assertNotEqual(result['comparison']['1600']['shared_expected_accuracy'], 80.)
        self.assertEqual(result['full_game_lichess_accuracy'], {'white': 83.2, 'black': 75.})
        summary = summarize_moves(analysis)
        index = RATINGS.index(1600)
        self.assertEqual(result['comparison']['1600']['shared_expected_accuracy'],
                         summary['expected_accuracy'][index])
        self.assertAlmostEqual(result['comparison']['1600']['shared_absolute_deviation'],
                               summary['absolute_deviation'][index], places=2)

    def test_each_requested_anchor_is_compared_at_the_same_elo_for_both_sides(self):
        result = AccuracyComparison(prepared_game()).compare_positions([600, 2000, 2600])
        self.assertEqual(list(result['comparison']), ['600', '2000', '2600'])
        self.assertEqual(result['players']['black']['maia']['2000']['expected_accuracy'], 65.4)
        self.assertEqual(result['players']['white']['maia']['2000']['expected_accuracy'], 90.27)
        self.assertEqual(result['comparison']['2000']['white_minus_black_expected_accuracy'], 24.87)

    def test_color_swap_reverses_delta_without_changing_shared_metrics(self):
        original = prepared_game()
        mirrored = copy.deepcopy(original)
        mirrored['start_fen'] = chess.Board(original['start_fen']).mirror().fen()
        for row in mirrored['moves']:
            move = chess.Move.from_uci(row['played']['move'])
            row['played']['move'] = chess.Move(chess.square_mirror(move.from_square),
                                               chess.square_mirror(move.to_square)).uci()
            row['side'] = 'black' if row['side'] == 'white' else 'white'
            row.pop('label')
        a = AccuracyComparison(original).compare_positions()['comparison']['1600']
        b = AccuracyComparison(mirrored).compare_positions()['comparison']['1600']
        self.assertEqual(a['shared_expected_accuracy'], b['shared_expected_accuracy'])
        self.assertEqual(a['shared_absolute_deviation'], b['shared_absolute_deviation'])
        self.assertEqual(a['white_minus_black_expected_accuracy'], -b['white_minus_black_expected_accuracy'])
        self.assertEqual(b['higher_expected_accuracy_side'], 'black')

    def test_range_and_stage_select_positions_and_keep_full_game_lichess_separate(self):
        tool = AccuracyComparison(prepared_game())
        selected = tool.compare_positions(stage='middlegame', from_ply=2, to_ply=4)
        self.assertEqual(selected['players']['white']['positions'], 1)
        self.assertEqual(selected['players']['white']['average_accuracy'], 90.)
        self.assertEqual(selected['players']['black']['maia']['1600']['expected_accuracy'], 60.)
        self.assertEqual(selected['full_game_lichess_accuracy']['white'], 83.2)
        rows = tool.by_move(side='black', stage='middlegame', from_ply=2, to_ply=4)['rows']
        self.assertEqual([row['ply'] for row in rows], [4])

    def test_by_move_order_and_pagination_use_expectation_not_played_accuracy(self):
        tool = AccuracyComparison(prepared_game())
        result = tool.by_move(limit=2)
        self.assertEqual(result['positions_available'], 5)
        self.assertEqual(result['returned'], 2)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['next_from_ply'], 3)
        self.assertEqual(result['rows'][0], {'ply': 1, 'move_number': 1, 'label': '1. e4', 'side': 'white',
            'stage': 'opening', 'accuracy': 100., 'expected_accuracy': 90., 'absolute_deviation': 4.,
            'actual_minus_expected': 10.})
        self.assertEqual([row['ply'] for row in tool.by_move(order='hardest', limit=2)['rows']], [4, 2])
        self.assertEqual([row['ply'] for row in tool.by_move(order='easiest', limit=2)['rows']], [5, 1])
        self.assertIsNone(tool.by_move(order='hardest', limit=2)['next_from_ply'])
        self.assertEqual([row['ply'] for row in tool.by_move(from_ply=result['next_from_ply'])['rows']], [3, 4, 5])

    def test_forced_gap_is_excluded_counted_and_does_not_renumber(self):
        analysis = prepared_game('[FEN "8/8/8/8/7r/2k5/8/K7 w - - 0 41"]\n\n'
                                 '41. Ka2 Ra4+ 42. Kb1 Rh4 43. Kc1 *')
        tool = AccuracyComparison(analysis)
        result = tool.by_move(side='white', limit=1)
        self.assertEqual(result['positions_available'], 2)
        self.assertEqual(result['forced_moves_excluded'], {'white': 1, 'black': 0})
        self.assertEqual(result['next_from_ply'], 5)
        self.assertEqual(tool.by_move(side='white')['rows'][1]['label'], '43. Kc1')
        self.assertEqual([row['move_number'] for row in tool.by_move(side='white')['rows']], [41, 43])
        side = tool.compare_positions()['players']['white']
        self.assertEqual(side['positions'], 2)
        self.assertEqual(side['forced_moves_excluded'], 1)
        self.assertEqual(side['maia']['1600']['expected_accuracy'], 95.)
        forced_only = tool.compare_positions(from_ply=3, to_ply=3)
        self.assertEqual(forced_only['players']['white']['forced_moves_excluded'], 1)
        self.assertIsNone(forced_only['players']['white']['average_accuracy'])

    def test_custom_black_start_preserves_game_relative_ply_and_pgn_label(self):
        fen = chess.STARTING_FEN.replace(' w ', ' b ').rsplit(' ', 1)[0]+' 37'
        analysis = prepared_game(f'[FEN "{fen}"]\n\n37... e5 38. Nf3 Nc6 *')
        tool = AccuracyComparison(analysis)
        rows = tool.by_move(side='black')['rows']
        self.assertEqual([(row['ply'], row['move_number'], row['label']) for row in rows],
                         [(1, 37, '37... e5'), (3, 38, '38... Nc6')])
        self.assertEqual([row['move_number'] for row in tool.by_move()['rows']], [37, 38, 38])
        del analysis['moves'][0]['label']
        self.assertEqual(AccuracyComparison(analysis).by_move()['rows'][0]['label'], '37... e5')

    def test_fullmove_pairs_keep_each_decisions_accuracy_and_match_plot_measurements(self):
        analysis = prepared_game()
        rows = AccuracyComparison(analysis).by_move()['rows']
        # At PGN move 1, White's pre-e4 position and Black's pre-e5 position
        # have different saved expectations and actual moves; never average them.
        self.assertEqual([(row['move_number'], row['side'], row['expected_accuracy'], row['accuracy'])
                          for row in rows],
                         [(1, 'white', 90., 100.), (1, 'black', 70., 60.),
                          (2, 'white', 80., 90.), (2, 'black', 60., 70.), (3, 'white', 100., 80.)])
        graph = expected_accuracy_by_move(analysis)
        for side in ('white', 'black'):
            tool_values = [(row['ply'], row['move_number'], row['expected_accuracy'], row['accuracy'])
                           for row in rows if row['side'] == side]
            plot_values = [(row['ply'], row['move_number'], row['expected_accuracy'], row['accuracy'])
                           for row in graph['players'][side]]
            self.assertEqual(tool_values, plot_values)

    def test_empty_one_side_forced_only_and_equal_means_are_explicit(self):
        for analysis in (prepared_game('*'), prepared_game(
                '[FEN "8/8/8/8/r7/2k5/K7/8 w - - 0 42"]\n\n42. Kb1 *')):
            tool = AccuracyComparison(analysis)
            self.assertEqual(tool.by_move()['rows'], [])
            self.assertIsNone(tool.compare_positions()['comparison']['1600']['shared_expected_accuracy'])
            self.assertIsNone(tool.compare_positions()['comparison']['1600']['higher_expected_accuracy_side'])
        one_side = AccuracyComparison(prepared_game('1. e4 *')).compare_positions()
        self.assertEqual(one_side['comparison']['1600']['shared_expected_accuracy'], 90.)
        self.assertIsNone(one_side['comparison']['1600']['white_minus_black_expected_accuracy'])
        self.assertIsNone(one_side['players']['black']['average_accuracy'])
        equal = AccuracyComparison(prepared_game('1. e4 e5 *', [(90., 2., 95.)])).compare_positions()
        self.assertEqual(equal['comparison']['1600']['higher_expected_accuracy_side'], 'equal')
        self.assertEqual(equal['comparison']['1600']['white_minus_black_expected_accuracy'], 0.)

    def test_invalid_arguments_are_rejected_without_coercion(self):
        tool = AccuracyComparison(prepared_game())
        for ratings in ([], [1600]*2, [600, 700, 800, 900, 1000, 1100, 1200],
                        [True], [1600.], ['1600'], [1650], [500], [2700], 1600, '1600'):
            with self.subTest(ratings=ratings), self.assertRaises(ValueError):
                tool.compare_positions(ratings=ratings)
        for kwargs in ({'stage': 'middle'}, {'from_ply': 0}, {'from_ply': True}, {'from_ply': 1.},
                       {'to_ply': False}, {'to_ply': 1.}, {'from_ply': 3, 'to_ply': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tool.compare_positions(**kwargs)
        for kwargs in ({'maia_elo': True}, {'maia_elo': 1650}, {'side': 'both'}, {'order': 'worst'},
                       {'limit': 0}, {'limit': 41}, {'limit': True}, {'limit': 2.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tool.by_move(**kwargs)

    def test_missing_invalid_metrics_and_broken_replay_fail_honestly(self):
        for field in ('expected_accuracy', 'absolute_deviation', 'accuracy'):
            for value in (None, True, '90', -1., 101., float('nan'), float('inf')):
                invalid = prepared_game()
                target = invalid['moves'][0] if field == 'accuracy' else invalid['moves'][0]['maia']['1600']
                target[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'refresh'):
                    AccuracyComparison(invalid).compare_positions()
                with self.assertRaises(ValueError):
                    AccuracyComparison(invalid).by_move()
        for field, value in (('maia', {}), ('maia', {'1600': []}), ('side', 'black'),
                             ('ply', 2), ('stage', None), ('played', {'move': 'e2e5'})):
            invalid = prepared_game()
            invalid['moves'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                AccuracyComparison(invalid).by_move()

    def test_saved_measurements_are_read_without_mutation_raw_evidence_or_cache(self):
        class PreparedOnly(dict):
            def get(self, key, *args):
                if key in ('positions', 'accuracy_curve'):
                    raise AssertionError('Query must use prepared move measurements.')
                return super().get(key, *args)

        analysis = PreparedOnly(prepared_game())
        before = copy.deepcopy(analysis)
        with patch('analysis.cache.storage.JsonCache.get', side_effect=AssertionError('No cache reads')):
            tool = AccuracyComparison(analysis)
            tool.compare_positions([1600, 1800])
            tool.by_move(order='hardest')
        self.assertEqual(analysis, before)


if __name__ == '__main__':
    unittest.main()
