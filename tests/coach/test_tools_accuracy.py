"""Saved accuracy evidence reaches Codex without engine work or checked-line credit."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chess.pgn

from analysis.accuracy.service import summarize_moves
from analysis.game.pipeline import analyze_game
from analysis.game.summary import compact_summary
from coach.agent_budget import CoachingLimitError, RunBudget
from coach.agent_codex import CodexChessSession, tool_specs
from coach.agent_progress import CoachProgress
from coach.tools_chess import ChessTools
from tests.coach.fixtures import FakeAnalysisSession


class AccuracyCoachToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = FakeAnalysisSession()
        self.addCleanup(self.session._temp.cleanup)
        game = chess.pgn.read_game(io.StringIO(
            '[WhiteElo "1430"]\n[BlackElo "2100"]\n\n'
            '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *'))
        self.analysis = analyze_game(game, self.session, progress=lambda _: None)
        self.analysis.pop('positions')
        # Deliberately simple, per-game saved measurements make side/range
        # comparisons independently calculable; no labels or game assets fit them.
        for row in self.analysis['moves']:
            ply = row['ply']
            white = row['side'] == 'white'
            row['stage'] = ('opening', 'middlegame', 'endgame')[(ply-1)//2]
            row['accuracy'] = 100-10*ply
            for rating, record in row['maia'].items():
                record['expected_accuracy'] = (80 if white else 70)+ply+(int(rating)-1600)/500
                record['absolute_deviation'] = 4. if white else 8.
        self.analysis['accuracy_curve'] = summarize_moves(self.analysis)
        self.before = copy.deepcopy(self.analysis)
        for method in ('human', 'human_pairs', 'sf', 'initial_analysis'):
            setattr(self.session, method, Mock(side_effect=AssertionError('Saved evidence must not run engines')))
        self.session.cache.get = Mock(side_effect=AssertionError('No raw cache reads'))
        self.session.cache.put = Mock(side_effect=AssertionError('No raw cache writes'))
        self.messages = []

    def library(self, max_calls=20):
        return ChessTools(self.analysis, self.session, self.root, side='white', max_calls=max_calls,
                          progress=CoachProgress(self.messages.append))

    def assert_prepared_only(self, value):
        if isinstance(value, dict):
            self.assertFalse({'policy', 'cp_vec', 'candidate_moves'} & set(value))
            if 'positions' in value:
                self.assertIs(type(value['positions']), int)  # A count, never raw records.
            self.assertFalse(any(str(key).startswith('maia_kdd_') for key in value))
            for item in value.values():
                self.assert_prepared_only(item)
        elif isinstance(value, list):
            for item in value:
                self.assert_prepared_only(item)

    def test_registered_typed_tools_are_read_only_and_not_chess_investigations(self):
        library = self.library()
        specs = {spec['name']: spec for spec in tool_specs(library.tools)}
        self.assertIn('compare_position_difficulty', specs)
        self.assertIn('get_accuracy_by_move', specs)
        compared = library.tools_by_name['compare_position_difficulty'](ratings=[1600, 2200])
        moves = library.tools_by_name['get_accuracy_by_move'](maia_elo=1600, limit=3)
        for result in (compared, moves):
            self.assertIn('result_id', result)
            self.assert_prepared_only(result)
            json.loads(str(result))
        self.assertEqual(library.investigated, set())
        self.assertEqual(library.diagrams.paths, set())
        self.assertEqual(self.analysis, self.before)
        self.assertEqual(len(library.invocations), 2)

    def test_same_rating_comparison_uses_each_sides_saved_measurements(self):
        library = self.library()
        compared = library.tools_by_name['compare_position_difficulty'](ratings=[1600])
        white, black = (compared['players'][side] for side in ('white', 'black'))
        self.assertEqual((white['positions'], black['positions']), (3, 3))
        self.assertEqual((white['average_accuracy'], black['average_accuracy']), (70., 60.))
        self.assertEqual(white['maia']['1600'], {
            'expected_accuracy': 83., 'absolute_deviation': 4., 'actual_minus_expected': -13.})
        self.assertEqual(black['maia']['1600'], {
            'expected_accuracy': 74., 'absolute_deviation': 8., 'actual_minus_expected': -14.})
        self.assertEqual(compared['comparison']['1600'], {
            'shared_expected_accuracy': 78.5, 'shared_absolute_deviation': 6.,
            'white_minus_black_expected_accuracy': 9., 'higher_expected_accuracy_side': 'white'})
        # Account ratings differ, but this comparison holds Maia skill constant.
        self.assertEqual(compared['conditioning'], 'equal_rating')
        self.assertEqual(compared['rating_scale'], 'lb')
        middlegame = library.tools_by_name['compare_position_difficulty'](ratings=[1600], stage='middlegame')
        self.assertEqual([middlegame['players'][side]['positions'] for side in ('white', 'black')], [1, 1])
        self.assertEqual(middlegame['full_game_lichess_accuracy'], compared['full_game_lichess_accuracy'])

    def test_by_move_selection_orders_difficulty_not_played_accuracy_and_paginates(self):
        tool = self.library().tools_by_name['get_accuracy_by_move']
        first = tool(maia_elo=1600, limit=2)
        self.assertEqual([row['ply'] for row in first['rows']], [1, 2])
        self.assertEqual([(row['move_number'], row['side'], row['accuracy'], row['expected_accuracy'])
                          for row in first['rows']],
                         [(1, 'white', 90., 81.), (1, 'black', 80., 72.)])
        self.assertEqual((first['positions_available'], first['returned']), (6, 2))
        self.assertTrue(first['truncated'])
        self.assertEqual(first['next_from_ply'], 3)
        rest = tool(maia_elo=1600, from_ply=first['next_from_ply'])
        self.assertEqual([row['ply'] for row in rest['rows']], [3, 4, 5, 6])
        self.assertEqual([row['move_number'] for row in rest['rows']], [2, 2, 3, 3])
        self.assertFalse(rest['truncated'])
        self.assertIsNone(rest['next_from_ply'])
        hardest = tool(maia_elo=1600, order='hardest', limit=2)
        easiest = tool(maia_elo=1600, order='easiest', limit=2)
        self.assertEqual([row['ply'] for row in hardest['rows']], [2, 4])
        self.assertEqual([row['ply'] for row in easiest['rows']], [5, 3])
        self.assertIsNone(hardest['next_from_ply'])
        selected = tool(maia_elo=1600, side='white', stage='middlegame', from_ply=2, to_ply=4)
        self.assertEqual([row['ply'] for row in selected['rows']], [3])
        self.assertEqual(selected['rows'][0]['actual_minus_expected'], -13.)

    def test_batch_rejects_invalid_types_and_limits_before_any_child_runs(self):
        library = self.library()
        good = {'tool': 'compare_position_difficulty', 'arguments': {}}
        for arguments in ({'maia_elo': '1600'}, {'limit': 0}, {'limit': 41}, {'order': 'unknown'}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                library.call('investigate_batch', {'requests': [good,
                    {'tool': 'get_accuracy_by_move', 'arguments': arguments}]})
            self.assertEqual(library.invocations, [])

    def test_codex_can_batch_both_tools_and_repeat_uses_saved_tool_result(self):
        library = self.library()
        codex = CodexChessSession(library, RunBudget())
        arguments = {'ratings': [1600], 'stage': 'middlegame'}
        response = codex.handle_request('item/tool/call', {
            'tool': 'investigate_batch', 'arguments': {'requests': [
                {'tool': 'compare_position_difficulty', 'arguments': arguments},
                {'tool': 'get_accuracy_by_move', 'arguments': {'maia_elo': 1600, 'limit': 2}},
            ]}})
        self.assertTrue(response['success'], response)
        results = json.loads(response['contentItems'][0]['text'])['results']
        self.assertEqual([result['index'] for result in results], [0, 1])
        self.assertTrue(all('result' in result for result in results))
        with patch.object(library, 'compare_position_difficulty', side_effect=AssertionError('Already cached')):
            repeated = library.call('compare_position_difficulty', arguments)
        self.assertEqual(repeated['result_id'], results[0]['result']['result_id'])
        self.assertTrue(any('Reused saved evidence' in message for message in self.messages))
        events = [json.loads(line) for line in library.trace.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([event['tool'] for event in events],
                         ['compare_position_difficulty', 'get_accuracy_by_move', 'compare_position_difficulty'])
        self.assertTrue(all(event['status'] == 'ok' and event['response_chars'] > 0 for event in events))
        self.assertEqual(library.investigated, set())
        self.assertEqual(self.analysis, self.before)

    def test_batch_budget_is_enforced_before_any_saved_queries_run(self):
        library = self.library(max_calls=1)
        with self.assertRaises(CoachingLimitError):
            library.call('investigate_batch', {'requests': [
                {'tool': 'compare_position_difficulty', 'arguments': {}},
                {'tool': 'get_accuracy_by_move', 'arguments': {}},
            ]})
        self.assertEqual(library.invocations, [])
        library.tools_by_name['get_accuracy_by_move']()
        with self.assertRaises(CoachingLimitError):
            library.tools_by_name['compare_position_difficulty']()
        self.assertEqual(len(library.invocations), 1)
        self.assertEqual(library.investigated, set())

    def test_invalid_requests_are_returned_to_codex_without_engine_work(self):
        library = self.library()
        codex = CodexChessSession(library, RunBudget())
        cases = [
            ('compare_position_difficulty', {'ratings': [1650]}),
            ('compare_position_difficulty', {'ratings': [500]}),
            ('compare_position_difficulty', {'ratings': [2700]}),
            ('compare_position_difficulty', {'ratings': []}),
            ('compare_position_difficulty', {'ratings': ['1600']}),
            ('compare_position_difficulty', {'stage': 'unknown'}),
            ('compare_position_difficulty', {'from_ply': 0}),
            ('compare_position_difficulty', {'from_ply': 4, 'to_ply': 2}),
            ('get_accuracy_by_move', {'maia_elo': 1650}),
            ('get_accuracy_by_move', {'maia_elo': True}),
            ('get_accuracy_by_move', {'side': 'both'}),
            ('get_accuracy_by_move', {'order': 'unknown'}),
            ('get_accuracy_by_move', {'limit': 0}),
            ('get_accuracy_by_move', {'limit': 100000}),
            ('get_accuracy_by_move', {'limit': 2.5}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                response = codex.handle_request('item/tool/call', {'tool': name, 'arguments': arguments})
                self.assertFalse(response['success'], response)
                self.assertIn('error', json.loads(response['contentItems'][0]['text']))
        self.assertEqual(library.investigated, set())
        self.assertEqual(self.analysis, self.before)

    def test_compact_summary_supplies_small_initial_same_rating_comparison(self):
        summary = compact_summary(self.analysis, 'white')
        comparison = summary['position_difficulty']
        self.assertIsNotNone(comparison)
        self.assertEqual(set(comparison['comparison']), {'1400'})
        for side in ('white', 'black'):
            self.assertEqual(set(comparison['players'][side]['maia']), {'1400'})
        self.assert_prepared_only(comparison)
        self.assertEqual(self.analysis, self.before)
        self.assertLess(len(json.dumps(comparison)), 5000)


if __name__ == '__main__':
    unittest.main()
