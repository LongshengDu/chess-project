"""CLI rating sources are explicit, complete, and validated before analysis starts."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import chess.pgn

from analysis import elo_convert
from analysis.cache.artifacts import AnalysisStore
from analysis.game.metadata import game_metadata
from analysis.game.summary import compact_summary
from analysis.maia_context import analysis_scale, native_actual_ratings
from coach.coach import main
from tests.coach.fixtures import FakeAnalysisSession


class CoachRatingArgumentsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pgn = self.root / 'game.pgn'
        self.output = self.root / 'output' / 'game-full'
        self.cache = self.root / 'cache'

    def write_game(self, **changes):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        headers = {'Site': 'Chess.com', 'TimeControl': '600+5',
                   'WhiteElo': '1250', 'BlackElo': '1800'}
        headers.update(changes)
        for name, value in headers.items():
            if value is None:
                game.headers.pop(name, None)
            else:
                game.headers[name] = value
        self.pgn.write_text(str(game), encoding='utf-8')

    def assert_rejected(self, *arguments, expected=None):
        with patch('coach.coach.AnalysisSession') as session, \
             patch('coach.coach.AnalysisStore') as store, \
             patch('coach.coach.analyze_game') as analyze, \
             patch('coach.agent_runner.run_coach') as coach, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            status = main([str(self.pgn), '--side', 'white', '--analysis-only',
                           '--cache-dir', str(self.cache), *arguments])
        self.assertEqual(status, 1)
        self.assertIn('--elo', errors.getvalue())
        self.assertIn('--rating-scale', errors.getvalue())
        if expected is not None:
            self.assertIn(expected, errors.getvalue())
        session.assert_not_called()
        store.assert_not_called()
        analyze.assert_not_called()
        coach.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.cache.exists())

    def test_partial_override_is_rejected_before_loading_pgn(self):
        # Even a complete PGN cannot silently supply half an explicit override.
        self.write_game()
        for arguments in (['--elo', '1600'], ['--rating-scale', 'lb']):
            with self.subTest(arguments=arguments), patch('coach.coach.load_game') as load:
                self.assert_rejected(*arguments)
            load.assert_not_called()

    def test_missing_or_invalid_pgn_context_requires_explicit_pair(self):
        cases = [
            {'WhiteElo': None}, {'BlackElo': None}, {'WhiteElo': '?'},
            {'BlackElo': 'unknown'}, {'WhiteElo': '0'}, {'BlackElo': 'nan'},
            {'TimeControl': None}, {'TimeControl': '?'}, {'TimeControl': 'not a clock'},
            {'Site': None}, {'Site': 'Unknown chess site'},
        ]
        for changes in cases:
            with self.subTest(headers=changes):
                self.write_game(**changes)
                self.assert_rejected()

    def run_analysis(self, side, *arguments):
        source_pgn = self.pgn.read_bytes()
        fake = FakeAnalysisSession(cache_directory=self.cache)
        self.addCleanup(fake._temp.cleanup)
        context = MagicMock()
        context.__enter__.return_value = fake
        with patch('coach.coach.AnalysisSession', return_value=context), \
             patch('analysis.accuracy.figures.export_saved_figures'), \
             patch('coach.agent_runner.run_coach') as coach, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            status = main([str(self.pgn), *(['--side', side] if side else []), '--analysis-only',
                           '--cache-dir', str(self.cache), *arguments])
        self.assertEqual(status, 0, errors.getvalue())
        coach.assert_not_called()
        context.__exit__.assert_called_once()
        self.assertEqual(self.pgn.read_bytes(), source_pgn)
        self.assertEqual((self.output / 'game.pgn').read_bytes(), source_pgn)
        return AnalysisStore(self.cache).load(self.output / 'analysis.json')

    def test_analysis_only_needs_no_coaching_side_and_stays_neutral(self):
        self.write_game()
        analysis = self.run_analysis(None)
        self.assertEqual(analysis['coaching'], {})
        public = json.loads((self.output / 'analysis.json').read_text(encoding='utf-8'))
        self.assertEqual(public['coaching'], {})
        self.assertNotIn('side', public)
        first = deepcopy(analysis)
        white = compact_summary(analysis, 'white')
        black = compact_summary(analysis, 'black')
        self.assertEqual(white['coaching'], {'side': 'white'})
        self.assertEqual(black['coaching'], {'side': 'black'})
        self.assertEqual(analysis, first)

    def test_coaching_requires_target_before_input_or_output_changes(self):
        for mode in ([], ['--coach-only']):
            with self.subTest(mode=mode), \
                 patch('coach.coach.load_game') as load, \
                 patch('coach.coach.AnalysisStore') as store, \
                 patch('coach.coach.AnalysisSession') as session, \
                 redirect_stderr(io.StringIO()) as errors:
                status = main([str(self.pgn), *mode])
            self.assertEqual(status, 1)
            self.assertIn('--side', errors.getvalue())
            load.assert_not_called()
            store.assert_not_called()
            session.assert_not_called()
        self.assertFalse(self.output.exists())

    def assert_game_context(self, analysis, expected):
        before = deepcopy(analysis)
        summary = compact_summary(analysis, 'white')
        public = json.loads((self.output / 'analysis.json').read_text(encoding='utf-8'))
        headers = dict(chess.pgn.read_game(io.StringIO(self.pgn.read_text(encoding='utf-8'))).headers)
        self.assertEqual(public['headers'], headers)
        self.assertNotIn('headers', summary)
        for side, rating in expected.items():
            self.assertEqual(summary['game'][side]['elo'], rating)
            self.assertEqual(public['game'][side]['elo'], rating)
        self.assertEqual(public['coaching'], analysis['coaching'])
        self.assertEqual(analysis, before)
        for key in ('player', 'selected_player', 'rating_scale', 'rating_scale_override', 'rating_account_overrides'):
            self.assertNotIn(key, public)
            self.assertNotIn(key, summary)

    def test_default_uses_distinct_pgn_ratings_and_selected_side(self):
        self.write_game()
        for side in ('white', 'black'):
            with self.subTest(side=side):
                analysis = self.run_analysis(side)
                self.assertEqual(analysis['coaching'], {})
                self.assertEqual(analysis['game'], game_metadata(analysis['headers']))
                self.assertEqual(analysis_scale(analysis), 'cr')
                self.assert_game_context(analysis, {'white': 1250, 'black': 1800})
                for name, rating in [('White', 1250), ('Black', 1800)]:
                    self.assertAlmostEqual(native_actual_ratings(analysis)[name],
                                           elo_convert.convert(rating, 'cr', 'lb'), places=8)

    def test_explicit_pair_replaces_both_ratings_and_ignores_invalid_pgn_context(self):
        self.write_game(WhiteElo='?', BlackElo='100', Site='Unknown site', TimeControl=None)
        for scale in ('lb', 'cr', *elo_convert.SCALE_IDENTIFIERS.values()):
            with self.subTest(scale=scale):
                analysis = self.run_analysis('black', '--elo', '1600', '--rating-scale', scale)
                self.assertEqual(analysis['coaching'], {})
                self.assertEqual(analysis['game'], game_metadata(analysis['headers'], actual_elo=1600, rating_scale=scale))
                self.assertEqual(analysis_scale(analysis), elo_convert.normalize_scale(scale))
                native = elo_convert.convert(1600, scale, 'lb')
                self.assertEqual(native_actual_ratings(analysis), {'White': native, 'Black': native})
                self.assertEqual(analysis['headers']['WhiteElo'], '?')
                self.assertEqual(analysis['headers']['BlackElo'], '100')
                self.assert_game_context(analysis, {'white': 1600, 'black': 1600})

    def test_summary_uses_game_metadata_and_never_exposes_headers(self):
        self.write_game(White='Known White', Black='?', Result='1-0', Termination='Normal',
                        Event='Local event', Date='2026.10.08', Opening='King pawn game',
                        ECO='C20', WhiteEloEstimate='2999', BlackEloEstimate='2998')
        analysis = self.run_analysis(None, '--elo', '1600', '--rating-scale', 'lb')
        analysis['headers']['White'] = 'Header must not replace game name'
        analysis['headers']['Result'] = '0-1'
        original = deepcopy(analysis)
        for side in ('white', 'black'):
            summary = compact_summary(analysis, side)
            self.assertNotIn('headers', summary)
            self.assertNotIn('player', summary)
            self.assertEqual(summary['game']['white']['name'], 'Known White')
            self.assertIsNone(summary['game']['black']['name'])
            self.assertEqual(summary['game']['result'], '1-0')
            for field in ('termination', 'event', 'site', 'time_control'):
                self.assertNotIn(field, summary['game'])
            self.assertEqual(summary['game']['opening'], "King's Pawn Game")
            self.assertEqual(summary['game']['eco'], 'C20')
            self.assertNotIn('EloEstimate', json.dumps(summary))
        self.assertEqual(analysis, original)
        self.assertEqual(analysis['headers']['WhiteElo'], '1250')
        self.assertEqual(analysis['headers']['WhiteEloEstimate'], '2999')

    def test_unknown_result_and_metadata_remain_null_in_agent_context(self):
        self.write_game(White='?', Black='?', Result='*')
        analysis = self.run_analysis(None)
        summary = compact_summary(analysis, 'white')
        self.assertIsNone(summary['game']['white']['name'])
        self.assertIsNone(summary['game']['black']['name'])
        self.assertIsNone(summary['game']['result'])
        self.assertNotIn('termination', summary['game'])
        self.assertIsNone(summary['game']['opening'])


if __name__ == '__main__':
    unittest.main()
