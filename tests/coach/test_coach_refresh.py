"""Coach-only scale/account revisions reuse game evidence; all session/API mocked."""
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
from analysis.cache.storage import write_json
from analysis.game.pipeline import analyze_game
from analysis.cache.artifacts import AnalysisStore
from analysis.game.metadata import game_metadata
from analysis.maia_context import analysis_scale, native_actual_ratings
from coach.coach import main
from tests.coach.fixtures import FakeAnalysisSession


class CoachScaleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pgn = self.root/'revised.pgn'
        self.output = self.root/'output'/'revised-full'
        self.output.mkdir(parents=True)

    def write_game(self, **headers):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        game.headers.update(Site='Chess.com', TimeControl='300+0', WhiteElo='1400', BlackElo='1300', **headers)
        self.pgn.write_text(str(game), encoding='utf-8')
        return game

    def test_cache_refresh_is_explicit_for_analysis_and_analysis_plus_coach(self):
        game = self.write_game()
        analysis = {'start_fen': game.board().fen(), 'moves': [], 'agent_run': {'usage': {}}}
        for mode in ([], ['--analysis-only']):
            for refresh in (False, True):
                with self.subTest(mode=mode, refresh=refresh), \
                     patch('coach.coach.AnalysisSession') as constructor, \
                     patch('coach.coach.analyze_game', return_value=analysis) as analyze, \
                     patch('coach.coach.AnalysisStore') as store, \
                     patch('coach.agent_runner.run_coach') as coach, \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    status = main([str(self.pgn), '--side', 'white', '--elo', '1500', '--rating-scale', 'cb', *mode,
                                   *(['--refresh-cache'] if refresh else [])])
                self.assertEqual(status, 0)
                self.assertEqual(constructor.call_args.kwargs['refresh_cache'], refresh)
                analyze.assert_called_once()
                store.return_value.save.assert_called_once()
                self.assertEqual(coach.call_count, int(not mode))

    def test_refresh_cache_and_coach_only_are_rejected_before_io_or_engines(self):
        with patch('coach.coach.load_game') as load, \
             patch('coach.coach.AnalysisSession') as session, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            status = main([str(self.pgn), '--side', 'white', '--elo', '1500',
                           '--rating-scale', 'cb', '--coach-only', '--refresh-cache'])
        self.assertEqual(status, 1)
        self.assertIn('--refresh-cache requires game analysis', errors.getvalue())
        load.assert_not_called()
        session.assert_not_called()

    def test_coach_only_uses_revised_pgn_and_replaces_game_and_coaching_context(self):
        original_game = self.write_game()
        seed_session = FakeAnalysisSession(original_game.board().fen())
        self.addCleanup(seed_session._temp.cleanup)
        analysis = analyze_game(original_game, seed_session, actual_elo=1499,
                                rating_scale='lb', progress=lambda _: None)
        write_json(self.output/'analysis.json', analysis)
        saved_moves = deepcopy(analysis['moves'])
        shared = deepcopy(analysis['accuracy_curve'])
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        game.headers.update(Site='https://lichess.org/revised', TimeControl='600+0',
                            WhiteElo='1800', BlackElo='1750', White='Updated player', Event='Current metadata')
        self.pgn.write_text(str(game), encoding='utf-8')
        session = MagicMock()
        session.__enter__.return_value = session
        received = []

        def coach_stub(current, opened, output, *, side, **_kwargs):
            self.assertIs(opened, session)
            self.assertEqual(output, self.output)
            received.append(side)
            current['agent_run'] = {'usage': {'total_tokens': 0}}

        def run(side, *extra):
            with patch('coach.coach.AnalysisSession', return_value=session) as constructor, \
                 patch('coach.coach.analyze_game', side_effect=AssertionError('Coach-only must not reanalyze the game')), \
                 patch('analysis.accuracy.figures.export_saved_figures') as figures, \
                 patch('coach.agent_runner.run_coach', side_effect=coach_stub) as coach, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main([str(self.pgn), '--side', side, '--coach-only',
                               '--cache-dir', str(seed_session.cache.directory), *extra])
            self.assertEqual(status, 0)
            constructor.assert_called_once()
            coach.assert_called_once()
            figures.assert_called_once()
            self.assertEqual(figures.call_args.kwargs.get('output_dir', figures.call_args.args[1]),
                             self.output)
            public = json.loads((self.output/'analysis.json').read_text(encoding='utf-8'))
            for key in ('schema_version', 'game_id', 'rating_account_overrides', 'analysis_execution',
                        'selected_player', 'rating_scale', 'rating_scale_override'):
                self.assertNotIn(key, public)
            self.assertNotIn('positions', public)
            self.assertEqual(public['moves'], saved_moves)
            saved = AnalysisStore(seed_session.cache.directory).load(self.output/'analysis.json')
            self.assertEqual(saved['headers'], dict(game.headers))
            self.assertEqual(saved['performance']['players']['white']['name'], 'Updated player')
            self.assertEqual(saved['moves'], saved_moves)
            self.assertNotIn('positions', saved)
            self.assertEqual(saved['accuracy_curve'], shared)
            self.assertEqual((self.output/'game.pgn').read_text(encoding='utf-8'), self.pgn.read_text(encoding='utf-8'))
            return saved

        inferred = run('black')
        self.assertEqual(inferred['game'], game_metadata(dict(game.headers)))
        self.assertEqual(inferred['coaching'], {})
        self.assertEqual(analysis_scale(inferred), 'lr')
        self.assertAlmostEqual(native_actual_ratings(inferred)['White'],
                               elo_convert.convert(1800., 'lr', 'lb'), places=8)
        self.assertAlmostEqual(native_actual_ratings(inferred)['Black'],
                               elo_convert.convert(1750., 'lr', 'lb'), places=8)
        overridden = run('white', '--elo', '1650', '--rating-scale', 'cr')
        self.assertEqual(overridden['game'], game_metadata(dict(game.headers), actual_elo=1650, rating_scale='cr'))
        self.assertEqual(overridden['coaching'], {})
        self.assertEqual(analysis_scale(overridden), 'cr')
        for side in ('White', 'Black'):
            self.assertAlmostEqual(native_actual_ratings(overridden)[side],
                                   elo_convert.convert(1650., 'cr', 'lb'), places=8)
        restored = run('black')
        self.assertEqual(restored['game'], inferred['game'])
        self.assertEqual(native_actual_ratings(restored), native_actual_ratings(inferred))
        side_neutral_bytes = (self.output/'analysis.json').read_bytes()
        other_side = run('white')
        self.assertEqual(other_side['game'], restored['game'])
        self.assertEqual((self.output/'analysis.json').read_bytes(), side_neutral_bytes)
        self.assertEqual(received, ['black', 'white', 'black', 'white'])
        self.assertEqual(session.__exit__.call_count, 4)

    def test_unsupported_scale_is_rejected_before_engines_or_game_analysis(self):
        game = self.write_game()
        game.headers.update(Site='https://lichess.org/fixture', TimeControl='60+0')
        self.pgn.write_text(str(game), encoding='utf-8')
        with patch('coach.coach.AnalysisSession') as session, patch('coach.coach.analyze_game') as analyze, \
             patch('coach.agent_runner.run_coach') as coach, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            status = main([str(self.pgn), '--side', 'white', '--analysis-only'])
        self.assertEqual(status, 1)
        self.assertIn('bullet', errors.getvalue())
        session.assert_not_called()
        analyze.assert_not_called()
        coach.assert_not_called()

    def test_chesscom_rating_floor_is_rejected_before_engines_for_headers_and_cli(self):
        for source in ('header', 'selected'):
            with self.subTest(source=source):
                game = self.write_game()
                if source == 'header':
                    game.headers['BlackElo'] = '100'
                    self.pgn.write_text(str(game), encoding='utf-8')
                override = ['--elo', '100', '--rating-scale', 'cb'] if source == 'selected' else []
                with patch('coach.coach.AnalysisSession') as session, patch('coach.coach.analyze_game') as analyze, \
                     patch('coach.agent_runner.run_coach') as coach, \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                    status = main([str(self.pgn), '--side', 'white', '--analysis-only', *override])
                self.assertEqual(status, 1)
                self.assertIn('lower asymptote', errors.getvalue())
                session.assert_not_called()
                analyze.assert_not_called()
                coach.assert_not_called()


if __name__ == '__main__':
    unittest.main()
