"""Batch cache preflight uses temporary evidence and never starts engines."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess.pgn

from analysis.cache.artifacts import AnalysisStore
from analysis.game.pipeline import analyze_game
from tests.analysis import analyze_pgn_batch as batch
from tests.coach.fixtures import FakeAnalysisSession


class BatchPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = self.directory / 'cache'
        self.paths, self.analyses = [], []
        self.store = AnalysisStore(self.cache)
        for index, moves in enumerate(('1. e4 e5 *', '1. d4 d5 *')):
            text = '[Site "https://lichess.org"]\n[TimeControl "300+0"]\n[WhiteElo "1400"]\n[BlackElo "1600"]\n\n' + moves
            path = self.directory / f'game{index}.pgn'
            path.write_text(text, encoding='utf-8')
            game = chess.pgn.read_game(io.StringIO(text))
            session = FakeAnalysisSession(game.board().fen(), cache_directory=self.cache)
            self.addCleanup(session._temp.cleanup)
            analysis = analyze_game(game, session, progress=lambda _: None)
            self.store.save(self.output(path) / 'analysis.json', analysis)
            self.paths.append(path)
            self.analyses.append(analysis)
        self.timing = self.directory / 'timings' / 'run.json'
        configuration = patch.dict(batch.CONFIG['ANALYSIS'], {'CACHE_DIR': self.cache})
        configuration.start()
        self.addCleanup(configuration.stop)
        for target in ('tests.analysis.analyze_pgn_batch.AnalysisSession', 'engine.runtime.EngineRuntime.__init__'):
            guard = patch(target, side_effect=AssertionError('No engine startup'))
            guard.start()
            self.addCleanup(guard.stop)

    @staticmethod
    def output(path):
        return path.parent / 'output' / f'{path.stem}-full'

    def snapshot(self):
        return {path: path.read_bytes() for path in self.directory.rglob('*') if path.is_file()}

    def test_check_cache_accepts_existing_outputs_and_writes_nothing(self):
        before = self.snapshot()
        result = batch.run(self.paths, self.timing, check_cache=True)
        self.assertEqual(result['checked_games'], 2)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.timing.parent.exists())

    def test_later_missing_policy_stops_before_any_output_or_timing_change(self):
        analysis = self.analyses[1]
        del analysis['positions'][1]['maia']['maia_kdd_600']
        del analysis['position_references'][1]['maia']['maia_kdd_600']
        self.store.save(self.output(self.paths[1]) / 'analysis.json', analysis)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'game1.pgn: Missing saved Maia rating pair 600/600'):
            batch.run(self.paths, self.timing, rebuild_from_cache=True, replace_output=True)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.timing.parent.exists())

    def test_rebuild_from_cache_reuses_preflight_sessions_and_preserves_other_files(self):
        preserved = self.output(self.paths[0]) / 'coaching.md'
        preserved.write_text('Accepted report', encoding='utf-8')
        cached = batch.CachedAnalysisSession
        with patch.object(batch, 'CachedAnalysisSession', wraps=cached) as sessions, \
                patch('analysis.accuracy.figures.export_saved_figures'):
            result = batch.run(self.paths, self.timing, rebuild_from_cache=True, replace_output=True)
        self.assertEqual(sessions.call_count, len(self.paths))
        self.assertEqual(len(result['games']), 2)
        self.assertTrue(result['engines_closed'])
        self.assertEqual(result['engine_load_seconds'], 0)
        self.assertEqual(preserved.read_text(encoding='utf-8'), 'Accepted report')
        for entry in result['games']:
            self.assertEqual(entry['engine_statistics']['stockfish_calls'], 0)
            self.assertEqual(entry['engine_statistics']['maia_batches'], 0)

    def test_existing_outputs_require_explicit_replacement(self):
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            batch.run(self.paths, self.timing, rebuild_from_cache=True)
        self.assertEqual(before, self.snapshot())

    def test_multiple_games_are_rejected_before_cache_or_outputs(self):
        path = self.directory / 'multiple.pgn'
        path.write_text('1. e4 *\n\n1. d4 *\n', encoding='utf-8')
        before = self.snapshot()
        with patch.object(batch, 'CachedAnalysisSession', side_effect=AssertionError('No cache read')):
            with self.assertRaisesRegex(ValueError, 'one game per PGN'):
                batch.run([path], self.timing, check_cache=True)
        self.assertEqual(before, self.snapshot())

    def test_mixed_starting_positions_are_rejected_before_cache_or_outputs(self):
        path = self.directory / 'custom.pgn'
        path.write_text('[FEN "8/8/8/8/r7/2k5/K7/8 w - - 0 42"]\n\n42. Kb1 *', encoding='utf-8')
        before = self.snapshot()
        with patch.object(batch, 'CachedAnalysisSession', side_effect=AssertionError('No cache read')):
            with self.assertRaisesRegex(ValueError, 'same starting position'):
                batch.run([self.paths[0], path], self.timing, check_cache=True)
        self.assertEqual(before, self.snapshot())

    def test_invalid_supplied_rating_context_fails_before_cache_or_outputs(self):
        path = self.directory / 'invalid-context.pgn'
        path.write_text('[Site "unknown.example"]\n[TimeControl "300+0"]\n\n1. e4 *', encoding='utf-8')
        before = self.snapshot()
        with patch.object(batch, 'CachedAnalysisSession', side_effect=AssertionError('No cache read')):
            with self.assertRaises(ValueError):
                batch.run([path], self.timing, check_cache=True)
        self.assertEqual(before, self.snapshot())


if __name__ == '__main__':
    unittest.main()
