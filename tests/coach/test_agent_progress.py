"""Offline console progress: no extra model turns, raw reasoning or leaked threads."""
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import chess.pgn

from coach.agent_runner import run_coach
from coach.tools_chess import ChessTools
from analysis.game.pipeline import analyze_game
from coach.agent_progress import CoachProgress, print_progress
from coach.agent_runner import CoachingRequest
from tests.coach.fixtures import FakeEngines, ScriptedCodex


class ProgressTests(unittest.TestCase):
    def test_public_commentary_is_bounded_and_private_fields_are_not_printed(self):
        messages = []
        progress = CoachProgress(messages.append)
        progress.commentary('<think>PRIVATE</think>')
        progress.commentary('<analysis>PRIVATE')
        self.assertEqual(messages, [])
        progress.commentary('Checking the earlier pawn break.')
        progress.commentary('Checking\x1b[2J the defense.\n' + 'x'*900)
        self.assertEqual(len(messages), 2)
        self.assertIn('earlier pawn break', messages[0])
        self.assertNotIn('PRIVATE', str(messages))
        self.assertNotIn('\x1b', str(messages))
        self.assertLessEqual(len(messages[1]), 615)

    def test_wait_updates_are_throttled_and_stop_after_close(self):
        messages = []
        with patch('coach.agent_progress.time.monotonic', return_value=0) as clock:
            progress = CoachProgress(messages.append, interval=15)
            progress.stage('Waiting for the coach.')
            clock.return_value = 14
            progress.pulse()
            self.assertEqual(len(messages), 1)
            clock.return_value = 15
            progress.pulse()
            self.assertEqual(len(messages), 2)
            self.assertIn('still in progress', messages[-1])
            progress.close()
            clock.return_value = 100
            progress.pulse()
            self.assertEqual(len(messages), 2)

    def test_output_failure_does_not_break_coaching_and_thread_closes(self):
        def broken(message):
            raise BrokenPipeError()
        progress = CoachProgress(broken)
        progress.start()
        progress.stage('Checking evidence.')
        self.assertIsNone(progress.sink)
        progress.close()
        self.assertFalse(progress.thread.is_alive())
        with patch('builtins.print') as output:
            print_progress('status')
        output.assert_called_once_with('status', flush=True)

    def test_tools_and_scripted_agent_show_progress_without_extra_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            engines = FakeEngines()
            self.addCleanup(engines._temp.cleanup)
            game = chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 Nc6 *'))
            analysis = analyze_game(game, engines, 'white', 1400, progress=lambda _: None)
            model, messages = ScriptedCodex(), []
            report = model.run(analysis, engines, Path(tmp), progress=messages.append)
            text = '\n'.join(messages)
            self.assertIn('Preparing the critical decisions', text)
            self.assertIn('Comparing likely human replies', text)
            self.assertIn('Maia continuations at 1400/1600/1800/2000 Elo', text)
            self.assertIn('Report saved: coaching.md', text)
            self.assertEqual(model.index, 6)
            self.assertNotIn('PRIVATE_REASONING_SENTINEL', text)
            self.assertNotIn('[Coach ', report)
            self.assertFalse(any(t.name == 'coach-progress' for t in threading.enumerate()))

    def test_failed_investigation_is_visible_and_silent_sink_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            engines = FakeEngines()
            self.addCleanup(engines._temp.cleanup)
            game = chess.pgn.read_game(io.StringIO('1. e4 *'))
            analysis = analyze_game(game, engines, 'white', 1400, progress=lambda _: None)
            messages = []
            library = ChessTools(analysis, engines, Path(tmp), progress=CoachProgress(messages.append))
            with self.assertRaises(ValueError):
                library.call('get_position', {'ply': 999})
            self.assertIn('could not complete', messages[-1])
            library.call('get_position', {'ply': 1})
            library.call('get_position', {'ply': 1})
            self.assertIn('Reused saved evidence', messages[-1])
            silent = CoachProgress(None)
            silent.start()
            silent.stage('Silent')
            silent.close()
            self.assertIsNone(silent.thread)

    def test_preparation_failure_stops_progress_without_contacting_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            engines = FakeEngines()
            self.addCleanup(engines._temp.cleanup)
            game = chess.pgn.read_game(io.StringIO('1. e4 *'))
            analysis = analyze_game(game, engines, 'white', 1400, progress=lambda _: None)
            def fail(library):
                raise ValueError('Local preparation failed')
            model, messages = ScriptedCodex(), []
            with self.assertRaisesRegex(ValueError, 'Local preparation failed'):
                model.run(analysis, engines, Path(tmp), progress=messages.append,
                          request=CoachingRequest(prepare_task=fail))
            self.assertEqual(model.index, 0)
            self.assertIn('stopped before completion', messages[-1])
            self.assertFalse(any(t.name == 'coach-progress' for t in threading.enumerate()))


if __name__ == '__main__':
    unittest.main()
