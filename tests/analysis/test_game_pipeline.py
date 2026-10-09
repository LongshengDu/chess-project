"""The web and coach consume one full-position, accuracy curve analysis contract."""
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chess
import chess.pgn

from analysis.game.cancellation import AnalysisCancelled
from analysis.game.pipeline import analyze_game
from analysis.cache.artifacts import AnalysisStore
from analysis.cache.session import CachedAnalysisSession
from analysis.session import AnalysisSession, Limits
from engine.stockfish_pool import StockfishPool
from tests.coach.fixtures import FakeAnalysisSession


class GamePipelineTests(unittest.TestCase):
    def session(self, game):
        session = FakeAnalysisSession(game.board().fen())
        self.addCleanup(session._temp.cleanup)
        return session

    def test_web_and_coach_produce_the_same_position_and_accuracy_evidence(self):
        game = chess.pgn.read_game(io.StringIO('[WhiteElo "1400"]\n[BlackElo "1700"]\n\n1. e4 e5 2. Nf3 Nc6 *'))
        web = analyze_game(game, self.session(game), progress=lambda _: None)
        coach = analyze_game(game, self.session(game), progress=lambda _: None)
        self.assertEqual(web['positions'], coach['positions'])
        self.assertEqual(web['moves'], coach['moves'])
        self.assertEqual(web['accuracy_curve'], coach['accuracy_curve'])
        self.assertEqual(web['performance'], coach['performance'])
        self.assertEqual(len(web['positions']), len(web['moves'])+1)
        self.assertEqual(web['coaching'], {})
        self.assertEqual(coach['coaching'], {})
        self.assertEqual(coach['game']['white'], {'name': None, 'elo': 1400})
        self.assertEqual(coach['game']['black'], {'name': None, 'elo': 1700})
        self.assertEqual(coach['game']['rating_scale'], 'lichess_blitz')
        self.assertEqual(web['game'], coach['game'])
        self.assertTrue(all('flags' in move for move in web['moves']))

    def test_configuration_is_reported_before_inference_for_each_game(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        session = self.session(game)
        session.engines.maia_model = 'fixture-custom-model'
        session.engines.signature['device'] = 'cuda'
        session.engines.analysis_pool = SimpleNamespace(threads_per_worker=3, hash_mb=96)
        session.analysis_workers = 2
        session.limits = Limits(depth=12, verify_ms=750, max_ms=1500)
        original = session.human_pair_batches
        messages = []

        def infer(requests):
            self.assertIn('model=fixture-custom-model, device=cuda', messages[0])
            for setting in ('workers=2', 'threads_per_worker=3', 'hash_mb_per_worker=96',
                            'depth_ceiling=12', 'search_seconds=0.75', 'max_seconds=1.5'):
                self.assertIn(setting, messages[1])
            return original(requests)

        with patch.object(session, 'human_pair_batches', side_effect=infer):
            for _ in range(2):
                messages.clear()
                analyze_game(game, session, progress=messages.append)

    def test_position_progress_uses_recorded_depth_and_available_search_time(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 *'))
        cases = (({'elapsed_seconds': 1.25, 'phases': [{'wall_ms': 10}]}, '1.250s'),
                 ({'phases': [{'wall_ms': 125}, {'wall_ms': 250}]}, '0.375s'),
                 ({'budget_seconds': 10}, 'unavailable'))
        for timing, expected in cases:
            with self.subTest(timing=timing):
                session = self.session(game)
                original = session.initial_analysis

                def scan(*args):
                    result = original(*args)
                    result['search'].update(depth=13, target_depth=24, **timing)
                    return result

                messages = []
                with patch.object(session, 'initial_analysis', side_effect=scan):
                    analysis = analyze_game(game, session, progress=messages.append)
                self.assertIn(f'Analyzed 1/2: 1. e4 (depth 13, search time {expected})', messages)

                # Rebuilding from cache reports original timing, never new search work.
                AnalysisStore(session.cache.directory).save(session.cache.directory / 'analysis.json', analysis)
                cached = CachedAnalysisSession(game, session.cache.directory)
                replayed = []
                with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('No engines')):
                    analyze_game(game, cached, progress=replayed.append)
                self.assertEqual(replayed[0], 'Analysis: rebuilding from cache; Maia and Stockfish disabled.')
                self.assertEqual([line for line in messages if line.startswith('Analyzed ')],
                                 [line for line in replayed if line.startswith('Analyzed ')])

    def test_out_of_order_callbacks_publish_complete_maps_and_preserve_saved_order(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        session = self.session(game)
        def reverse_scans(jobs, *, cancel=None):
            for index in reversed(range(len(jobs))):
                yield index, session.initial_analysis(*jobs[index])
        session.analyze_positions = reverse_scans
        positions = []
        result = analyze_game(game, session, progress=lambda _: None,
                              on_position=lambda index, value: positions.append((index, value)))
        self.assertEqual([index for index, _ in positions], [2, 1, 0])
        board = game.board()
        for index, position in enumerate(result['positions']):
            self.assertEqual(position['ply'], index)
            self.assertEqual(position['fen'], board.fen(en_passant='fen'))
            self.assertEqual(set(position['maia']), {f'maia_kdd_{r}' for r in range(600,2601,100)})
            legal = {move.uci() for move in board.legal_moves}
            self.assertEqual(set(position['stockfish']['cp_vec']), legal)
            self.assertEqual(set(position['stockfish']['root_move_depth_vec']), legal)
            for prediction in position['maia'].values():
                self.assertEqual(set(prediction['policy']), legal)
                self.assertIn('value', prediction)
            if index < len(result['moves']):
                board.push_uci(result['moves'][index]['played']['move'])
        self.assertIn(' e3 ', result['positions'][1]['fen'])
        self.assertEqual([history for history, _, _ in session.pair_calls], [[], ['e2e4'], ['e2e4','e7e5']])

    def test_final_checkmate_has_terminal_evidence_and_no_extra_accuracy_observation(self):
        game = chess.pgn.read_game(io.StringIO('1. f3 e5 2. g4 Qh4# 0-1'))
        session = self.session(game)
        result = analyze_game(game, session, progress=lambda _: None)
        self.assertEqual(len(result['positions']), 5)
        final = result['positions'][-1]
        self.assertEqual(final['stockfish']['terminal_cp'], -10000)
        self.assertEqual(final['stockfish']['cp_vec'], {})
        self.assertEqual(final['stockfish']['mate_vec'], {'': 0})
        self.assertTrue(final['stockfish']['is_checkmate'])
        self.assertTrue(all(value == {'policy': {}, 'value': 0.} for value in final['maia'].values()))
        self.assertEqual(len(session.pair_calls), 4)
        self.assertEqual(result['performance']['players']['white']['moves_total'], 2)
        self.assertEqual(result['performance']['players']['black']['moves_total'], 2)

    def test_fen_without_moves_is_analyzed_without_inventing_decisions_or_accuracy(self):
        for fen in (chess.STARTING_FEN, '8/8/8/8/8/6k1/8/7K w - - 0 1',
                    '7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'):
            with self.subTest(fen=fen):
                game = chess.pgn.Game()
                game.setup(chess.Board(fen))
                session = self.session(game)
                result = analyze_game(game, session, progress=lambda _: None)
                self.assertEqual(result['moves'], [])
                self.assertEqual(len(result['positions']), 1)
                self.assertEqual(result['positions'][0]['fen'], fen)
                self.assertTrue(all(player['average_accuracy'] is None and player['moves_used'] == 0
                                    for player in result['accuracy_curve']['players'].values()))
                if game.board().is_game_over():
                    self.assertEqual(session.sf_calls, [])
                    self.assertEqual(session.pair_calls, [])

    def test_cancelled_callback_closes_scan_generator_before_aggregation(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 *'))
        session, cancelled, closed = self.session(game), threading.Event(), threading.Event()
        def scans(jobs, *, cancel=None):
            try:
                for index, job in enumerate(jobs):
                    yield index, session.initial_analysis(*job)
            finally:
                closed.set()
        session.analyze_positions = scans
        with patch('analysis.game.pipeline.refresh_saved_curve') as aggregate, self.assertRaises(AnalysisCancelled):
            analyze_game(game, session, progress=lambda _: None, cancel=cancelled,
                         on_position=lambda *_: cancelled.set())
        self.assertTrue(closed.is_set())
        aggregate.assert_not_called()


class BorrowedEngineTests(unittest.TestCase):
    def test_focused_search_uses_a_shared_worker_without_starting_or_closing_another_engine(self):
        from tests.analysis.test_stockfish_search import FakeSearch

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory)/'fixture.exe'
            executable.touch()
            pool = StockfishPool(executable, workers=2, threads_per_worker=1, hash_mb=64)
            self.addCleanup(pool.close)
            scorer = Mock(executable=executable, threads_per_worker=1, hash_mb=64, analysis_pool=pool)
            maia = Mock(model_signature={'maia': ['fixture', 1], 'history_window': 8, 'device': 'cpu'})
            engine = Mock()
            move = chess.Move.from_uci('e2e4')
            engine.analysis.side_effect = lambda *_a, **_k: FakeSearch([{
                'score': chess.engine.PovScore(chess.engine.Cp(25), chess.WHITE),
                'pv': [move], 'depth': 12}])
            with patch('engine.stockfish_pool.start_stockfish', return_value=engine) as start, \
                    patch('engine.runtime.start_stockfish') as standalone:
                with AnalysisSession.borrowed(chess.STARTING_FEN, directory, maia, scorer) as session:
                    result = session.sf([], 100, multipv=1, root_moves=['e2e4'])
                    self.assertEqual(result['lines'][0]['cp'], 25)
                    self.assertEqual(session.sf([], 100, multipv=1, root_moves=['e2e4']), result)
                start.assert_called_once()
                standalone.assert_not_called()
                engine.quit.assert_not_called()
                self.assertFalse(pool._closed.is_set())

    def test_cancellation_stops_only_this_jobs_searches_and_keeps_shared_pool_usable(self):
        searches, engine_instances = [], []
        started = threading.Condition()

        class BlockingSearch:
            def __init__(self):
                self.stopped = threading.Event()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __iter__(self):
                with started:
                    searches.append(self)
                    started.notify_all()
                if not self.stopped.wait(5):
                    raise RuntimeError('Search was not cancelled')
                return iter(())
            def stop(self):
                self.stopped.set()

        def process(*args, **kwargs):
            engine = Mock()
            engine.analysis.side_effect = lambda *_a, **_k: BlockingSearch()
            engine_instances.append(engine)
            return engine

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory)/'fixture.exe'
            executable.touch()
            pool = StockfishPool(executable, workers=3, threads_per_worker=1, hash_mb=96)
            self.addCleanup(pool.close)
            scorer = Mock(executable=executable, threads_per_worker=1, hash_mb=96, analysis_pool=pool)
            maia = Mock(model_signature={'maia': ['fixture', 1], 'history_window': 8, 'device': 'cpu'})
            cancel = threading.Event()
            errors = []
            with patch('engine.stockfish_pool.start_stockfish', side_effect=process), \
                    patch('engine.runtime.start_stockfish') as standalone_start, pool.acquire() as unrelated:
                with AnalysisSession.borrowed(chess.STARTING_FEN, directory, maia, scorer, cancel=cancel) as session:
                    def run():
                        try:
                            list(session.analyze_positions([([], None, []), (['e2e4'], None, [])]))
                        except Exception as exc:
                            errors.append(exc)
                    worker = threading.Thread(target=run)
                    worker.start()
                    with started:
                        ready = started.wait_for(lambda: len(searches) == 2, timeout=3)
                    cancel.set()
                    worker.join(timeout=3)
                    self.assertTrue(ready)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], AnalysisCancelled)
                    self.assertTrue(all(search.stopped.is_set() for search in searches))
                    self.assertFalse(pool._closed.is_set())
                    unrelated.analysis.assert_not_called()
                standalone_start.assert_not_called()
                for engine in engine_instances:
                    engine.close.assert_not_called()
                    engine.quit.assert_not_called()
                with pool.acquire() as reusable:
                    self.assertIsNotNone(reusable)
                scorer.close.assert_not_called()
                maia.load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
