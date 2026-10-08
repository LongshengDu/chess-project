"""Shared position evidence preserves histories and reuses individual requests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import chess

from analysis.cache.requests import maia_request, stockfish_initial_request
from analysis.session import AnalysisSession, Limits
from analysis.stockfish_search import search_candidates
from tests.analysis import test_stockfish_search


class PositionSessionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.session = AnalysisSession(chess.STARTING_FEN, self.directory, stockfish_path='unused')
        self.addCleanup(self.session.close)
        self.session.engines.signature = {'maia': ['model', 1], 'device': 'cpu', 'history_window': 8,
                                          'stockfish': ['engine', 1], 'threads': 2, 'hash_mb': 128}
        self.model = self.session.engines.maia = Mock()
        self.model.batch_evaluate.side_effect = self.predict

    @staticmethod
    def predict(fens, own, opponents, *, boards, **kwargs):
        boards = [boards[fen] for fen in fens] if isinstance(boards, dict) else boards
        return [{'policy': {move.uci(): 1/board.legal_moves.count() for move in board.legal_moves},
                 'value': .5} for board in boards]

    def test_batch_single_overlap_and_duplicates_share_rating_pairs(self):
        first = self.session.human_pairs([], [1400, 1600], [1500, 1500])
        self.assertEqual(self.session.human([], [1600], 1500)['1600'], first[1])
        result = self.session.human_pair_batches([
            (chess.Board(), [1600, 1800, 1800], [1500, 1500, 1500]),
            (chess.Board(), [1800, 2000], [1500, 1500]),
        ])
        self.assertEqual(self.model.batch_evaluate.call_count, 2)
        self.assertEqual(self.model.batch_evaluate.call_args.args[1], [1800, 2000])
        self.assertEqual(result[0][1], result[1][0])
        self.assertEqual(len(list((self.directory/'positions').glob('*.json'))), 1)

    def test_initial_search_promotes_both_limits_and_candidates_then_reuses(self):
        board = chess.Board()
        request = stockfish_initial_request(self.session.engines.signature, 18, 2, 8,
                                           'bounded', {'forcedCandidateMoves': ['d2d4']})
        self.session.cache.put(board, 'stockfish', request,
                               test_stockfish_search.evaluation_frame(board, depth=6))
        self.session.engines.stockfish = Mock()
        self.session.limits = Limits(depth=12, verify_ms=3000, max_ms=8000)
        with patch('analysis.session.stream_evaluations', side_effect=lambda *args, **kwargs:
                   iter([test_stockfish_search.evaluation_frame(board, depth=5)])) as search:
            self.session.initial_analysis([], 'e2e4', [])
            self.assertEqual(search.call_args.args[2], 18)
            self.assertEqual(search.call_args.kwargs['seconds'], 3)
            self.assertEqual(set(search.call_args.args[3]['forcedCandidateMoves']), {'d2d4', 'e2e4'})
            # A stronger depth request preserves the prior larger time too.
            self.session.limits = Limits(depth=20, verify_ms=1000, max_ms=8000)
            self.session.initial_analysis([], 'e2e4', [])
            self.assertEqual(search.call_args.args[2], 20)
            self.assertEqual(search.call_args.kwargs['seconds'], 3)
            self.session.limits = Limits(depth=18, verify_ms=3000, max_ms=8000)
            self.session.initial_analysis([], 'd2d4', [])
            self.assertEqual(search.call_count, 2)
        self.assertEqual(self.session.stats['stockfish_cache_hits'], 1)

    def test_equal_fen_different_histories_share_file_not_result(self):
        histories = [['g1f3', 'g8f6', 'b1c3', 'b8c6'], ['b1c3', 'b8c6', 'g1f3', 'g8f6']]
        boards = [self.session.board(history) for history in histories]
        self.assertEqual(boards[0].fen(), boards[1].fen())
        self.session.human_pair_batches([(board, [1600], [1600]) for board in boards])
        self.assertEqual(len(self.model.batch_evaluate.call_args.args[0]), 2)
        self.assertEqual([list(b.move_stack) for b in self.model.batch_evaluate.call_args.kwargs['boards']],
                         [list(b.move_stack) for b in boards])
        files = list((self.directory/'positions').glob('*.json'))
        self.assertEqual(len(files), 1)
        self.assertEqual(len(json.loads(files[0].read_text(encoding='utf-8'))['histories']), 2)

    def test_engine_signatures_are_independent(self):
        self.session.human([], [1600], 1600)
        self.session.engines.signature.update(stockfish=['changed', 2], threads=4, hash_mb=256)
        self.session.human([], [1600], 1600)
        self.assertEqual(self.model.batch_evaluate.call_count, 1)
        engine = self.session.engines.stockfish = test_stockfish_search.SearchTests().engine()
        first = self.session.initial_analysis([], 'e2e4', [])
        searches = engine.analysis.call_count
        self.session.engines.signature.update(maia=['changed', 2], device='cuda', history_window=4)
        self.assertEqual(self.session.initial_analysis([], 'e2e4', []), first)
        self.assertEqual(engine.analysis.call_count, searches)
        self.session.human([], [1600], 1600)
        self.assertEqual(self.model.batch_evaluate.call_count, 2)

    def test_new_session_reuses_saved_individual_rating_pairs(self):
        values = self.session.human_pairs([], [1400, 1600], [1500, 1500])
        session = AnalysisSession(chess.STARTING_FEN, self.directory, stockfish_path='unused')
        self.addCleanup(session.close)
        session.engines.signature = self.session.engines.signature.copy()
        session.engines.maia = Mock()
        self.assertEqual(session.human_pairs([], [1600, 1400], [1500, 1500]), values[::-1])
        session.engines.maia.batch_evaluate.assert_not_called()
        self.assertEqual(list(self.directory.glob('*.json')), [])

    def test_stockfish_stream_frame_reused_and_search_settings_are_distinct(self):
        board = chess.Board()
        self.session.engines.stockfish = test_stockfish_search.SearchTests().engine()
        first = self.session.initial_analysis([], 'e2e4', [])
        options = {'forcedCandidateMoves': ['e2e4'], 'maiaCandidateMoves': []}
        request = stockfish_initial_request(self.session.engines.signature, self.session.limits.depth,
            self.session.limits.verify_ms/1000, self.session.limits.max_ms/1000,
            self.session.limits.analysis_strategy, options)
        frame = self.session.cache.get(board, 'stockfish', request)
        self.assertEqual(set(frame['cp_vec']), {move.uci() for move in board.legal_moves})
        self.assertNotIn('lines', frame)
        engine = self.session.engines.stockfish
        searches = engine.analysis.call_count
        self.assertEqual(self.session.initial_analysis([], 'e2e4', []), first)
        self.assertEqual(engine.analysis.call_count, searches)
        self.session.initial_analysis([], 'd2d4', [])
        self.assertGreater(engine.analysis.call_count, searches)
        self.assertEqual(len(list((self.directory/'positions').glob('*.json'))), 1)

    def test_cached_scans_publish_original_requested_limits_without_changing_pinned_measurements(self):
        board = chess.Board()
        for strategy in ('bounded', 'staged', 'exhaustive'):
            with self.subTest(strategy=strategy):
                self.session.limits = Limits(depth=12, verify_ms=1000, max_ms=2000,
                                             analysis_strategy=strategy)
                request = stockfish_initial_request(self.session.engines.signature, 18, 4, 8,
                    strategy, {'forcedCandidateMoves': ['e2e4']})
                frame = test_stockfish_search.evaluation_frame(board, depth=6, strategy=strategy)
                reference = self.session.cache.put(board, 'stockfish', request, frame)
                before = {path: path.read_bytes() for path in self.directory.rglob('*.json')}
                with patch.object(self.session.engines, 'get_stockfish', side_effect=AssertionError('cache must not search')):
                    scan = self.session.initial_analysis([], 'e2e4', [])
                self.assertEqual(scan['search']['target_depth'], 18)
                self.assertEqual(scan['search']['budget_seconds'], 4 if strategy == 'bounded' else 8)
                self.assertEqual(scan['search']['policy_version'], request['policy_version'])
                self.assertEqual(scan['search']['candidate_moves'], request['candidate_moves'])
                self.assertEqual(self.session.cache.get_reference(reference), frame)
                self.assertEqual(before, {path: path.read_bytes() for path in self.directory.rglob('*.json')})

    def test_refresh_session_recomputes_maia_and_both_stockfish_searches_once(self):
        self.session.engines.stockfish = test_stockfish_search.SearchTests().engine()
        self.session.human_pairs([], [1400, 1600], [1600, 1600])
        self.session.initial_analysis([], 'e2e4', [])
        self.session.sf([], 100, multipv=2)
        preserved = self.session.cache.put(chess.Board(), 'future', {}, {'keep': True})

        refresh = AnalysisSession(chess.STARTING_FEN, self.directory, stockfish_path='unused',
                                  refresh_cache=True)
        self.addCleanup(refresh.close)
        refresh.engines.signature = self.session.engines.signature.copy()
        refresh.engines.maia = Mock()
        refresh.engines.maia.batch_evaluate.side_effect = self.predict
        refresh.engines.stockfish = test_stockfish_search.SearchTests().engine()

        first = refresh.human_pairs([], [1400, 1600], [1600, 1600])
        self.assertEqual(refresh.engines.maia.batch_evaluate.call_count, 1)
        self.assertEqual(refresh.human([], [1600], 1600)['1600'], first[1])
        self.assertEqual(refresh.engines.maia.batch_evaluate.call_count, 1)
        scan = refresh.initial_analysis([], 'e2e4', [])
        initial_calls = refresh.engines.stockfish.analysis.call_count
        self.assertGreater(initial_calls, 0)
        self.assertEqual(refresh.initial_analysis([], 'e2e4', []), scan)
        self.assertEqual(refresh.engines.stockfish.analysis.call_count, initial_calls)
        exploration = refresh.sf([], 100, multipv=2)
        self.assertEqual(refresh.engines.stockfish.analysis.call_count, initial_calls + 1)
        self.assertEqual(refresh.sf([], 100, multipv=2), exploration)
        self.assertEqual(refresh.engines.stockfish.analysis.call_count, initial_calls + 1)
        self.assertEqual(self.session.cache.get_reference(preserved), {'keep': True})

    def test_only_effective_stockfish_options_affect_identity(self):
        args = (self.session.engines.signature, 18, 1., 4., 'bounded')
        empty = stockfish_initial_request(*args, {})
        self.assertEqual(empty, stockfish_initial_request(*args, {'maiaPolicy': {'e2e4': .5}}))
        options = {'maiaCandidateMoves': ['e2e4', 'd2d4', 'c2c4', 'g1f3', 'a2a4']}
        expected = {'maiaCandidateMoves': options['maiaCandidateMoves'][:4]}
        self.assertEqual(stockfish_initial_request(*args, options),
                         stockfish_initial_request(*args, expected))

    def test_candidate_roles_and_duplicates_reuse_the_same_search(self):
        engine = self.session.engines.stockfish = test_stockfish_search.SearchTests().engine()
        first = self.session.initial_analysis([], 'e2e4', ['d2d4', 'e2e4', 'd2d4'])
        searches = engine.analysis.call_count
        self.assertGreater(searches, 0)
        second = self.session.initial_analysis([], None, ['e2e4', 'd2d4'])
        self.assertEqual(second, first)
        self.assertEqual(engine.analysis.call_count, searches)
        self.assertEqual(self.session.stats['stockfish_cache_hits'], 1)

    def test_candidate_order_and_selection_match_each_search_strategy(self):
        options = {'forcedCandidateMoves': ['c2c4', 'd2d4', 'c2c4'],
                   'maiaCandidateMoves': ['e2e4', 'd2d4', 'e2e4', 'g1f3', 'b1c3']}
        self.assertEqual(search_candidates(18, 'bounded', options),
                         ['c2c4', 'd2d4', 'e2e4', 'g1f3'])
        self.assertEqual(search_candidates(18, 'staged', options),
                         ['e2e4', 'd2d4', 'g1f3', 'c2c4'])
        # Truncate to the first four supplied Maia candidates BEFORE deduplication.
        self.assertNotIn('b1c3', search_candidates(18, 'bounded', options))
        for strategy in ('bounded', 'staged'):
            args = (self.session.engines.signature, 18, 1., 4., strategy)
            merged = {'forcedCandidateMoves': search_candidates(18, strategy, options)}
            self.assertEqual(stockfish_initial_request(*args, options),
                             stockfish_initial_request(*args, merged))

    def test_exhaustive_and_shallow_staged_search_ignore_candidate_options(self):
        options = {'forcedCandidateMoves': ['e2e4'], 'maiaCandidateMoves': ['d2d4']}
        for strategy, depth in (('exhaustive', 18), ('exhaustive', 4), ('staged', 4)):
            with self.subTest(strategy=strategy, depth=depth):
                args = (self.session.engines.signature, depth, 1., 4., strategy)
                self.assertEqual(stockfish_initial_request(*args, options),
                                 stockfish_initial_request(*args, {}))
        args = (self.session.engines.signature, 4, 1., 4., 'bounded')
        self.assertNotEqual(stockfish_initial_request(*args, options),
                            stockfish_initial_request(*args, {}))

    def test_depth_driven_searches_use_only_the_effective_time_allowance(self):
        signature = self.session.engines.signature
        options = {'forcedCandidateMoves': ['e2e4'], 'maiaCandidateMoves': ['d2d4']}
        board = chess.Board()
        for strategy in ('staged', 'exhaustive'):
            with self.subTest(strategy=strategy):
                nominal = stockfish_initial_request(signature, 18, 1., 4., strategy, options)
                effective = stockfish_initial_request(signature, 18, 4., 4., strategy, options)
                self.assertEqual(nominal, effective)
                frame = test_stockfish_search.evaluation_frame(board, strategy=strategy)
                self.session.cache.put(board, 'stockfish', nominal, frame)
                self.assertEqual(self.session.cache.get(board, 'stockfish', effective), frame)
                changed_max = stockfish_initial_request(signature, 18, 1., 5., strategy, options)
                self.assertIsNone(self.session.cache.get(board, 'stockfish', changed_max))
        self.assertNotEqual(
            stockfish_initial_request(signature, 18, 1., 4., 'bounded', options),
            stockfish_initial_request(signature, 18, 4., 4., 'bounded', options))

    def test_meaningful_search_differences_remain_cache_misses(self):
        signature = self.session.engines.signature
        options = {'forcedCandidateMoves': ['e2e4', 'd2d4']}
        args = (signature, 18, 1., 4., 'bounded', options)
        baseline = stockfish_initial_request(*args)
        board = chess.Board()
        self.session.cache.put(board, 'stockfish', baseline, test_stockfish_search.evaluation_frame(board))
        variants = [
            (signature, 19, 1., 4., 'bounded', options),
            (signature, 18, 2., 4., 'bounded', options),
            (signature, 18, 1., 4., 'staged', options),
            (signature, 18, 1., 4., 'bounded', {'forcedCandidateMoves': ['g1f3']}),
        ]
        variants.extend(({**signature, key: value}, *args[1:]) for key, value in (
            ('stockfish', ['changed', 2]), ('threads', 4), ('hash_mb', 256)))
        for variant in variants:
            with self.subTest(request=variant):
                request = stockfish_initial_request(*variant)
                self.assertIsNone(self.session.cache.get(board, 'stockfish', request))

    def test_stockfish_request_preserves_history_context(self):
        first = self.session.board(['g1f3', 'g8f6', 'b1c3', 'b8c6'])
        second = self.session.board(['b1c3', 'b8c6', 'g1f3', 'g8f6'])
        self.assertEqual(first.fen(), second.fen())
        request = stockfish_initial_request(self.session.engines.signature, 18, 1., 4., 'bounded', {})
        self.session.cache.put(first, 'stockfish', request, test_stockfish_search.evaluation_frame(first))
        self.assertIsNone(self.session.cache.get(second, 'stockfish', request))

    def test_terminal_maia_pairs_need_no_inference(self):
        history = ['f2f3', 'e7e5', 'g2g4', 'd8h4']
        result = self.session.human_pairs(history, [1600, 1800], [1600, 1800])
        self.assertEqual(result, [{'policy': {}, 'value': 0.}, {'policy': {}, 'value': 0.}])
        self.model.batch_evaluate.assert_not_called()
        self.assertEqual(self.session.cache.get(self.session.board(history), 'maia',
                         maia_request(self.session.engines.signature, 1600, 1600)), result[0])
        scan = self.session.initial_analysis(history, None, [])
        self.assertEqual(scan['search']['result'], '0-1')
        files = list((self.directory/'positions').glob('*.json'))
        self.assertEqual(len(files), 1)
        document = json.loads(files[0].read_text(encoding='utf-8'))
        self.assertEqual(set(document['evidence']), {'maia', 'stockfish'})


if __name__ == '__main__':
    unittest.main()
