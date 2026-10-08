"""Shared analysis defaults, search policy selection and cache compatibility."""
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn

from coach.tools_chess import ChessTools
from analysis.game.pipeline import analyze_game
from coach.coach import parser
from analysis.session import AnalysisSession, Limits
from coach.settings import CONFIG
from tests.coach.fixtures import FakeAnalysisSession
from tests.analysis import test_stockfish_search as search_fixtures


class SharedAnalysisSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Both consumers read the same YAML; inject each consumer explicitly.
        self.enterContext(patch.dict('analysis.session.CONFIG', CONFIG))

    def test_fractional_seconds_keep_millisecond_precision_for_agent_tools(self):
        with patch.dict(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'],
                        DEFAULT_SEARCH_SECONDS=3.125, MAX_SEARCH_SECONDS=13.875):
            args = parser().parse_args(['example.pgn','--side','white','--elo','1600', '--rating-scale', 'lb'])
            limits = Limits()
            self.assertEqual((args.verify_ms,args.max_ms),(3125,13875))
            self.assertEqual((limits.verify_ms,limits.max_ms),(3125,13875))
            with self.assertRaises(ValueError):
                Limits(max_ms=13876)


    def test_cli_and_direct_engine_use_shared_defaults(self):
        with patch.dict(CONFIG['MAIA'], DEVICE='cpu'), \
             patch.dict(CONFIG['STOCKFISH'], START_TIMEOUT_SECONDS=7), \
             patch.dict(CONFIG['ANALYSIS'], STOCKFISH_SEARCH_STRATEGY='bounded'), \
             patch.dict(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], MAX_DEPTH=22, DEFAULT_SEARCH_SECONDS=1.2, MAX_SEARCH_SECONDS=15.0):
            args = parser().parse_args(['example.pgn', '--side', 'white', '--elo', '1600', '--rating-scale', 'lb'])
            session = AnalysisSession(chess.STARTING_FEN, self.root)
            self.assertEqual(session.engines.device, 'cpu')
            for field in ('verify_ms', 'max_ms', 'depth', 'analysis_strategy'):
                self.assertEqual(getattr(session.limits, field), getattr(args, field))
            self.assertEqual((args.verify_ms, args.max_ms, args.depth), (1200, 15000, 22))
            self.assertNotIn('--tactical-ms', parser().format_help())
            self.assertNotIn('--analysis-depth', parser().format_help())
            self.assertNotIn('--analysis-time-scale', parser().format_help())
            args = parser().parse_args(['example.pgn', '--side', 'white', '--elo', '1600', '--rating-scale', 'lb',
                '--verify-ms', '900', '--max-ms', '14000', '--depth', '19', '--analysis-strategy', 'staged'])
            self.assertEqual((args.verify_ms, args.max_ms, args.depth, args.analysis_strategy),
                             (900, 14000, 19, 'staged'))
        self.assertFalse(any(key.startswith(('COACH_ANALYSIS_', 'COACH_STOCKFISH_')) for key in CONFIG))
        for key in ('COACH_DEVICE', 'COACH_VERIFY_MS', 'COACH_TACTICAL_MS', 'COACH_MAX_MS'):
            self.assertNotIn(key, CONFIG)

    def test_limits_honor_shared_time_ceiling_and_analysis_bounds(self):
        with patch.dict(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], MAX_SEARCH_SECONDS=30.0):
            self.assertEqual(Limits(max_ms=20000).max_ms, 20000)
            for options in ({'max_ms': 30001}, {'verify_ms': 30001}, {'max_ms': True},
                            {'depth': CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['MAX_DEPTH'] + 1},
                            {'depth': True}, {'depth': 0}, {'analysis_strategy': 'unknown'},
                            {'max_ms': 4000, 'verify_ms': 4001}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    Limits(**options)

    def test_initial_analysis_uses_shared_strategies_and_separate_caches(self):
        session = AnalysisSession(chess.STARTING_FEN, self.root, limits=Limits(verify_ms=6000, depth=18))
        session.engines.stockfish = search_fixtures.SearchTests().engine()
        session.engines.signature = {'test': True}
        self.addCleanup(session.close)
        saved = {}
        for strategy in ('bounded', 'staged', 'exhaustive'):
            session.limits = replace(session.limits, analysis_strategy=strategy)
            before = session.engines.stockfish.analysis.call_count
            saved[strategy] = session.initial_analysis([], 'e2e4', ['d2d4'])
            result = saved[strategy]
            self.assertGreater(session.engines.stockfish.analysis.call_count, before)
            self.assertEqual(result['search']['strategy'], strategy)
            self.assertEqual(result['search']['budget_seconds'], 6. if strategy == 'bounded' else session.limits.max_ms / 1000)
            self.assertEqual(result['search']['max_budget_seconds'], result['search']['budget_seconds'])
            self.assertTrue(result['search']['coverage_complete'])
            self.assertEqual({row['uci'] for row in result['lines']}, {move.uci() for move in chess.Board().legal_moves})
            self.assertIn(result['best_move'], result['engine_moves'])
            before = session.engines.stockfish.analysis.call_count
            self.assertEqual(session.initial_analysis([], 'e2e4', ['d2d4']), result)
            self.assertEqual(session.engines.stockfish.analysis.call_count, before)
        session.limits = replace(session.limits, analysis_strategy='bounded')
        before = session.engines.stockfish.analysis.call_count
        self.assertEqual(session.initial_analysis([], 'e2e4', ['d2d4']), saved['bounded'])
        self.assertEqual(session.engines.stockfish.analysis.call_count, before)
        session.limits = replace(session.limits, verify_ms=7000)
        self.assertEqual(session.initial_analysis([], 'e2e4', ['d2d4'])['search']['budget_seconds'], 7.)
        self.assertGreater(session.engines.stockfish.analysis.call_count, before)
        before = session.engines.stockfish.analysis.call_count
        session.limits = replace(session.limits, max_ms=15000)
        self.assertEqual(session.initial_analysis([], 'e2e4', ['d2d4'])['search']['max_budget_seconds'], 7.)
        self.assertEqual(session.engines.stockfish.analysis.call_count, before)
        before = session.engines.stockfish.analysis.call_count
        session.limits = replace(session.limits, depth=15)
        self.assertEqual(session.initial_analysis([], 'e2e4', ['d2d4'])['search']['target_depth'], 18)
        self.assertEqual(session.engines.stockfish.analysis.call_count, before)

    def test_tool_schema_exposes_the_effective_shared_search_limits(self):
        session = FakeAnalysisSession()
        self.addCleanup(session._temp.cleanup)
        session.limits = replace(session.limits, max_ms=12000, verify_ms=2000)
        game = chess.pgn.read_game(io.StringIO('1. e4 *'))
        analysis = analyze_game(game, session, actual_elo=1600, progress=lambda _: None)
        tools = ChessTools(analysis, session, self.root, side='white').tools
        tool = next(tool for tool in tools if tool.name == 'stockfish_analyze')
        argument = tool.input_schema['properties']['movetime_ms']
        self.assertEqual((argument['minimum'], argument['maximum']), (1, 12000))
        self.assertIn('normally 2000', argument['description'])


if __name__ == '__main__':
    unittest.main()
