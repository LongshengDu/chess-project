from coach.settings import CONFIG as COACH_CONFIG
import io
import json
import os
import re
import runpy
import tempfile
import unittest

from coach.report_output import validate_report
from pathlib import Path
from unittest.mock import Mock, patch

import chess
import chess.engine
import chess.pgn
import numpy as np

from analysis.game.pipeline import analyze_game, prepare_game_policies
from analysis.position_evaluation import RATINGS, eval_loss, eval_value, stage
from analysis.game.history import history_at
from analysis.game.study import load_game
from analysis.player_rating.parameters import RATINGS as FIT_RATINGS
from analysis.engine_session import Engines, Limits
from analysis.cache import JsonCache
from analysis.game.history import replay
from coach.agent_runner import run_coach
from coach.tools_chess import ChessTools
from coach.coach import CONFIG, codex_model, parser


from tests.coach.fixtures import FakeEngines, ScriptedCodex


class LocalCoachTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def game(self, pgn):
        path = self.root/'input.pgn'; path.write_text(pgn, encoding='utf-8')
        return load_game(path)

    def test_chess_rules_history_start_fen_and_invalid_games(self):
        for pgn, square, piece in [
            ('1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O *', chess.F1, chess.Piece(chess.ROOK,chess.WHITE)),
            ('1. e4 a6 2. e5 d5 3. exd6 *', chess.D6, chess.Piece(chess.PAWN,chess.WHITE)),
            ('[SetUp "1"]\n[FEN "7k/P7/8/8/8/8/8/7K w - - 0 1"]\n\n1. a8=Q+ *', chess.A8, chess.Piece(chess.QUEEN,chess.WHITE))]:
            game = self.game(pgn)
            history = [m.uci() for m in game.mainline_moves()]
            board = replay(game.board().fen(), history)
            self.assertEqual(board.piece_at(square),piece)
            self.assertEqual([m.uci() for m in board.move_stack],history)
        mate = self.game('1. f3 e5 2. g4 Qh4# 0-1')
        self.assertTrue(mate.end().board().is_checkmate())
        for pgn in ('garbage', '1. e4 e5 2. Bh6 *', '1. e4 *\n\n1. d4 *'):
            with self.assertRaises(ValueError): self.game(pgn)
        with self.assertRaises(ValueError): replay(chess.STARTING_FEN,['e2e5'])

    def test_full_scan_all_ratings_played_outside_top_five_and_deduplication(self):
        game = self.game('[WhiteElo "1400"]\n[BlackElo "1700"]\n\n1. e4 e5 2. Nf3 Nc6 *')
        engines = FakeEngines()
        self.addCleanup(engines._temp.cleanup)
        data = analyze_game(game, engines, 'white', 1400, progress=lambda _:None)
        self.assertEqual(len(data['moves']),4)
        for row in data['moves']:
            self.assertEqual(set(map(int,row['maia'])),set(RATINGS))
            self.assertEqual(history_at(data, row['ply']),[m.uci() for m in list(game.mainline_moves())[:row['ply']-1]])
            for rating in row['maia'].values():
                self.assertEqual(len(rating),5)
                self.assertTrue(all(m['p'] > 0 for m in rating))
            self.assertIn(row['played']['move'], [c['move'] for c in row['candidate_moves']])
        self.assertNotIn('e2e4',[m['move'] for m in data['moves'][0]['maia']['1400']])
        restricted = [(tuple(h),tuple(roots)) for h,_,_,roots in engines.sf_calls if roots]
        self.assertEqual(len(set(restricted)),len(restricted))
        self.assertEqual(data['schema_version'], 4)
        for index, row in enumerate(data['moves']):
            calls = [(own, opponents) for history, own, opponents in engines.pair_calls if len(history) == index]
            pairs = [(a, b) for own, opponents in calls for a, b in zip(own, opponents)]
            # Opening decisions also contribute evidence; diagonal policies remain complete.
            expected = [(r, r) for r in FIT_RATINGS]
            self.assertTrue(set(expected).issubset(pairs))
            self.assertTrue(all(len(own) <= 42 for own, _ in calls))
            roots = {c['move'] for c in row['candidate_moves']}
            self.assertEqual(roots, {row['played']['move']} | {m['move'] for moves in row['maia'].values() for m in moves})
            self.assertTrue(all(set(c['maia_p']) == set(map(str, RATINGS)) for c in row['candidate_moves']))

    def test_default_policy_preparation_reuses_common_cache(self):
        game = self.game('1. e4 e5 *')
        engines = FakeEngines()
        self.addCleanup(engines._temp.cleanup)
        data = analyze_game(game, engines, 'white', 1400, progress=lambda _: None)
        engines.human_pairs = Mock(side_effect=AssertionError('Should reuse diagonal policy cache'))
        self.assertIsNone(prepare_game_policies(game, data['moves'], engines))
        for row in data['moves']:
            self.assertEqual(set(row['_rating_policies']), set(FIT_RATINGS))
        self.assertIsNone(data['played_elo']['white']['estimate'])
        self.assertIsNone(data['played_elo']['white']['interval'])

    def test_signed_loss_white_orientation_and_mates(self):
        self.assertEqual(eval_loss(1., -.2, 'white'), 1.2)
        self.assertEqual(eval_loss(-1., .2, 'black'), 1.2)
        self.assertEqual(eval_loss(-1., -1.03, 'black'), -.03)
        self.assertIsNone(eval_loss('#3', 1., 'white'))
        self.assertEqual(eval_value({'cp': None, 'mate': -2}), '#-2')
        self.assertEqual(stage(chess.Board()),'opening')
        self.assertEqual(stage(chess.Board('7k/8/8/8/8/8/P7/K7 w - - 0 1')),'endgame')

    def test_initial_analysis_uses_demo_search_and_keeps_white_mate_perspective(self):
        from tests.analysis.test_stockfish_search import SearchTests, FakeSearch
        engines = Engines(chess.STARTING_FEN, self.root/'cache', limits=Limits(verify_ms=6000, depth=18))
        engines.signature = {'test': True}
        engines.stockfish = SearchTests().engine(incomplete=True)
        self.addCleanup(engines.close)
        result = engines.initial_analysis([], 'e2e4', ['d2d4', 'c2c4'])
        self.assertEqual(len(result['lines']), 20)
        self.assertEqual(result['search']['budget_seconds'], 6)
        self.assertEqual(result['search']['stop_reason'], 'time')
        self.assertEqual({line['uci']: line['depth'] for line in result['lines']}['e2e4'], 17)
        calls = engines.stockfish.analysis.call_count
        self.assertEqual(engines.initial_analysis([], 'e2e4', ['d2d4', 'c2c4']), result)
        self.assertEqual(engines.stockfish.analysis.call_count, calls)
        self.assertTrue(all(call.args[1].time <= 6 for call in engines.stockfish.analysis.call_args_list))
        def mate_search(board, limit, multipv, root_moves=None):
            moves = root_moves or list(board.legal_moves)[:multipv]
            return FakeSearch([{'depth': limit.depth, 'pv': [m],
                                'score': chess.engine.PovScore(chess.engine.Mate(3), board.turn)} for m in moves])
        engines.stockfish.analysis.side_effect = mate_search
        black = engines.initial_analysis(['e2e4'], 'e7e5', ['d7d5'])
        self.assertTrue(all(line['mate'] == -3 and line['cp'] is None for line in black['lines']))

    def test_joint_batches_preserve_opponents_and_history_in_cache(self):
        engines = Engines(chess.STARTING_FEN, self.root/'cache')
        engines.signature = {'test': True}
        engines.maia = Mock()
        def predict(fens, ratings, opponents, *, boards):
            self.assertEqual([m.uci() for m in boards[fens[0]].move_stack], current_history)
            return [{'policy': {m.uci(): 1/boards[f].legal_moves.count() for m in boards[f].legal_moves}} for f in fens]
        engines.maia.batch_evaluate.side_effect = predict
        own, opponents = [1400]*42, list(range(1400, 1442))
        current_history = ['g1f3','g8f6','b1c3','b8c6']
        engines.human_pairs(current_history, own, opponents)
        engines.human_pairs(current_history, own, opponents)
        self.assertEqual(engines.maia.batch_evaluate.call_count, 1)
        self.assertEqual(engines.maia.batch_evaluate.call_args.args[2], opponents)
        current_history = ['b1c3','b8c6','g1f3','g8f6']  # Same FEN, different history.
        engines.human_pairs(current_history, own, opponents)
        self.assertEqual(engines.maia.batch_evaluate.call_count, 2)
        engines.human_pairs(current_history, own, [1500]*42)
        self.assertEqual(engines.maia.batch_evaluate.call_count, 3)
        with self.assertRaises(ValueError):
            engines.human_pairs(current_history, own+[1400], opponents+[1400])

    def test_engine_lifecycle_history_cache_and_bounded_root_search(self):
        checkpoint=self.root/'model.pt'; checkpoint.write_bytes(b'fixture')
        executable=self.root/'stockfish'; executable.write_bytes(b'fixture')
        model=Mock(model_signature={'maia': [str(checkpoint), checkpoint.stat().st_mtime_ns],
                                  'history_window': 8, 'device': 'cpu'})
        def batch(fens,ratings,opponents,*,boards):
            return [{'policy':{m.uci():1/boards[fen].legal_moves.count() for m in boards[fen].legal_moves},'value':.5} for fen in fens]
        model.batch_evaluate.side_effect=batch
        sf=Mock()
        def search(board,limit,**kwargs):
            self.assertIsNotNone(limit.time)
            self.assertLessEqual(limit.time,10)
            move=(kwargs['root_moves'] or list(board.legal_moves))[0]
            context=Mock()
            exact={'score':chess.engine.PovScore(chess.engine.Cp(45),board.turn),'pv':[move],'depth':8}
            context.__enter__=Mock(return_value=iter([exact,{**exact,'depth':20,'lowerbound':True}]))
            context.__exit__=Mock(return_value=False)
            return context
        sf.analysis.side_effect=search
        with patch('engine.maia.MaiaPolicy',return_value=model) as ctor, patch('analysis.engine_session.start_stockfish',return_value=sf) as start:
            with Engines(chess.STARTING_FEN,self.root/'cache',stockfish_path=executable) as engines:
                a=['g1f3','g8f6','b1c3','b8c6']; b=['b1c3','b8c6','g1f3','g8f6']
                self.assertEqual(engines.board(a).fen(),engines.board(b).fen())
                engines.human(a,[1400],1700); engines.human(a,[1400],1700); engines.human(b,[1400],1700)
                self.assertEqual(model.batch_evaluate.call_count,2)
                self.assertEqual([m.uci() for m in model.batch_evaluate.call_args.kwargs['boards'][engines.board(b).fen()].move_stack],b)
                first=engines.sf(['e2e4'],100,multipv=1,root_moves=['e7e5'])
                self.assertEqual(first['lines'][0]['cp'],-45)
                self.assertEqual(first['lines'][0]['depth'],8)
                engines.sf(['e2e4'],100,multipv=1,root_moves=['e7e5'])
                self.assertEqual(sf.analysis.call_count,1)
                with self.assertRaises(ValueError): engines.sf([],engines.limits.max_ms + 1)
                with self.assertRaises(ValueError): engines.sf([],100,root_moves=['e2e5'])
                ctor.assert_called_once(); model.load.assert_called_once(); start.assert_called_once()
                self.assertEqual(start.call_args.kwargs['timeout'], CONFIG['STOCKFISH']['START_TIMEOUT_SECONDS'])
            sf.quit.assert_called_once()

    def test_actual_tool_calling_agent_extends_history_diagrams_and_private_trace(self):
        game=self.game('[WhiteElo "1400"]\n[BlackElo "1500"]\n\n1. e4 e5 2. Nf3 Nc6 *')
        engines=FakeEngines()
        self.addCleanup(engines._temp.cleanup)
        data=analyze_game(game,engines,'white',1400,progress=lambda _:None)
        output=self.root/'output'
        report=ScriptedCodex().run(data,engines,output,max_model_responses=12)
        self.assertIn('## Exercises',report)
        self.assertTrue((output/'coaching.md').is_file())
        self.assertTrue(any((output/'positions').glob('*.svg')))
        trace=(output/'agent_trace.jsonl').read_text(encoding='utf-8')
        self.assertNotIn('PRIVATE_REASONING_SENTINEL',trace)
        self.assertNotIn('PRIVATE_REASONING_SENTINEL',(output/'analysis.json').read_text(encoding='utf-8'))
        events = [json.loads(line) for line in trace.splitlines()]
        self.assertEqual(len([e for e in events if e['phase'] == 'initial']), 2)
        self.assertEqual([e['tool'] for e in events if e['phase'] == 'followup'],['get_game_analysis','get_position','maia_compare','compare_played_vs_candidate','compare_played_vs_candidate'])
        self.assertTrue(any(history[:3]==['e2e4','e7e5','b1c3'] and len(history)>=4 for history,_,_ in engines.human_calls))
        library=ChessTools(data,engines,self.root/'unchecked')
        with self.assertRaises(ValueError): validate_report(library, report)
        with self.assertRaises(ValueError): library.call('python',{'code':'print(1)'})

    def test_removed_provider_and_decorated_tool_schemas(self):
        game=self.game('[WhiteElo "1400"]\n[BlackElo "1500"]\n\n1. e4 e5 2. Nf3 Nc6 *')
        engines=FakeEngines()
        self.addCleanup(engines._temp.cleanup)
        data=analyze_game(game,engines,'white',1400,progress=lambda _:None)
        with self.assertRaises(TypeError):
            run_coach(data,engines,self.root/'removed',provider='removed')
        library=ChessTools(data,engines,self.root/'tools')
        tools={t.name:t for t in library.tools}
        self.assertEqual(len(tools),9)
        self.assertEqual(tools['maia_compare'].input_schema['properties']['ratings']['items']['type'],'integer')
        self.assertEqual(tools['get_position'].input_schema['properties']['line']['anyOf'][0]['items']['type'],'string')
        self.assertIn({'type':'null'}, tools['get_position'].input_schema['properties']['line']['anyOf'])


    def test_engine_configuration_ignores_environment_and_cli_overrides_file(self):
        environment={'STOCKFISH_PATH':'ignored-env-engine', 'MAIA_CHECKPOINT':'ignored-env-model',
                     'STOCKFISH_THREADS':'63', 'STOCKFISH_HASH_MB':'8192',
                     'MAX_STOCKFISH_TIME_MS':'invalid', 'DEFAULT_STOCKFISH_TIME_MS':'invalid'}
        with patch.dict(os.environ,environment):
            loaded = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'coach/settings.py'))['CONFIG']
            self.assertEqual(loaded,CONFIG)
            with patch.dict(COACH_CONFIG['ANALYSIS'], STOCKFISH_THREADS_PER_WORKER=6, STOCKFISH_HASH_MB_PER_WORKER=384), patch.dict(COACH_CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], MAX_DEPTH=15), patch.dict(COACH_CONFIG['STOCKFISH'], EXECUTABLE='configured-stockfish'), patch.dict(COACH_CONFIG['MAIA'], CHECKPOINT='configured-maia', DEVICE='cpu'), patch.dict(COACH_CONFIG['ANALYSIS'], CACHE_DIR=self.root/'cache'):
                args=parser().parse_args(['input.pgn','--side','white','--elo','1400'])
                self.assertEqual((args.threads_per_worker,args.hash_mb,args.stockfish_path,args.maia_checkpoint),
                                 (6,384,'configured-stockfish','configured-maia'))
                self.assertEqual((args.depth,args.device,args.cache_dir),(15,'cpu',self.root/'cache'))
                args=parser().parse_args(['input.pgn','--side','white','--elo','1400','--threads','2','--hash-mb-per-worker','256','--depth','12'])
                self.assertEqual((args.threads_per_worker,args.hash_mb,args.depth),(2,256,12))





if __name__=='__main__':
    unittest.main()
