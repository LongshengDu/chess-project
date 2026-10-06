from analysis.stockfish_search import BOUNDED_POLICY_VERSION
from contextlib import contextmanager
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chess
import chess.engine

from backend.app import create_app
from backend.analysis_positions import PlatformAnalysis
from backend.settings import CONFIG as BACKEND_CONFIG


class PlatformAnalysisTests(unittest.TestCase):
    def setUp(self):
        presets = patch.dict(BACKEND_CONFIG['FRONTEND'], STOCKFISH_TIME_SCALE_BY_DEPTH={12: .2, 15: .5, 18: 1.})
        presets.start()
        self.addCleanup(presets.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'index.html').write_text('upstream analysis')
        self.maia = Mock()
        self.stockfish = Mock()
        self.platform = PlatformAnalysis(self.maia, self.stockfish, self.root / 'studies.sqlite3', strategy='staged')
        self.client = create_app(self.platform, self.root).test_client()

    def save(self, **data):
        response = self.client.post('/api/platform/games', json=data)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json['game_id']

    def test_persist_game_variations_and_fen_across_instances(self):
        game_id = self.save(name='Branches', pgn='1. e4 (1. d4 d5) e5 *')
        snapshot = self.client.get(f'/api/platform/games/{game_id}').json['snapshot']
        self.assertEqual(snapshot['ucis'], ['e2e4', 'e7e5'])
        self.assertEqual([v['uci'] for v in snapshot['variations']], ['d2d4', 'd7d5'])
        reopened = PlatformAnalysis(self.maia, self.stockfish, self.root / 'studies.sqlite3')
        with reopened.repository.connect() as db:
            self.assertEqual(db.execute('SELECT name FROM games WHERE id=?', (game_id,)).fetchone()[0], 'Branches')
        fen = 'r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R b KQkq - 9 23'
        fen_id = self.save(fen=fen)
        self.assertEqual(self.client.get(f'/api/platform/games/{fen_id}').json['snapshot']['positions'][0]['fullFen'], fen)

    def test_metadata_favorites_cache_and_delete_cascade(self):
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}'
        self.client.patch(url, json={'custom_name':'Renamed', 'is_favorited':True})
        favorite = self.client.get('/api/platform/games?type=favorites').json['games'][0]
        self.assertEqual(favorite['custom_name'], 'Renamed')
        cache = [{'ply':0, 'fen':chess.STARTING_FEN, 'maia':{'maia_kdd_1500':{
            'value':.5, 'policy':{'e2e4':.8, 'a2a3':.2}}}, 'stockfish':{'depth':12, 'cp_vec':{'e2e4':30}}}]
        self.assertEqual(self.client.post(url + '/analysis', json=cache).status_code, 200)
        restored = self.client.get(url + '/analysis').json['positions']
        self.assertEqual(restored, cache)
        self.assertEqual(list(restored[0]['maia']['maia_kdd_1500']['policy']), ['e2e4', 'a2a3'])
        self.assertEqual(self.client.delete(url).status_code, 200)
        self.assertEqual(self.client.get(url).status_code, 404)
        with self.platform.repository.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM analyses').fetchone()[0], 0)

    def test_maia_rating_contract_and_cache(self):
        ratings = list(range(600, 2601, 100))
        self.maia.batch_evaluate.return_value = [{'value':.5, 'policy':{'e2e4':1.0}}] * 21
        data = {'fens':[chess.STARTING_FEN]*21, 'ratings':ratings, 'opponents':ratings}
        for _ in range(2):
            result = self.client.post('/api/platform/maia', json=data)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(len(result.json['result']), 21)
        self.maia.batch_evaluate.assert_called_once_with(data['fens'], ratings, ratings)
        for patch in ({'ratings':[599]*21}, {'opponents':[]}, {'fens':['bad']*21}, {'fens':[]}):
            self.assertEqual(self.client.post('/api/platform/maia', json={**data, **patch}).status_code, 400)

    def test_multi_position_maia_batch_preserves_mainline_histories(self):
        board=chess.Board()
        fens=[board.fen()]; board.push_uci('e2e4'); fens.append(board.fen())
        board.push_uci('e7e5'); fens.append(board.fen())
        request={'fens':[fen for fen in fens for _ in range(21)],'ratings':list(range(600,2601,100))*3,
                 'opponents':[1500]*63,'start_fen':chess.STARTING_FEN,
                 'histories':dict(zip(fens,[[],['e2e4'],['e2e4','e7e5']]))}
        self.maia.batch_evaluate.return_value=[{'policy':{},'value':.5}]*63
        response=self.client.post('/api/platform/maia',json=request)
        self.assertEqual(response.status_code,200,response.text)
        passed=self.maia.batch_evaluate.call_args.kwargs['boards']
        self.assertEqual([len(passed[fen].move_stack) for fen in fens],[0,1,2])
        request['histories'][fens[-1]]=['e2e4']
        self.assertEqual(self.client.post('/api/platform/maia',json=request).status_code,400)

    def test_analysis_cache_survives_reopen_without_reports_or_rating_work(self):
        game_id = self.save(pgn='[White "A"]\n[Black "B"]\n\n1. e4 e5 *')
        url = f'/api/platform/games/{game_id}'
        snapshot = self.client.get(url).json['snapshot']
        positions = []
        for ply, position in enumerate(snapshot['positions']):
            board = chess.Board(position['fullFen'])
            legal = [move.uci() for move in board.legal_moves]
            positions.append({
                'ply': ply, 'fen': board.fen(),
                'maia': {f'maia_kdd_{rating}': {'value': .5,
                    'policy': dict.fromkeys(legal, 1 / len(legal))}
                    for rating in range(600, 2601, 100)},
                'stockfish': {'complete': True, 'depth': 18, 'strategy': 'staged',
                    'cp_vec': dict.fromkeys(legal, 0), 'best_move': legal[0]},
            })
        with patch('analysis.player_rating.service.fit_game', side_effect=AssertionError('Saving demo analysis must not fit ratings')):
            for _ in range(2):
                response = self.client.post(url + '/analysis', json=positions)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json, {'ok': True})
                self.assertEqual(self.client.get(url + '/analysis').json['positions'], positions)
            reopened = PlatformAnalysis(self.maia, self.stockfish, self.root / 'studies.sqlite3', strategy='staged')
            client = create_app(reopened, self.root).test_client()
            self.assertEqual(client.get(url + '/analysis').json['positions'], positions)
            self.assertEqual(client.get(url).json['snapshot'], snapshot)
            for current in (self.client, client):
                self.assertFalse(any('/report' in rule.rule for rule in current.application.url_map.iter_rules()))
                self.assertEqual(current.get(url + '/report').status_code, 404)
                # Flask's GET-only static catch-all rejects POST with 405.
                self.assertEqual(current.post(url + '/report', json={}).status_code, 405)
                for filename in ('coaching.md', 'annotated.pgn', 'evidence.json', 'manifest.json'):
                    with self.subTest(filename=filename):
                        route = url + '/report/0123456789abcdef/' + filename
                        self.assertEqual(current.get(route).status_code, 404)
        self.assertEqual(self.maia.mock_calls, [])
        self.assertEqual(self.stockfish.mock_calls, [])
        self.assertEqual({path.name for path in self.root.iterdir()}, {'index.html', 'studies.sqlite3'})


    def test_exploration_rejects_invalid_limits_and_moves(self):
        for patch in ({'depth':23}, {'depth':True}, {'seconds':0}, {'seconds':float('nan')},
                      {'seconds':True}, {'move':'e2e5'}, {'fen':'bad'}):
            response = self.client.post('/api/platform/explore', json={'fen':chess.STARTING_FEN, **patch})
            self.assertEqual(response.status_code, 400, response.text)
        self.stockfish._get_engine.assert_not_called()

    def test_exploration_defaults_reach_search_in_seconds_and_plies(self):
        with patch.dict(BACKEND_CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION'],
                        DEFAULT_SEARCH_SECONDS=.625, MAX_DEPTH=16), \
             patch('backend.routes_analysis.explore',return_value={'fixture':True}) as explore:
            response = self.client.post('/api/platform/explore',json={'fen':chess.STARTING_FEN})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(explore.call_args.kwargs,{'move':None,'seconds':.625,'depth':16})

    def test_exploration_enforces_configured_maximum_instead_of_fixed_two_seconds(self):
        with patch('backend.routes_analysis.explore',return_value={'fixture':True}) as explore:
            for maximum, requested, status in ((.75,.75,200),(.75,.751,400),(3.,2.5,200),(3.,3.001,400)):
                with self.subTest(maximum=maximum, requested=requested), \
                     patch.dict(BACKEND_CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION'], MAX_SEARCH_SECONDS=maximum):
                    explore.reset_mock()
                    response = self.client.post('/api/platform/explore',json={
                        'fen':chess.STARTING_FEN,'seconds':requested})
                    self.assertEqual(response.status_code,status,response.text)
                    if status == 200:
                        self.assertEqual(explore.call_args.kwargs['seconds'],requested)
                    else:
                        explore.assert_not_called()
            with patch.dict(BACKEND_CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION'],
                            DEFAULT_SEARCH_SECONDS=1., MAX_SEARCH_SECONDS=.75):
                explore.reset_mock()
                self.assertEqual(self.client.post('/api/platform/explore',json={'fen':chess.STARTING_FEN}).status_code,400)
                explore.assert_not_called()

    def test_stream_complete_depths_white_score_and_cancellation(self):
        board = chess.Board('7k/8/8/8/8/8/8/R5K1 b - - 0 1')
        moves = list(board.legal_moves)
        infos = [dict(depth=depth, pv=[move], score=chess.engine.PovScore(chess.engine.Cp(-200-i), chess.BLACK))
                 for depth in (1, 2) for i, move in enumerate(moves)]
        class Search:
            closed = False
            def __enter__(self): return iter(infos)
            def __exit__(self, *args): self.closed = True
        search = Search()
        self.stockfish._get_engine.return_value.analysis.return_value = search
        stream = self.platform.stream_stockfish(board, 12)
        first = json.loads(next(stream))
        self.assertEqual(first['depth'], 1)
        self.assertEqual(set(first['cp_vec']), {move.uci() for move in moves})
        self.assertTrue(all(cp >= 200 for cp in first['cp_vec'].values()))
        stream.close()
        self.assertTrue(search.closed)
        self.assertFalse(self.platform.stockfish_lock.locked())

    def test_invalid_payloads_and_route_isolation(self):
        for body in ({}, {'fen':'bad'}, {'pgn':'nonsense'}, {'fen':chess.STARTING_FEN,'pgn':'1. e4 *'}, []):
            self.assertEqual(self.client.post('/api/platform/games', json=body).status_code, 400)
        for depth in (0, 19, '12'):
            self.assertEqual(self.client.post('/api/platform/stockfish', json={'fen':chess.STARTING_FEN,'depth':depth}).status_code, 400)
        with self.client.get('/analysis/arbitrary/custom') as response:
            self.assertEqual(response.text, 'upstream analysis')
        self.assertEqual(self.client.get('/api/state').status_code, 404)
        rules = {rule.rule for rule in self.client.application.url_map.iter_rules()}
        legacy = ('/api/state', '/api/move', '/api/reply', '/api/action', '/api/new-game',
                  '/api/analysis/config', '/api/analysis/pgn', '/api/analysis/fen',
                  '/api/analysis/position', '/api/analysis/move')
        for route in legacy:
            with self.subTest(route=route):
                self.assertNotIn(route, rules)
                self.assertEqual(self.client.get(route).status_code, 404)
                # GET-only static catch-all yields 405 for removed POST routes.
                self.assertEqual(self.client.post(route, json={}).status_code, 405)

    def test_factory_uses_current_config_static_path_and_always_registers_platform(self):
        with patch.dict(BACKEND_CONFIG['FRONTEND'], STATIC_DIR=self.root):
            app = create_app(self.platform)
        self.assertEqual(Path(app.static_folder), self.root)
        client = app.test_client()
        with client.get('/') as response:
            self.assertEqual(response.text, 'upstream analysis')
        self.assertEqual(client.get('/api/platform/config').status_code, 200)
        self.assertEqual(client.get('/api/platform/play/config').status_code, 200)

    def test_mate_distance_uses_side_to_move_but_cp_uses_white(self):
        self.platform.strategy = 'exhaustive'
        board = chess.Board('7k/8/8/8/8/8/8/R5K1 b - - 0 1')
        infos = [dict(depth=12, pv=[move], score=chess.engine.PovScore(chess.engine.Mate(1), chess.BLACK))
                 for move in board.legal_moves]
        search = Mock()
        search.__enter__ = Mock(return_value=iter(infos))
        search.__exit__ = Mock(return_value=False)
        self.stockfish._get_engine.return_value.analysis.return_value = search
        result = json.loads(list(self.platform.stream_stockfish(board, 12))[-1])
        self.assertTrue(all(cp == -10000 for cp in result['cp_vec'].values()))
        self.assertTrue(all(mate == 1 for mate in result['mate_vec'].values()))

    def test_terminal_position_requires_no_stockfish_search(self):
        board = chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
        result = json.loads(next(self.platform.stream_stockfish(board, 18)))
        self.assertEqual(result['terminal_cp'], 10000)
        self.assertEqual(result['mate_vec'], {'':0})
        self.assertTrue(result['is_checkmate'])
        self.stockfish._get_engine.assert_not_called()

    def test_exhaustive_mode_does_not_restore_staged_depth_as_exhaustive(self):
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}/analysis'
        saved = [{'ply':0, 'fen':chess.STARTING_FEN, 'maia':{},
                  'stockfish':{'depth':18, 'strategy':'staged', 'cp_vec':{'e2e4':25},
                               'root_move_depth_vec':{'e2e4':18, 'a2a3':10}}}]
        self.client.post(url, json=saved)
        self.platform.strategy = 'exhaustive'
        self.assertNotIn('stockfish', self.client.get(url).json['positions'][0])
        self.platform.strategy = 'staged'
        self.assertEqual(self.client.get(url).json['positions'], saved)

    def test_search_budget_configuration_and_bounded_cache_is_not_exhaustive(self):
        self.platform.strategy = 'bounded'
        config = self.client.get('/api/platform/config').json
        self.assertEqual(config['budgets'], {'12':2.0, '15':5.0, '18':10.0})
        self.assertEqual(config['max_budgets'], {'12':6.0, '15':15.0, '18':30.0})
        self.assertEqual(config['strategy'], 'bounded')
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}/analysis'
        saved = [{'ply':0, 'fen':chess.STARTING_FEN, 'maia':{},
                  'stockfish':{'depth':14, 'target_depth':18, 'complete':True, 'strategy':'bounded',
                               'stop_reason':'time', 'budget_seconds':10., 'policy_version':BOUNDED_POLICY_VERSION, 'cp_vec':{'e2e4':25}, 'root_move_depth_vec':{'e2e4':14}}}]
        self.client.post(url, json=saved)
        self.assertEqual(self.client.get(url).json['positions'], saved)
        self.platform.strategy = 'exhaustive'
        self.assertNotIn('stockfish', self.client.get(url).json['positions'][0])

    def test_completed_search_is_cached_and_invalid_options_rejected(self):
        self.platform.strategy = 'exhaustive'
        engine, search = Mock(), Mock()
        infos = [dict(depth=12, pv=[move], score=chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE))
                 for move in chess.Board().legal_moves]
        search.__enter__ = Mock(return_value=iter(infos))
        search.__exit__ = Mock(return_value=False)
        engine.analysis.return_value = search
        self.stockfish._get_engine.return_value = engine
        payload = {'fen':chess.STARTING_FEN, 'depth':12}
        first = self.client.post('/api/platform/stockfish', json=payload)
        frames = [json.loads(line) for line in first.text.splitlines()]
        self.assertTrue(frames[-1]['complete'])
        second = self.client.post('/api/platform/stockfish', json=payload)
        self.assertEqual([json.loads(line) for line in second.text.splitlines()], [frames[-1]])
        self.assertEqual(engine.analysis.call_count, 1)
        for options in ([], {'maiaCandidateMoves':['e2e5']}, {'forcedCandidateMoves':'e2e4'}):
            self.assertEqual(self.client.post('/api/platform/stockfish', json={**payload, 'options':options}).status_code, 400)

    def test_changed_config_budget_does_not_restore_old_bounded_analysis(self):
        self.platform.strategy = 'bounded'
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}/analysis'
        saved = [{'ply':0,'fen':chess.STARTING_FEN,'stockfish':{
            'strategy':'bounded','depth':14,'target_depth':18,
            'budget_seconds':self.platform.analysis_limits(18).verify_ms / 1000,
            'policy_version':BOUNDED_POLICY_VERSION,
            'complete':True,'cp_vec':{'e2e4':25}}}]
        self.client.post(url,json=saved)
        self.assertEqual(self.client.get(url).json['positions'],saved)
        with patch.dict(BACKEND_CONFIG['FRONTEND'], STOCKFISH_TIME_SCALE_BY_DEPTH={12:.2,15:.5,18:.8}):
            self.assertNotIn('stockfish',self.client.get(url).json['positions'][0])
        self.assertEqual(self.client.get(url).json['positions'],saved)
        self.assertEqual(self.platform.search_controls, {})

    def test_preset_scales_default_maximum_and_frontend_depth_ceiling(self):
        self.platform.strategy = 'bounded'
        for depth, seconds, maximum in ((12, 2, 6), (15, 5, 15), (18, 10, 30)):
            limits = self.platform.analysis_limits(depth)
            self.assertEqual((limits.depth, limits.verify_ms, limits.max_ms),
                             (depth, seconds * 1000, maximum * 1000))
            self.assertEqual(self.platform.analysis_limits(depth, maximum).verify_ms, maximum * 1000)
            with self.assertRaises(ValueError):
                self.platform.analysis_limits(depth, maximum + .001)
        with patch.dict(BACKEND_CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], MAX_DEPTH=15):
            config = self.client.get('/api/platform/config').json
            self.assertEqual(config['default_depth'], 15)
            self.assertEqual(set(config['budgets']), {'12', '15'})
            self.assertEqual(self.platform.analysis_limits().depth, 15)
            with self.assertRaises(ValueError):
                self.platform.analysis_limits(18)

    def test_position_requests_forward_scaled_time_and_enforce_scaled_maximum(self):
        with patch.object(self.platform, 'stream_stockfish', return_value=iter(['{}\n'])) as stream:
            response = self.client.post('/api/platform/stockfish', json={'fen': chess.STARTING_FEN, 'depth': 12})
            self.assertEqual(response.status_code, 200)
            response.get_data()
            self.assertEqual(stream.call_args.kwargs['seconds'], 2.)
            stream.reset_mock()
            self.assertEqual(self.client.post('/api/platform/stockfish', json={
                'fen': chess.STARTING_FEN, 'depth': 12, 'seconds': 6.001}).status_code, 400)
            stream.assert_not_called()

    def test_old_policy_and_changed_maximum_invalidate_saved_full_analysis(self):
        self.platform.strategy = 'bounded'
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}/analysis'
        position = {'ply': 0, 'fen': chess.STARTING_FEN, 'stockfish': {
            'strategy': 'bounded', 'depth': 12, 'target_depth': 12, 'cp_vec': {'e2e4': 25},
            'budget_seconds': 2., 'max_budget_seconds': 6., 'policy_version': BOUNDED_POLICY_VERSION}}
        self.platform.repository.save_full_analysis(game_id, {'positions': [position]})
        self.assertIsNotNone(self.client.get(url).json['analysis'])
        with patch.dict(BACKEND_CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], MAX_SEARCH_SECONDS=20.):
            saved = self.client.get(url).json
            self.assertNotIn('stockfish', saved['positions'][0])
            self.assertIsNone(saved['analysis'])
        position['stockfish']['policy_version'] = BOUNDED_POLICY_VERSION - 1
        self.client.post(url, json=[position])
        self.assertNotIn('stockfish', self.client.get(url).json['positions'][0])

    def test_invalid_saved_analysis_cannot_replace_readable_cache(self):
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}/analysis'
        saved = [{'ply': 0, 'stockfish': {'depth': 12, 'cp_vec': {'e2e4': 20}}}]
        self.assertEqual(self.client.post(url, json=saved).status_code, 200)
        invalid = [None, {}, [None], [3], [{'stockfish': None}], [{'stockfish': []}],
                   [{'stockfish': {'depth': True, 'cp_vec': {}}}],
                   [{'stockfish': {'target_depth': '18', 'cp_vec': {}}}],
                   [{'stockfish': {'budget_seconds': float('nan'), 'cp_vec': {}}}],
                   [{'stockfish': {}}], [{'stockfish': {'cp_vec': None}}],
                   [{'stockfish': {'cp_vec': {'e2e4': float('inf')}}}],
                   [{'maia': None}], [{'maia': {'maia_kdd_1500': None}}],
                   [{'maia': {'maia_kdd_1500': {}}}],
                   [{'maia': {'maia_kdd_1500': {'policy': None}}}],
                   [{'maia': {'maia_kdd_1500': {'policy': []}}}],
                   [{'maia': {'maia_kdd_1500': {'policy': {'e2e4': 'invalid'}}}}]]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post(url, data=json.dumps(payload), content_type='application/json')
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(self.client.get(url).json['positions'], saved)

    def test_saved_analysis_accepts_partial_rows_and_empty_terminal_vectors(self):
        game_id = self.save(pgn='1. f3 e5 2. g4 Qh4# 0-1')
        url = f'/api/platform/games/{game_id}/analysis'
        rows = [
            {'ply': 0, 'maia': {}},
            {'ply': 1, 'maia': {'maia_kdd_1500': {'policy': {'e7e5': .7}, 'value': .4}}},
            {'ply': 2, 'stockfish': {'depth': 12, 'cp_vec': {'g2g4': -500}}},
            {'ply': 3},
            {'ply': 4, 'maia': {'maia_kdd_1500': {'policy': {}, 'value': 0}},
             'stockfish': {'depth': 18, 'terminal_cp': -10000, 'cp_vec': {}, 'mate_vec': {'': 0}}},
        ]
        self.assertEqual(self.client.post(url, json=rows).status_code, 200)
        self.assertEqual(self.client.get(url).json['positions'], rows)

    def test_deletion_during_analysis_save_returns_not_found_without_orphaned_cache(self):
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}/analysis'
        repository = self.platform.repository
        connect = repository.connect

        @contextmanager
        def delete_before_write():
            with connect() as db:
                class Connection:
                    def execute(self, sql, parameters=()):
                        if sql.startswith('INSERT OR REPLACE INTO analyses'):
                            # A second connection removes the study immediately
                            # before publication, after any separate existence check.
                            with connect() as other:
                                other.execute('DELETE FROM games WHERE id=?', (game_id,))
                        return db.execute(sql, parameters)
                yield Connection()

        rows = [{'ply': 0, 'stockfish': {'cp_vec': {'e2e4': 25}}}]
        with patch.object(repository, 'connect', side_effect=delete_before_write):
            response = self.client.post(url, json=rows)
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(repository.load_analysis(game_id), [])
        self.assertEqual(self.client.post(url, json=rows).status_code, 404)

    def test_stockfish_cache_distinguishes_repetition_history_at_same_fen(self):
        history = chess.Board()
        for move in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
            history.push_uci(move)
        position = chess.Board(history.fen())

        def evaluate(engine, board, *args, **kwargs):
            yield {'complete': True, 'history_length': len(board.move_stack)}

        with patch('backend.analysis_positions.stream_evaluations', side_effect=evaluate) as search:
            for board, expected in ((history, 4), (position, 0), (history, 4), (position, 0)):
                result = json.loads(list(self.platform.stream_stockfish(board, 12))[-1])
                self.assertEqual(result['history_length'], expected)
            self.assertEqual(search.call_count, 2)

    def test_games_page_filters_and_counts_are_preserved_in_repository(self):
        for index in range(27):
            game_id = self.save(name=f'Game {index}', fen=chess.STARTING_FEN)
            if index % 2 == 0:
                self.client.patch(f'/api/platform/games/{game_id}', json={'is_favorited': True})
        first = self.client.get('/api/platform/games?page=1').json
        second = self.client.get('/api/platform/games?page=2').json
        self.assertEqual((first['total_games'], first['total_pages']), (27, 2))
        self.assertEqual([len(first['games']), len(second['games'])], [25, 2])
        self.assertTrue({row['game_id'] for row in first['games']}.isdisjoint(
            row['game_id'] for row in second['games']))
        self.assertEqual(self.client.get('/api/platform/games?type=favorites').json['total_games'], 14)
        for kind in ('play', 'unknown'):
            self.assertEqual(self.client.get(f'/api/platform/games?type={kind}').json,
                             {'total_games': 0, 'total_pages': 1, 'games': []})


if __name__ == '__main__':
    unittest.main()
