from analysis.stockfish_search import BOUNDED_POLICY_VERSION
from analysis.session import AnalysisSession, Limits
from analysis.cache.positions import PositionCache
from analysis.cache.requests import maia_request, stockfish_initial_request
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
from tests.analysis.test_stockfish_search import evaluation_frame


def prediction(fen=chess.STARTING_FEN, value=.5):
    board = chess.Board(fen)
    return {'policy': {move.uci(): 1 / board.legal_moves.count() for move in board.legal_moves},
            'value': value}


class PlatformAnalysisTests(unittest.TestCase):
    def setUp(self):
        presets = patch.dict(BACKEND_CONFIG['FRONTEND'], STOCKFISH_TIME_SCALE_BY_DEPTH={12: .2, 15: .5, 18: 1.})
        presets.start()
        self.addCleanup(presets.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enterContext(patch.dict(BACKEND_CONFIG['ANALYSIS'], CACHE_DIR=self.root/'evidence'))
        (self.root / 'index.html').write_text('upstream analysis')
        self.maia = Mock()
        self.stockfish = Mock()
        self.maia.model_signature = {'test_maia': 1}
        self.stockfish.executable = __file__
        self.stockfish.threads_per_worker = 1
        self.stockfish.hash_mb = 16
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
        self.maia.batch_evaluate.return_value = [prediction()] * 21
        data = {'fens':[chess.STARTING_FEN]*21, 'ratings':ratings, 'opponents':ratings}
        for _ in range(2):
            result = self.client.post('/api/platform/maia', json=data)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(len(result.json['result']), 21)
        self.assertEqual(self.maia.batch_evaluate.call_count, 1)
        self.assertEqual(self.maia.batch_evaluate.call_args.args, (data['fens'], ratings, ratings))
        for patch in ({'ratings':[599]*21}, {'opponents':[]}, {'fens':['bad']*21}, {'fens':[]}):
            self.assertEqual(self.client.post('/api/platform/maia', json={**data, **patch}).status_code, 400)

    def test_multi_position_maia_batch_preserves_mainline_histories(self):
        board=chess.Board()
        fens=[board.fen()]; board.push_uci('e2e4'); fens.append(board.fen())
        board.push_uci('e7e5'); fens.append(board.fen())
        request={'fens':[fen for fen in fens for _ in range(21)],'ratings':list(range(600,2601,100))*3,
                 'opponents':[1500]*63,'start_fen':chess.STARTING_FEN,
                 'histories':dict(zip(fens,[[],['e2e4'],['e2e4','e7e5']]))}
        self.maia.batch_evaluate.return_value=[prediction(fen) for fen in request['fens']]
        response=self.client.post('/api/platform/maia',json=request)
        self.assertEqual(response.status_code,200,response.text)
        passed=self.maia.batch_evaluate.call_args.kwargs['boards']
        self.assertEqual([len(passed[i].move_stack) for i in (0,21,42)],[0,1,2])
        request['histories'][fens[-1]]=['e2e4']
        self.assertEqual(self.client.post('/api/platform/maia',json=request).status_code,400)

    def test_maia_reuses_individual_rating_pairs_after_restart_and_only_infers_missing_pairs(self):
        self.maia.batch_evaluate.side_effect = lambda fens, own, opponent, **kwargs: [
            prediction(value=rating / 3000) for rating in own]
        request = {'fens': [chess.STARTING_FEN] * 2, 'ratings': [1500, 1600], 'opponents': [1500, 1500]}
        expected = self.platform.evaluate_maia(request)['result']
        reopened = PlatformAnalysis(self.maia, self.stockfish, self.root/'studies.sqlite3')
        self.assertEqual(reopened.evaluate_maia({'fens': [chess.STARTING_FEN], 'ratings': [1600],
                                                'opponents': [1500]})['result'], expected[1:])
        self.assertEqual(self.maia.batch_evaluate.call_count, 1)
        values = reopened.evaluate_maia({**request, 'ratings': [1600, 1700]})['result']
        self.assertEqual(values[0], expected[1])
        self.assertEqual(self.maia.batch_evaluate.call_args.args[1:], ([1700], [1500]))
        self.assertEqual(len(list((self.root/'evidence'/'positions').glob('*.json'))), 1)

    def test_running_web_reads_refreshed_maia_measurement_instead_of_memory(self):
        board = chess.Board()
        self.maia.batch_evaluate.return_value = [prediction(value=.4)]
        self.assertEqual(self.platform.maia_predictions([board], [1600], [1600])[0]['value'], .4)
        self.assertTrue(self.platform.maia_cache)
        cache = PositionCache(self.root/'evidence')
        request = maia_request(self.maia.model_signature, 1600, 1600)
        pinned = cache.resolve_reference(board, 'maia', request)
        cache.put(board, 'maia', request, prediction(value=.7))
        self.maia.batch_evaluate.side_effect = AssertionError('Refreshed disk evidence must be reused')
        self.assertEqual(self.platform.maia_predictions([board], [1600], [1600])[0]['value'], .7)
        self.assertEqual(cache.get_reference(pinned)['value'], .4)

    def test_running_web_reads_refreshed_stockfish_measurement_instead_of_memory(self):
        board = chess.Board()
        self.platform.strategy = 'bounded'
        frame = evaluation_frame(board, depth=12)
        with patch('backend.analysis_positions.stream_evaluations', return_value=iter([frame])):
            original = json.loads(list(self.platform.stream_stockfish(board, 12))[-1])
        self.assertTrue(self.platform.stockfish_cache)
        limits = self.platform.analysis_limits(12)
        request = stockfish_initial_request(self.platform.stockfish_signature, 12,
            limits.verify_ms / 1000, limits.max_ms / 1000, 'bounded', {})
        cache = PositionCache(self.root/'evidence')
        pinned = cache.resolve_reference(board, 'stockfish', request)
        refreshed = {**original, 'cp_vec': {**original['cp_vec'], 'e2e4': 35}}
        cache.put(board, 'stockfish', request, refreshed)
        with patch.object(self.platform, 'search_engine', side_effect=AssertionError('Must reuse disk evidence')):
            current = json.loads(list(self.platform.stream_stockfish(board, 12))[-1])
        self.assertEqual(current, refreshed)
        self.assertEqual(cache.get_reference(pinned), original)

    def test_incomplete_stockfish_streams_never_enter_disk_or_memory_cache(self):
        board = chess.Board()
        complete = evaluation_frame(board, depth=12)
        for cold in (False, True):
            self.platform.profiler = Mock(active=True) if cold else None
            for partial in ({**complete, 'coverage_complete': False},
                            {**complete, 'cp_vec': {'e2e4': 20}},
                            {**complete, 'complete': False}):
                with self.subTest(cold=cold, partial=partial), patch(
                        'backend.analysis_positions.stream_evaluations',
                        side_effect=lambda *args, **kwargs: iter([partial.copy()])) as search:
                    for _ in range(2):
                        self.assertTrue(list(self.platform.stream_stockfish(board, 12)))
                    self.assertEqual(search.call_count, 2)
                    self.assertFalse(self.platform.stockfish_cache)
                    self.assertEqual(list((self.root/'evidence'/'positions').glob('*.json')), [])

    def test_maia_same_fen_with_distinct_histories_shares_file_but_not_prediction(self):
        histories = [['g1f3', 'g8f6', 'b1c3', 'b8c6'], ['b1c3', 'b8c6', 'g1f3', 'g8f6']]
        boards = [chess.Board(), chess.Board()]
        for board, history in zip(boards, histories):
            for move in history:
                board.push_uci(move)
        self.assertEqual(boards[0].fen(), boards[1].fen())
        self.maia.batch_evaluate.return_value = [prediction(boards[0].fen(), .2), prediction(boards[1].fen(), .8)]
        payload = {'fens': [board.fen() for board in boards], 'ratings': [1600, 1600],
                   'opponents': [1600, 1600], 'start_fen': chess.STARTING_FEN, 'histories': histories}
        for _ in range(2):
            result = self.client.post('/api/platform/maia', json=payload)
            self.assertEqual([row['value'] for row in result.json['result']], [.2, .8])
        self.assertEqual(self.maia.batch_evaluate.call_count, 1)
        self.assertEqual([[move.uci() for move in board.move_stack]
                         for board in self.maia.batch_evaluate.call_args.kwargs['boards']], histories)
        files = list((self.root/'evidence'/'positions').glob('*.json'))
        self.assertEqual(len(files), 1)
        self.assertEqual(len(json.loads(files[0].read_text())['histories']), 2)

    def test_interactive_and_game_session_share_maia_and_stockfish_evidence(self):
        self.platform.strategy = 'bounded'
        session = AnalysisSession(chess.STARTING_FEN, self.root/'evidence',
                                  limits=self.platform.analysis_limits(12), stockfish_path=__file__)
        session.engines.signature = {**self.platform.stockfish_signature, **self.maia.model_signature}
        session.engines.maia = self.maia
        session.engines.stockfish = Mock()
        self.maia.batch_evaluate.side_effect = lambda fens, own, opponent, *, boards, **kwargs: [
            {'policy': {move.uci(): 1 / board.legal_moves.count() for move in board.legal_moves},
             'value': .5} for board in (boards.values() if isinstance(boards, dict) else boards)]
        policy = session.human_pairs(['e2e4'], [1600], [1600])
        board = chess.Board()
        board.push_uci('e2e4')
        interactive = self.platform.evaluate_maia({'fens': [board.fen()], 'ratings': [1600],
            'opponents': [1600], 'start_fen': chess.STARTING_FEN, 'moves': ['e2e4']})
        self.assertEqual(interactive['result'], policy)
        self.assertEqual(self.maia.batch_evaluate.call_count, 1)
        frame = {'complete': True, 'coverage_complete': True, 'depth': 12,
                 'cp_vec': {move.uci(): 20 for move in board.legal_moves}, 'mate_vec': {},
                 'root_move_depth_vec': {move.uci(): 12 for move in board.legal_moves},
                 'best_move': 'e7e5', 'engine_moves': ['e7e5'], 'strategy': 'bounded'}
        with patch('analysis.session.stream_evaluations', return_value=iter([frame])):
            saved = session.initial_analysis(['e2e4'], 'e7e5', ['e7e5'])
        payload = {'fen': board.fen(), 'start_fen': chess.STARTING_FEN, 'moves': ['e2e4'], 'depth': 12,
                   'options': {'forcedCandidateMoves': ['e7e5'], 'maiaCandidateMoves': ['e7e5'],
                               'maiaPolicy': {'e7e5': 1.}}}
        with patch.object(self.platform, 'search_engine', side_effect=AssertionError('must use shared cache')):
            response = self.client.post('/api/platform/stockfish', json=payload)
            self.assertEqual(response.status_code, 200)
            returned = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(len(returned), 1)
        self.assertEqual(returned[0]['best_move'], saved['best_move'])
        # A different compatible search written by the browser is reused by the session.
        with patch('backend.analysis_positions.stream_evaluations', return_value=iter([frame])):
            list(self.platform.stream_stockfish(board, 12, {'forcedCandidateMoves': ['c7c5']}))
        with patch.object(session, '_initial_engine', side_effect=AssertionError('must use browser cache')):
            reused = session.initial_analysis(['e2e4'], 'c7c5', [])
        self.assertEqual(reused['best_move'], 'e7e5')
        self.assertEqual(session.stats['stockfish_cache_hits'], 1)
        partial = {**frame, 'coverage_complete': False, 'cp_vec': {'e7e5': 20}}
        with patch('backend.analysis_positions.stream_evaluations', return_value=iter([partial])):
            list(self.platform.stream_stockfish(board, 12, {'forcedCandidateMoves': ['d7d5']}))
        with patch('analysis.session.stream_evaluations', return_value=iter([frame])) as search:
            session.initial_analysis(['e2e4'], 'd7d5', [])
        search.assert_called_once()
        self.assertEqual(session.stats['stockfish_cache_hits'], 1)

    def test_stockfish_http_validates_history_before_search(self):
        board = chess.Board()
        history = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
        for move in history:
            board.push_uci(move)
        payload = {'fen': board.fen(), 'start_fen': chess.STARTING_FEN, 'moves': history, 'depth': 12}
        with patch.object(self.platform, 'stream_stockfish', return_value=iter(['{}\n'])) as stream:
            self.client.post('/api/platform/stockfish', json=payload).get_data()
            self.assertEqual([move.uci() for move in stream.call_args.args[0].move_stack], history)
            stream.reset_mock()
            for values in ({'moves': []}, {'moves': ['e2e5']}, {'start_fen': None}, {'moves': 'bad'}):
                self.assertEqual(self.client.post('/api/platform/stockfish', json={**payload, **values}).status_code, 400)
            null_board = chess.Board()
            null_board.push(chess.Move.null())
            self.assertEqual(self.client.post('/api/platform/stockfish', json={**payload,
                'fen': null_board.fen(), 'moves': ['0000']}).status_code, 400)
            stream.assert_not_called()

    def test_cold_profiler_bypasses_persistent_evidence_without_replacing_it(self):
        payload = {'fens': [chess.STARTING_FEN], 'ratings': [1600], 'opponents': [1600]}
        self.maia.batch_evaluate.return_value = [prediction(value=.5)]
        original = self.platform.evaluate_maia(payload)['result']
        self.platform.maia_cache.clear()
        self.platform.profiler = Mock(active=True)
        self.maia.batch_evaluate.return_value = [prediction(value=.6)]
        self.assertNotEqual(self.platform.evaluate_maia(payload)['result'], original)
        self.assertEqual(self.maia.batch_evaluate.call_count, 2)
        self.assertEqual(self.platform.evaluate_maia(payload)['result'][0]['value'], .6)
        self.assertEqual(self.maia.batch_evaluate.call_count, 2)
        self.platform.profiler.active = False
        # Ending a cold run restores persistent evidence without clearing its memory.
        self.assertEqual(self.platform.evaluate_maia(payload)['result'], original)
        self.assertEqual(self.maia.batch_evaluate.call_count, 2)

    def test_invalid_maia_batch_cannot_publish_partial_or_nonfinite_shared_evidence(self):
        board = chess.Board()
        valid = prediction()
        invalid_values = [
            {'policy': {'e2e4': 1.}, 'value': .5},
            {**valid, 'value': float('nan')},
            {**valid, 'policy': {**valid['policy'], 'e2e4': True}},
            {**valid, 'policy': dict.fromkeys(valid['policy'], 0.)},
        ]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                self.maia.batch_evaluate.return_value = [valid, invalid]
                with self.assertRaises(RuntimeError):
                    self.platform.maia_predictions([board, board], [1500, 1600], [1500, 1500])
                self.assertEqual(list((self.root/'evidence'/'positions').glob('*.json')), [])
                self.assertFalse(self.platform.maia_cache)
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
        with patch('analysis.accuracy.service.refresh_saved_curve', side_effect=AssertionError('Position autosave must not run full-game aggregation')):
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
        raw = {'lines': [{'cp': 20, 'mate': None, 'depth': 12, 'pv_uci': ['e2e4']}]}
        with patch.dict(BACKEND_CONFIG['ANALYSIS']['STOCKFISH_EXPLORATION'],
                        DEFAULT_SEARCH_SECONDS=.625, MAX_DEPTH=16), \
             patch('backend.routes_analysis.search_lines',return_value=raw) as explore:
            response = self.client.post('/api/platform/explore',json={'fen':chess.STARTING_FEN})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(explore.call_args.kwargs,{'seconds':.625,'depth':16})

    def test_web_exploration_promotes_requested_limits_and_reuses_lower_achieved_depth(self):
        raw = {'lines': [{'cp': 20, 'mate': None, 'depth': 6, 'pv_uci': ['e2e4']}]}
        with patch('backend.routes_analysis.search_lines', return_value=raw) as explore:
            first = self.client.post('/api/platform/explore', json={
                'fen': chess.STARTING_FEN, 'depth': 18, 'seconds': .5})
            self.assertEqual(first.status_code, 200, first.text)
            second = self.client.post('/api/platform/explore', json={
                'fen': chess.STARTING_FEN, 'depth': 12, 'seconds': .75})
            self.assertEqual(second.status_code, 200, second.text)
            self.assertEqual(explore.call_args.kwargs, {'seconds': .75, 'depth': 18})
            self.assertEqual(second.json['target_depth'], 18)
            self.assertEqual(second.json['time_limit_seconds'], .75)
            self.assertEqual(second.json['depth'], 6)
            third = self.client.post('/api/platform/explore', json={
                'fen': chess.STARTING_FEN, 'depth': 15, 'seconds': .625})
            self.assertEqual(third.status_code, 200, third.text)
            self.assertEqual(explore.call_count, 2)
            self.assertEqual(third.json, second.json)

    def test_web_evaluation_executes_max_requested_depth_and_time_on_upgrade(self):
        self.platform.strategy = 'bounded'
        board = chess.Board()
        with patch('backend.analysis_positions.stream_evaluations',
                   side_effect=lambda *args, **kwargs: iter([evaluation_frame(board, depth=6)])) as search:
            first = json.loads(list(self.platform.stream_stockfish(board, 18, seconds=2))[-1])
            self.assertEqual(first['target_depth'], 18)
            second = json.loads(list(self.platform.stream_stockfish(board, 12, seconds=3))[-1])
            self.assertEqual(search.call_args.args[2], 18)
            self.assertEqual(search.call_args.kwargs['seconds'], 3)
            self.assertEqual((second['target_depth'], second['budget_seconds'], second['depth']), (18, 3, 6))
            third = json.loads(list(self.platform.stream_stockfish(board, 15, seconds=2.5))[-1])
            self.assertEqual(third, second)
            self.assertEqual(search.call_count, 2)

    def test_exploration_enforces_configured_maximum_instead_of_fixed_two_seconds(self):
        raw = {'lines': [{'cp': 20, 'mate': None, 'depth': 12, 'pv_uci': ['e2e4']}]}
        with patch('backend.routes_analysis.search_lines',return_value=raw) as explore:
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

    def test_web_child_exploration_and_coach_share_full_pv_in_both_directions(self):
        session = AnalysisSession(chess.STARTING_FEN, self.root/'evidence',
                                  limits=Limits(depth=18), stockfish_path=__file__)
        session.engines.signature = {**self.platform.stockfish_signature, **self.maia.model_signature}
        session.engines.stockfish = Mock()
        self.addCleanup(session.close)
        raw = {'lines': [{'cp': 23, 'mate': None, 'depth': 15,
                          'pv_uci': ['e7e5', 'g1f3', 'b8c6', 'f1b5']}]}
        with patch('backend.routes_analysis.search_lines', return_value=raw) as search:
            response = self.client.post('/api/platform/explore', json={
                'fen': chess.STARTING_FEN, 'move': 'e2e4', 'seconds': .5, 'depth': 18})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual([move.uci() for move in search.call_args.args[1].move_stack], ['e2e4'])
            self.assertEqual(response.json['pv_uci'], ['e2e4', *raw['lines'][0]['pv_uci']])
        with patch.object(session, '_initial_engine', side_effect=AssertionError('Must reuse web PV')):
            short = session.sf(['e2e4'], 500, multipv=1, pv_plies=2)
            longer = session.sf(['e2e4'], 500, multipv=1, pv_plies=4)
        self.assertEqual(short['lines'][0]['pv_uci'], raw['lines'][0]['pv_uci'][:2])
        self.assertEqual(longer['lines'][0]['pv_uci'], raw['lines'][0]['pv_uci'])

        other = {'lines': [{'cp': 15, 'mate': None, 'depth': 14,
                            'pv_uci': ['d7d5', 'c2c4', 'e7e6']}]}
        with patch('analysis.session.search_lines', return_value=other):
            result = session.sf(['d2d4'], 500, multipv=1, pv_plies=1)
        self.assertEqual(result['lines'][0]['pv_uci'], ['d7d5'])
        with patch.object(self.platform, 'search_engine', side_effect=AssertionError('Must reuse full coach PV')):
            response = self.client.post('/api/platform/explore', json={
                'fen': chess.STARTING_FEN, 'move': 'd2d4', 'seconds': .5, 'depth': 18})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json['pv_uci'], ['d2d4', *other['lines'][0]['pv_uci']])

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
        self.assertEqual(self.client.get('/api/platform/unknown').status_code, 404)


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
                  'stockfish':{'depth':14, 'target_depth':18, 'complete':True, 'coverage_complete':True, 'strategy':'bounded',
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

    def test_changed_preset_preserves_complete_snapshot_for_browser_dominance_check(self):
        self.platform.strategy = 'bounded'
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}/analysis'
        saved = [{'ply':0,'fen':chess.STARTING_FEN,'stockfish':{
            'strategy':'bounded','depth':14,'target_depth':18,
            'budget_seconds':self.platform.analysis_limits(18).verify_ms / 1000,
            'policy_version':BOUNDED_POLICY_VERSION,
            'complete':True,'coverage_complete':True,'cp_vec':{'e2e4':25}}}]
        self.client.post(url,json=saved)
        self.assertEqual(self.client.get(url).json['positions'],saved)
        with patch.dict(BACKEND_CONFIG['FRONTEND'], STOCKFISH_TIME_SCALE_BY_DEPTH={12:.2,15:.5,18:.8}):
            self.assertEqual(self.client.get(url).json['positions'], saved)
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

    def test_current_complete_snapshot_survives_changed_maximum_but_old_policy_does_not(self):
        self.platform.strategy = 'bounded'
        game_id = self.save(fen=chess.STARTING_FEN)
        url = f'/api/platform/games/{game_id}/analysis'
        position = {'ply': 0, 'fen': chess.STARTING_FEN, 'stockfish': {
            'strategy': 'bounded', 'depth': 12, 'target_depth': 12, 'cp_vec': {'e2e4': 25},
            'complete': True, 'coverage_complete': True,
            'budget_seconds': 2., 'max_budget_seconds': 6., 'policy_version': BOUNDED_POLICY_VERSION}}
        self.platform.repository.save_full_analysis(game_id, {'positions': [position]})
        self.assertIsNotNone(self.client.get(url).json['analysis'])
        with patch.dict(BACKEND_CONFIG['ANALYSIS']['STOCKFISH_EVALUATION'], MAX_SEARCH_SECONDS=20.):
            saved = self.client.get(url).json
            self.assertEqual(saved['positions'], [position])
            self.assertIsNotNone(saved['analysis'])
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
            yield {**evaluation_frame(board, depth=12, strategy='staged'),
                   'history_length': len(board.move_stack)}

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
