"""Offline local play, full-history Maia reuse, persistence and retry contracts."""
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import chess

from backend.app import create_app
from backend.analysis_positions import PlatformAnalysis
from backend.play import sample_move
from backend.settings import CONFIG


class LocalPlayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'index.html').write_text('local play')
        self.maia, self.stockfish = Mock(), Mock()
        self.preferred = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
        def prediction(fens, ratings, opponents, *, boards):
            result = []
            for board in boards:
                legal = sorted(m.uci() for m in board.legal_moves)
                chosen = next((m for m in self.preferred if m in legal), legal[0])
                policy = {m: .9 if m == chosen else .1 / max(1, len(legal)-1) for m in legal}
                result.append({'policy': policy, 'value': .5})
            return result
        self.maia.batch_evaluate.side_effect = prediction
        self.platform = PlatformAnalysis(self.maia, self.stockfish, self.root / 'games.sqlite3')
        self.app = create_app(self.platform, self.root)
        self.client = self.app.test_client()
        self.addCleanup(lambda: self.assertEqual(self.stockfish.mock_calls, []))

    def start(self, **options):
        response = self.client.post('/api/platform/play/games', json={
            'player_color': 'white', 'sample_moves': False, **options})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json

    def move(self, game_id, moves, **options):
        response = self.client.post('/api/platform/play/move', json={'game_id': game_id, 'moves': moves, **options})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json

    def submit(self, game_id, moves, times=None, **options):
        response = self.client.post(f'/api/platform/play/games/{game_id}/moves', json={
            'moves': moves, 'move_times': [0] * len(moves) if times is None else times,
            'game_over_state': 'not_over', **options})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json

    def snapshot(self, game_id):
        return self.client.get(f'/api/platform/games/{game_id}').json['snapshot']

    def test_config_start_retry_and_no_fabricated_human_rating(self):
        with patch.dict(CONFIG['MAIA'], PLAYER_RATING=1777, TEMPERATURE=.25, MODEL='maia3-79m'):
            self.assertEqual(self.client.get('/api/platform/play/config').json,
                             {'player_rating': 1777, 'temperature': .25, 'model': 'maia3-79m'})
            first = self.start(request_id='start-retry')
            self.assertEqual(self.start(request_id='start-retry')['game_id'], first['game_id'])
            different = self.client.post('/api/platform/play/games', json={
                'player_color': 'black', 'sample_moves': False, 'request_id': 'start-retry'})
            self.assertEqual(different.status_code, 409)
        self.assertEqual(first['opponent_elo'], 1777)
        self.assertIsNone(first['player_elo'])
        snapshot = self.snapshot(first['game_id'])
        self.assertNotIn('WhiteElo', snapshot['headers'])
        self.assertEqual(snapshot['headers']['BlackElo'], '1777')
        self.assertEqual(snapshot['headers']['MaiaModel'], 'maia3-79m')
        listing = self.client.get('/api/platform/games?type=play').json
        self.assertEqual(listing['total_games'], 1)
        self.assertEqual(listing['games'][0]['game_id'], first['game_id'])
        self.client.patch(f"/api/platform/games/{first['game_id']}", json={'is_favorited': True})
        for kind in ('play', 'custom', 'favorites'):
            row = self.client.get(f'/api/platform/games?type={kind}').json['games'][0]
            self.assertEqual(row['game_type'], 'play')
            self.assertEqual(row['maia_name'], 'maia_kdd_1777')
            self.assertEqual(row['player_color'], 'white')
        for page in ('/play', '/play/maia'):
            with self.client.get(page) as response:
                self.assertEqual(response.text, 'local play')
        self.assertEqual(self.maia.mock_calls, [])

    def test_move_uses_same_maia_with_full_history_and_commits_before_logging(self):
        game_id = self.start(maia_rating=1900)['game_id']
        first = self.move(game_id, ['e2e4'])
        self.assertEqual(first['top_move'], 'e7e5')
        self.assertEqual(first['moves'], ['e2e4', 'e7e5'])
        args = self.maia.batch_evaluate.call_args
        self.assertEqual(args.args[1:], ([1900], [1900]))
        self.assertEqual([m.uci() for m in args.kwargs['boards'][0].move_stack], ['e2e4'])
        self.assertEqual(self.snapshot(game_id)['ucis'], first['moves'])
        self.assertEqual(self.client.post(f'/api/platform/play/games/{game_id}/analysis').json, {'game_id': game_id})
        self.assertTrue(self.move(game_id, ['e2e4'])['cached'])
        self.assertEqual(self.maia.batch_evaluate.call_count, 1)
        self.submit(game_id, first['moves'])
        stale = self.submit(game_id, ['e2e4'])
        self.assertTrue(stale['stale'])
        self.assertEqual(stale['moves'], first['moves'])
        self.assertEqual(self.snapshot(game_id)['ucis'], first['moves'])
        conflict = self.client.post(f'/api/platform/play/games/{game_id}/moves', json={
            'moves': ['e2e4', 'c7c5'], 'move_times': [0, 0]})
        self.assertEqual(conflict.status_code, 409)

    def test_black_and_nonstandard_fen_preserve_move_history(self):
        black = self.start(player_color='black')['game_id']
        self.assertEqual(self.move(black, [])['top_move'], 'e2e4')
        self.submit(black, ['e2e4', 'e7e5'])
        self.assertEqual(self.move(black, ['e2e4', 'e7e5'])['top_move'], 'g1f3')
        fen = 'r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R b KQkq - 9 23'
        game_id = self.start(start_fen=fen, player_color='white')['game_id']
        result = self.move(game_id, [], start_fen=fen)
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci(result['top_move']), board.legal_moves)
        supplied = self.maia.batch_evaluate.call_args.kwargs['boards'][0]
        self.assertEqual(supplied.root().fen(), fen)
        snapshot = self.snapshot(game_id)
        self.assertEqual(snapshot['headers']['FEN'], fen)
        self.assertEqual(snapshot['headers']['SetUp'], '1')
        self.assertEqual(snapshot['ucis'], [result['top_move']])

    def test_invalid_requests_do_not_run_maia_or_change_history(self):
        game_id = self.start()['game_id']
        for body in ({'moves': ['e2e5']}, {'moves': []}, {'moves': ['e2e4', 'e7e5']},
                     {'moves': ['e2e4'], 'start_fen': chess.Board().mirror().fen()}, {'moves': 'e2e4'}):
            response = self.client.post('/api/platform/play/move', json={'game_id': game_id, **body})
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.snapshot(game_id)['ucis'], [])
        self.assertEqual(self.maia.mock_calls, [])
        for option in ({'maia_rating': 599}, {'maia_rating': True}, {'time_control': '61+0'},
                       {'time_control': '0+0'}, {'sample_moves': 'true'}, {'start_fen': 'bad'}):
            self.assertEqual(self.client.post('/api/platform/play/games', json={
                'player_color': 'white', **option}).status_code, 400)

    def test_rules_terminal_is_persisted_without_extra_engine_calls(self):
        self.preferred = ['f7g7']
        fen = '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'
        game_id = self.start(player_color='black', start_fen=fen)['game_id']
        result = self.move(game_id, [])
        self.assertEqual(result['top_move'], 'f7g7')
        self.assertEqual(result['termination'], {'type': 'rules', 'winner': 'white', 'result': '1-0'})
        self.assertEqual(self.snapshot(game_id)['result'], '1-0')
        self.assertIsNone(self.move(game_id, ['f7g7'])['top_move'])
        self.assertEqual(self.move(game_id, [])['top_move'], 'f7g7')
        self.assertEqual(self.maia.batch_evaluate.call_count, 1)
        for _ in range(2):
            self.submit(game_id, ['f7g7'], game_over_state='rules', winner='white')
        stale = self.submit(game_id, [], game_over_state='resign', winner='black')
        self.assertEqual(stale['termination']['winner'], 'white')
        self.assertEqual(self.client.get('/api/platform/play/stats').json['play_games_played'], 1)

    def test_times_resignation_and_timeout_are_saved_once(self):
        game_id = self.start(time_control='5+2')['game_id']
        self.move(game_id, ['e2e4'])
        self.submit(game_id, ['e2e4', 'e7e5', 'g1f3'], times=[0, 0, 1500])
        self.move(game_id, ['e2e4', 'e7e5', 'g1f3'])
        moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
        result = self.submit(game_id, moves, times=[0, 0, 1500, 2300])
        self.assertEqual(result['clocks'], {'white_ms': 300500, 'black_ms': 299700})
        ended = self.submit(game_id, moves, times=[0, 0, 1500, 2300], game_over_state='time', winner='black')
        self.assertEqual(ended['clocks']['white_ms'], 0)
        self.assertEqual(ended['termination']['result'], '0-1')
        resigned = self.start(player_color='black')['game_id']
        self.submit(resigned, [], game_over_state='resign', winner='white')
        stats = self.client.get('/api/platform/play/stats').json
        self.assertEqual(stats['play_games_played'], 2)
        self.assertIsNone(stats['play_elo'])
        self.assertEqual(stats['hand_games_played'], 0)
        self.assertEqual(self.snapshot(game_id)['move_times'], [0, 0, 1500, 2300])
        unlimited = self.start()['game_id']
        self.assertEqual(self.client.post(f'/api/platform/play/games/{unlimited}/moves', json={
            'moves': [], 'move_times': [], 'game_over_state': 'time'}).status_code, 400)

    def test_concurrent_retries_and_independent_games_reuse_one_model(self):
        white = self.start()['game_id']
        black = self.start(player_color='black')['game_id']
        def request_move(game_id, moves):
            with self.app.test_client() as client:
                response = client.post('/api/platform/play/move', json={'game_id': game_id, 'moves': moves})
                self.assertEqual(response.status_code, 200, response.text)
                return response.json
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(request_move, white, ['e2e4']), pool.submit(request_move, white, ['e2e4']),
                    pool.submit(request_move, black, []), pool.submit(request_move, black, [])]
            results = [job.result() for job in jobs]
        self.assertEqual([r['top_move'] for r in results], ['e7e5', 'e7e5', 'e2e4', 'e2e4'])
        self.assertEqual(self.maia.batch_evaluate.call_count, 2)
        self.assertEqual(self.snapshot(white)['ucis'], ['e2e4', 'e7e5'])
        self.assertEqual(self.snapshot(black)['ucis'], ['e2e4'])

    def test_unknown_move_times_are_saved_and_later_measurements_fill_gaps(self):
        game_id = self.start()['game_id']
        self.move(game_id, ['e2e4'])
        result = self.submit(game_id, ['e2e4', 'e7e5'], times=[None, None])
        self.assertEqual(result['move_times'], [None, None])
        result = self.submit(game_id, ['e2e4', 'e7e5'], times=[0, 250])
        self.assertEqual(result['move_times'], [0, 250])
        for invalid in (-1, True, '250', float('inf'), 86400001):
            response = self.client.post(f'/api/platform/play/games/{game_id}/moves', json={
                'moves': ['e2e4', 'e7e5'], 'move_times': [0, invalid]})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.snapshot(game_id)['move_times'], [0, 250])

    def test_increment_only_games_have_time_before_the_first_clocked_move(self):
        started = self.start(time_control='0+2')
        game_id = started['game_id']
        self.assertEqual(started['clocks'], {'white_ms': 2000, 'black_ms': 2000})
        self.move(game_id, ['e2e4'], initial_clock=0, current_clock=2)
        result = self.submit(game_id, ['e2e4', 'e7e5', 'g1f3'], times=[0, 0, 1500])
        self.assertEqual(result['clocks'], {'white_ms': 2500, 'black_ms': 2000})

    def test_timeout_cancels_only_unacknowledged_reply_including_undelivered_mate(self):
        self.preferred = ['e7e5', 'd8h4']
        game_id = self.start(time_control='1+0')['game_id']
        self.move(game_id, ['f2f3'])
        played = ['f2f3', 'e7e5', 'g2g4']
        reply = self.move(game_id, played)
        self.assertEqual(reply['top_move'], 'd8h4')
        self.assertEqual(reply['termination']['winner'], 'black')
        # The reply was selected, but the UI's clock expired before it could
        # apply the move. The pre-reply history identifies exactly that race.
        ended = self.submit(game_id, played, game_over_state='time', winner='white')
        self.assertFalse(ended['stale'])
        self.assertEqual(ended['moves'], played)
        self.assertEqual(ended['termination'], {'type': 'time', 'winner': 'white', 'result': '1-0'})
        self.assertEqual(ended['clocks']['black_ms'], 0)
        self.assertEqual(self.snapshot(game_id)['ucis'], played)
        self.assertEqual(self.snapshot(game_id)['result'], '1-0')
        self.assertIsNone(self.move(game_id, played)['top_move'])
        self.assertEqual(self.maia.batch_evaluate.call_count, 2)
        for _ in range(2):
            self.submit(game_id, played, game_over_state='time', winner='white')
        self.assertEqual(self.client.get('/api/platform/play/stats').json['play_games_played'], 1)

    def test_stale_timeout_cannot_retract_an_acknowledged_reply(self):
        game_id = self.start(time_control='1+0')['game_id']
        self.move(game_id, ['e2e4'])
        played = ['e2e4', 'e7e5', 'g1f3']
        result = self.move(game_id, played)
        self.submit(game_id, result['moves'], times=[None] * 4)
        stale = self.submit(game_id, played, game_over_state='time', winner='white')
        self.assertTrue(stale['stale'])
        self.assertEqual(stale['moves'], result['moves'])
        self.assertIsNone(stale['termination'])
        # A subsequent human move also acknowledges a reply without depending
        # on whether its earlier timing log has reached the server yet.
        later = result['moves'] + ['f1c4']
        self.move(game_id, later)
        stale = self.submit(game_id, played, game_over_state='time', winner='white')
        self.assertTrue(stale['stale'])
        self.assertEqual(len(stale['moves']), 6)
        self.assertIsNone(stale['termination'])

    def test_inference_cannot_commit_a_move_after_the_clock_expires(self):
        game_id = self.start(time_control='1+0')['game_id']
        # The first two plies are unclocked, even with a zero clock payload.
        self.move(game_id, ['e2e4'], initial_clock=60, current_clock=0)
        played = ['e2e4', 'e7e5', 'g1f3']
        with patch('backend.play.time.monotonic', side_effect=[100., 100., 102.]):
            result = self.move(game_id, played, initial_clock=60, current_clock=1)
        self.assertIsNone(result['top_move'])
        self.assertEqual(result['moves'], played)
        self.assertEqual(result['termination']['winner'], 'white')
        self.assertEqual(result['termination']['type'], 'time')
        self.assertEqual(self.snapshot(game_id)['ucis'], played)
        self.assertIsNone(self.move(game_id, played, current_clock=0)['top_move'])
        self.assertEqual(self.maia.batch_evaluate.call_count, 2)
        new_id = self.start(time_control='1+0')['game_id']
        response = self.client.post('/api/platform/play/move', json={
            'game_id': new_id, 'moves': ['e2e4'], 'current_clock': -1})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.snapshot(new_id)['ucis'], [])

    def test_reopen_analysis_cache_and_delete_cascade(self):
        game_id = self.start()['game_id']
        self.move(game_id, ['e2e4'])
        self.submit(game_id, ['e2e4', 'e7e5'])
        cache = [{'ply': 0, 'fen': chess.STARTING_FEN, 'stockfish': {'cp_vec': {'e2e4': 5}}}]
        self.client.post(f'/api/platform/games/{game_id}/analysis', json=cache)
        reopened = PlatformAnalysis(self.maia, self.stockfish, self.root / 'games.sqlite3')
        client = create_app(reopened, self.root).test_client()
        self.assertEqual(client.get(f'/api/platform/play/games/{game_id}').json['moves'], ['e2e4', 'e7e5'])
        self.assertEqual(client.get(f'/api/platform/games/{game_id}/analysis').json['positions'], cache)
        self.submit(game_id, ['e2e4', 'e7e5', 'g1f3'])
        self.assertEqual(self.client.get(f'/api/platform/games/{game_id}/analysis').json['positions'], [])
        self.assertEqual(client.delete(f'/api/platform/games/{game_id}').status_code, 200)
        self.assertEqual(client.get(f'/api/platform/play/games/{game_id}').status_code, 404)
        with reopened.repository.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM local_play').fetchone()[0], 0)

    def test_sampling_applies_temperature_without_mutating_policy(self):
        board = chess.Board()
        policy = {move.uci(): 0. for move in board.legal_moves}
        policy.update(e2e4=.8, d2d4=.2)
        with patch('backend.play.random.choices', return_value=['e2e4']) as choose:
            self.assertEqual(sample_move(policy, board, temperature=.5, sampled=True), 'e2e4')
            legal = choose.call_args.args[0]
            weights = choose.call_args.kwargs['weights']
            self.assertAlmostEqual(weights[legal.index('d2d4')] / weights[legal.index('e2e4')], 1/16)
            sample_move(policy, board, temperature=1e-320, sampled=True)
            self.assertTrue(all(math.isfinite(w) for w in choose.call_args.kwargs['weights']))
            self.assertEqual(max(choose.call_args.kwargs['weights']), 1.)
            choose.reset_mock()
            self.assertEqual(sample_move(policy, board, temperature=0, sampled=True), 'e2e4')
            self.assertEqual(sample_move(policy, board, temperature=.5, sampled=False), 'e2e4')
            choose.assert_not_called()
        self.assertEqual(policy['e2e4'], .8)
        self.assertEqual(policy['d2d4'], .2)


if __name__ == '__main__':
    unittest.main()
