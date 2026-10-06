"""Exercise native web routes concurrently without starting a web server."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import create_app
from backend.analysis_positions import PlatformAnalysis
from backend.settings import CONFIG as BACKEND_CONFIG
from analysis.game.study import load_game
from engine.uci import stockfish_executable
from analysis.cache import write_json
from engine.maia import MaiaPolicy
from coach.settings import CONFIG
from engine.stockfish import StockfishScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pgn', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    game = load_game(args.pgn)
    board = game.board()
    start_fen = board.fen()
    positions, histories, moves = [], {}, []
    for move in list(game.mainline_moves())[:4]:
        positions.append((board.copy(), move.uci()))
        histories[board.fen()] = list(moves)
        moves.append(move.uci())
        board.push(move)
    maia = MaiaPolicy(CONFIG['MAIA']['MODEL'], CONFIG['MAIA']['CACHE_DIR'],
                      CONFIG['MAIA']['DEVICE'], checkpoint=CONFIG['MAIA']['CHECKPOINT'])
    maia.load()
    scorer = StockfishScorer(stockfish_executable(CONFIG['STOCKFISH']['EXECUTABLE'], CONFIG['STOCKFISH']['CACHE_DIR']))
    try:
        with tempfile.TemporaryDirectory(prefix='chess-web-benchmark-') as temp:
            platform = PlatformAnalysis(maia, scorer, Path(temp)/'analysis.sqlite3')
            app = create_app(platform, BACKEND_CONFIG['FRONTEND']['STATIC_DIR'])
            app.testing = True
            ratings = list(range(600, 2601, 100))
            payload = {
                'fens': [b.fen() for b, _ in positions for _ in ratings],
                'ratings': ratings*len(positions), 'opponents': ratings*len(positions),
                'start_fen': start_fen, 'histories': histories,
            }
            with app.test_client() as client:
                config = client.get('/api/platform/config').get_json()
                tick = time.perf_counter()
                response = client.post('/api/platform/maia', json=payload)
                maia_seconds = time.perf_counter()-tick
                assert response.status_code == 200, response.get_data(as_text=True)
                predictions = response.get_json()['result']
                assert len(predictions) == len(payload['fens'])
                for index, (position, _) in enumerate(positions):
                    for prediction in predictions[index*len(ratings):(index+1)*len(ratings)]:
                        assert set(prediction['policy']) == {m.uci() for m in position.legal_moves}
                        assert abs(sum(prediction['policy'].values())-1) < 1e-5

            def search(item):
                index, (position, played) = item
                policy = predictions[index*len(ratings)+10]['policy']
                request = {'fen': position.fen(), 'depth': 12, 'options': {
                    'maiaCandidateMoves': sorted(policy, key=policy.get, reverse=True)[:4],
                    'forcedCandidateMoves': [played]}}
                with app.test_client() as client:
                    response = client.post('/api/platform/stockfish', json=request)
                    assert response.status_code == 200
                    frames = [json.loads(line) for line in response.get_data(as_text=True).splitlines()]
                    assert frames and not any('error' in frame for frame in frames), frames
                    final = frames[-1]
                    assert final['complete'] is True
                    assert set(final['cp_vec']) | set(final['mate_vec']) >= {m.uci() for m in position.legal_moves}
                    return final

            rounds = []
            for _ in range(2):
                tick = time.perf_counter()
                with ThreadPoolExecutor(max_workers=config['analysis_workers']) as executor:
                    results = list(executor.map(search, enumerate(positions)))
                rounds.append(time.perf_counter()-tick)
                assert not platform.search_controls
            report = {'pgn': str(args.pgn), 'positions': len(positions),
                      'maia_rows': len(predictions), 'maia_seconds': maia_seconds,
                      'device': maia._engine.cfg.device,
                      'workers': scorer.analysis_pool.workers,
                      'threads_per_worker': scorer.analysis_pool.threads_per_worker,
                      'cold_stockfish_seconds': rounds[0], 'warm_stockfish_seconds': rounds[1],
                      'all_positions_complete': all(r['complete'] for r in results),
                      'all_legal_moves_scored': True, 'llm_calls': 0,
                      'server_started': False}
            write_json(args.output, report)
            print(json.dumps(report, indent=2))
    finally:
        scorer.close()


if __name__ == '__main__':
    main()
