"""Cold/warm local game analysis timing and score comparison; no coaching agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.game.pipeline import analyze_game
from analysis.position_evaluation import header_elo
from analysis.game.study import load_game
from analysis.engine_session import Engines
from analysis.cache import write_json
from coach.settings import CONFIG


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('pgn', type=Path)
    cli.add_argument('--output-dir', type=Path, required=True)
    cli.add_argument('--workers', type=int)
    cli.add_argument('--threads-per-worker', type=int, default=CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'])
    cli.add_argument('--cache-dir', type=Path, default=CONFIG['ANALYSIS']['CACHE_DIR'],
                     help='Override ANALYSIS.CACHE_DIR from config.yaml.')
    args = cli.parse_args(argv)
    game = load_game(args.pgn)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stages = {'maia_seconds':0., 'maia_calls':0, 'maia_rows':0, 'stockfish_seconds':0.}
    positions = []
    with Engines(game.board().fen(), args.cache_dir,analysis_workers=args.workers,
                 threads_per_worker=args.threads_per_worker) as engines:
        load_seconds = time.perf_counter()-started
        original_maia = engines.maia.batch_evaluate
        def maia(*a, **kw):
            tick = time.perf_counter()
            result = original_maia(*a, **kw)
            stages['maia_seconds'] += time.perf_counter()-tick
            stages['maia_calls'] += 1
            stages['maia_rows'] += len(a[0])
            return result
        engines.maia.batch_evaluate = maia
        original_scan = engines.initial_analysis
        def scan(history, *a, **kw):
            tick = time.perf_counter()
            result = original_scan(history, *a, **kw)
            elapsed = time.perf_counter()-tick
            positions.append({'ply':len(history)+1, 'wall_seconds':elapsed,
                'best_move':result['best_move'], 'search':result['search']})
            return result
        engines.initial_analysis = scan
        analysis_started = time.perf_counter()
        analysis = analyze_game(game, engines, 'white', header_elo(game.headers,'White'),
                                progress=lambda message:print(message,flush=True),
                                rating_output_dir=args.output_dir/'player-rating')
        elapsed = time.perf_counter()-analysis_started
        stats = dict(engines.stats)
        runtime = {'device':engines.signature['device'], **engines.last_analysis_execution}
        runtime['total_threads'] = runtime['workers'] * runtime['threads_per_worker']
    write_json(args.output_dir/'analysis.json',analysis)
    # Sum after all workers join; this is workload time, not game wall time.
    stages['stockfish_seconds'] = sum(position['wall_seconds'] for position in positions)
    write_json(args.output_dir/'timing.json',{'pgn':str(args.pgn),'plies':len(analysis['moves']),
        'cache_dir':str(args.cache_dir.resolve()),
        'load_seconds':load_seconds, 'analysis_seconds':elapsed, 'total_seconds':time.perf_counter()-started,
        'components':stages, 'engine_stats':stats, 'runtime':runtime, 'positions':sorted(positions,key=lambda p:p['ply']),
        'llm_calls':0,'llm_tokens':0})
    print(json.dumps({'analysis_seconds':elapsed,'components':stages,'runtime':runtime}),flush=True)


if __name__=='__main__':
    main()
