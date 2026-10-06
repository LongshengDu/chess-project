"""Analyze new PGNs locally with one persistent Maia/Stockfish session.

Uses unchanged project analysis limits and the default shared cache. Existing
full-game outputs are refused. No coach, language model, or server is started.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import chess.pgn

from analysis.cache import write_json
from analysis.engine_session import Engines, Limits
from analysis.game.pipeline import analyze_game
from analysis.settings import CONFIG


def run(paths, timing_output):
    prepared = []
    for path in paths:
        path = Path(path).resolve()
        with path.open(encoding='utf-8-sig') as stream:
            game = chess.pgn.read_game(stream)
        if game is None or game.errors or not list(game.mainline_moves()):
            raise ValueError(f'{path.name} is not a valid played game.')
        output = path.parent/'output'/f'{path.stem}-full'
        if (output/'analysis.json').exists():
            raise FileExistsError(f'Full analysis already exists: {output}.')
        prepared.append((path, game, output))
    if not prepared or len({game.board().fen() for _, game, _ in prepared}) != 1:
        raise ValueError('A batch must contain games with the same starting position.')
    timing_output = Path(timing_output)
    timing_output.parent.mkdir(parents=True, exist_ok=True)
    record = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'analysis_config': json.loads(json.dumps(CONFIG['ANALYSIS'], default=str)),
              'maia_config': json.loads(json.dumps(CONFIG['MAIA'], default=str)),
              'cache': str(Path(CONFIG['ANALYSIS']['CACHE_DIR']).resolve()), 'games': [],
              'engines_closed': False}
    start = time.perf_counter()
    try:
        print('Loading local engines once for the new-game batch.', flush=True)
        with Engines(prepared[0][1].board().fen(), CONFIG['ANALYSIS']['CACHE_DIR'], limits=Limits()) as engines:
            record['engine_load_seconds'] = time.perf_counter()-start
            record['engine_signature'] = engines.signature
            record['maia_device'] = str(engines.signature['device'])
            print(f'Maia device: {record["maia_device"]}', flush=True)
            for path, game, output in prepared:
                began = time.perf_counter()
                previous = began
                before = engines.stats.copy()
                events = []
                def progress(message):
                    nonlocal previous
                    now = time.perf_counter()
                    events.append({'seconds': now-began, 'message': message})
                    if not message.startswith('Analyzed ') or now-previous >= 20:
                        print(f'{path.stem} [{now-began:.1f}s] {message}', flush=True)
                        previous = now
                output.mkdir(parents=True, exist_ok=True)
                (output/'game.pgn').write_text(path.read_text(encoding='utf-8-sig'), encoding='utf-8')
                print(f'{path.stem}: starting {len(list(game.mainline_moves()))} played plies.', flush=True)
                analysis = analyze_game(game, engines, side=None, actual_elo=None, progress=progress,
                                        rating_output_dir=output/'player-rating')
                write_json(output/'analysis.json', analysis)
                entry = {'game': path.stem, 'pgn': str(path), 'output': str(output),
                         'plies': len(analysis['moves']), 'seconds': time.perf_counter()-began,
                         'execution': analysis['analysis_execution'],
                         'ratings': analysis['played_elo'],
                         'account_overrides': analysis['rating_account_overrides'],
                         'engine_statistics': {key: engines.stats[key]-before[key] for key in before},
                         'progress': events}
                record['games'].append(entry)
                write_json(timing_output, record)
                print(f'{path.stem}: complete in {entry["seconds"]:.1f}s; {output}', flush=True)
    finally:
        # The Engines context manager closes workers even if analysis fails.
        record['engines_closed'] = True
        record['elapsed_seconds'] = time.perf_counter()-start
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(timing_output, record)
        print('Batch engine context closed.', flush=True)
    return record


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('pgn', nargs='+', type=Path)
    cli.add_argument('--timing-output', type=Path,
                     default=Path(__file__).resolve().parent/'output/full-analysis-batch/run.json')
    args = cli.parse_args()
    run(args.pgn, args.timing_output)
