"""Analyze PGNs locally with one persistent Maia/Stockfish session.

Uses unchanged project analysis limits and the default shared cache. Existing
full-game outputs require --replace-output. No coach, language model, or server
is started, and cached measurements are never forcibly refreshed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from analysis.cache.storage import write_json
from analysis.session import AnalysisSession, Limits
from analysis.cache.session import CachedAnalysisSession
from analysis.cache.artifacts import AnalysisStore
from analysis.game.pipeline import analyze_game
from analysis.game.metadata import game_metadata
from analysis.game.study import load_game
from analysis.settings import CONFIG


def run(paths, timing_output, *, replace_output=False, rebuild_from_cache=False, check_cache=False,
        engine_signature=None):
    prepared = []
    for path in paths:
        path = Path(path).resolve()
        game = load_game(path)
        game_metadata(dict(game.headers), game=game)
        output = path.parent/'output'/f'{path.stem}-full'
        if (output/'analysis.json').exists() and not replace_output and not check_cache:
            raise FileExistsError(f'Full analysis already exists: {output}.')
        prepared.append((path, game, output))
    if not prepared or len({game.board().fen() for _, game, _ in prepared}) != 1:
        raise ValueError('A batch must contain games with the same starting position.')
    # Resolve and validate every game's complete evidence before touching any
    # output. Reuse these pinned sessions so reconstruction cannot switch heads.
    cached_sessions = []
    if rebuild_from_cache or check_cache:
        for path, game, _ in prepared:
            try:
                cached_sessions.append(CachedAnalysisSession(game, CONFIG['ANALYSIS']['CACHE_DIR'],
                    engine_signature=engine_signature))
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f'{path.name}: {exc}') from exc
        if check_cache:
            print(f'Cache preflight passed for {len(prepared)} games; no engines or outputs were created.', flush=True)
            return {'cache_only': True, 'checked_games': len(prepared),
                    'games': [str(path) for path, _, _ in prepared]}
    timing_output = Path(timing_output)
    timing_output.parent.mkdir(parents=True, exist_ok=True)
    record = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'analysis_config': json.loads(json.dumps(CONFIG['ANALYSIS'], default=str)),
              'maia_config': json.loads(json.dumps(CONFIG['MAIA'], default=str)),
              'cache': str(Path(CONFIG['ANALYSIS']['CACHE_DIR']).resolve()), 'games': [],
              'cache_only': rebuild_from_cache,
              'engines_closed': False}
    start = time.perf_counter()
    store = AnalysisStore(CONFIG['ANALYSIS']['CACHE_DIR'])
    try:
        with ExitStack() as contexts:
            if rebuild_from_cache:
                print('Rebuilding saved observations only; engines disabled.', flush=True)
                record['engine_load_seconds'] = 0.
                shared_session = None
            else:
                print('Preparing the shared local session for the game batch.', flush=True)
                shared_session = contexts.enter_context(AnalysisSession(
                    prepared[0][1].board().fen(), CONFIG['ANALYSIS']['CACHE_DIR'], limits=Limits()))
                record['engine_load_seconds'] = time.perf_counter()-start
                record['engine_signature'] = shared_session.engines.signature
                record['maia_device'] = str(shared_session.engines.signature['device'])
                print(f'Maia device: {record["maia_device"]}', flush=True)
            for index, (path, game, output) in enumerate(prepared):
                session = cached_sessions[index] if rebuild_from_cache else shared_session
                began = time.perf_counter()
                previous = began
                before = session.stats.copy()
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
                analysis = analyze_game(game, session, progress=progress,
                                        accuracy_output_dir=output)
                store.save(output/'analysis.json', analysis)
                entry = {'game': path.stem, 'pgn': str(path), 'output': str(output),
                         'plies': len(analysis['moves']), 'seconds': time.perf_counter()-began,
                         'execution': analysis['analysis_execution'],
                         'accuracy_curve': analysis['accuracy_curve'],
                         'game_context': analysis['game'],
                         'engine_statistics': {key: session.stats[key]-before[key] for key in before},
                         'progress': events}
                record['games'].append(entry)
                write_json(timing_output, record)
                print(f'{path.stem}: complete in {entry["seconds"]:.1f}s; {output}', flush=True)
    finally:
        # The AnalysisSession context manager closes workers even if analysis fails.
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
    cli.add_argument('--replace-output', action='store_true',
                     help='Replace generated analysis and plots using the existing cache; preserve other files.')
    cli.add_argument('--rebuild-from-cache', action='store_true',
                     help='Rebuild from saved game measurements only; fail on missing evidence and never load engines.')
    cli.add_argument('--check-cache', action='store_true',
                     help='Validate every game against saved evidence without engines, output changes or timing files.')
    args = cli.parse_args()
    run(args.pgn, args.timing_output, replace_output=args.replace_output, rebuild_from_cache=args.rebuild_from_cache,
        check_cache=args.check_cache)
