"""Run from the repository root or coach/: python coach.py game.pgn --side white."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.game.pipeline import ANALYSIS_VERSION, analyze_game
from analysis.cache.artifacts import AnalysisStore
from analysis.game.study import load_game
from analysis.accuracy.performance import refresh_performance
from analysis.game.metadata import game_metadata
from analysis.elo_convert import normalize_scale, resolve_scale
from analysis.session import AnalysisSession, Limits
from analysis.cache.storage import identity
from coach.settings import CONFIG
from engine.uci import seconds_to_milliseconds


def parser():
    cli = argparse.ArgumentParser(description='Local Maia-3/Stockfish analysis followed by Codex (ChatGPT sign-in).')
    cli.add_argument('pgn', type=Path)
    cli.add_argument('--side', choices=('white','black'),
                     help='Player to coach; required except with --analysis-only. Never stored in game analysis.')
    cli.add_argument('--elo', type=int,
                     help='Override BOTH players\' actual Elo with this value; requires --rating-scale. '
                          'Omit both options to use PGN WhiteElo/BlackElo, Site and TimeControl.')
    cli.add_argument('--rating-scale', type=normalize_scale, choices=('lb', 'lr', 'cb', 'cr'),
                     help='Scale of the --elo override for both players; requires --elo. '
                          'Accepts lichess_blitz, lichess_rapid, chess_com_blitz, chess_com_rapid '
                          'or lb/lr/cb/cr.')
    cli.add_argument('--model', help='Override COACH.CODEX.MODEL in config.yaml.')
    cli.add_argument('--stockfish-path', default=CONFIG['STOCKFISH']['EXECUTABLE'])
    cli.add_argument('--maia-checkpoint', default=CONFIG['MAIA']['CHECKPOINT'])
    cli.add_argument('--device', choices=('auto','cpu','cuda'), default=CONFIG['MAIA']['DEVICE'])
    cli.add_argument('--threads-per-worker', type=int, default=CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'],
                     help='Stockfish CPU threads for each worker, independent of the worker count.')
    cli.add_argument('--hash-mb-per-worker', dest='hash_mb', type=int,
                     default=CONFIG['ANALYSIS']['STOCKFISH_HASH_MB_PER_WORKER'],
                     help='Stockfish hash memory in MB for each worker, independent of the worker count.')
    cli.add_argument('--verify-ms', type=int, default=seconds_to_milliseconds(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['DEFAULT_SEARCH_SECONDS']),
                     help='Time per position in full analysis and each chess-tool evaluation, in milliseconds; YAML uses seconds.')
    cli.add_argument('--max-ms', type=int, default=seconds_to_milliseconds(CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['MAX_SEARCH_SECONDS']),
                     help='Maximum time per requested Stockfish search, in milliseconds; bounded by ANALYSIS.STOCKFISH_EVALUATION.MAX_SEARCH_SECONDS.')
    cli.add_argument('--depth', type=int, default=CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['MAX_DEPTH'], help='Stockfish depth ceiling; every search also has a time limit.')
    cli.add_argument('--analysis-strategy', choices=('bounded','staged','exhaustive'), default=CONFIG['ANALYSIS']['STOCKFISH_SEARCH_STRATEGY'])
    cli.add_argument('--max-model-responses', type=int, default=CONFIG['COACH']['MAX_MODEL_RESPONSES'],
                     help='Codex model-response budget; also sets the default chess-tool limit to four times this value. Does not limit game moves or Stockfish depth.')
    cli.add_argument('--unreported-output-token-estimate', dest='max_tokens', type=int,
                     default=CONFIG['COACH']['OUTPUT_TOKEN_ESTIMATE'],
                     help='Output tokens charged to the run budget when response usage is unavailable; does not cap response length.')
    cli.add_argument('--output-dir', type=Path, help='Override <PGN parent>/output/<game name>-full.')
    cli.add_argument('--cache-dir', type=Path, default=CONFIG['ANALYSIS']['CACHE_DIR'],
                     help='Analysis evidence cache directory; defaults to ANALYSIS.CACHE_DIR.')
    cli.add_argument('--refresh-cache', action='store_true',
                     help='Recompute requested Maia/Stockfish evidence and replace its cache entries; '
                          'preserve unrelated evidence. Cannot be combined with --coach-only.')
    cli.add_argument('--cache-only', action='store_true',
                     help='Rebuild --analysis-only from the saved game\'s original cached measurements. '
                          'Never load engines or calculate misses; incompatible with --refresh-cache.')
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument('--analysis-only', action='store_true', help='Run all local analysis without an LLM or API key.')
    modes.add_argument('--coach-only', action='store_true', help='Reuse this output directory\'s analysis.json; run the coach again.')
    return cli


def codex_model(args):
    return args.model or CONFIG['COACH']['CODEX']['MODEL']


def declared_game_context(headers, elo, scale, *, game=None):
    """Validate the CLI's explicit pair or complete PGN context before any work."""
    try:
        context = game_metadata(headers, actual_elo=elo, rating_scale=scale, game=game)
        missing = [side.title() + 'Elo' for side in ('white', 'black') if context[side]['elo'] is None]
        if missing:
            raise ValueError('PGN needs valid ' + ' and '.join(missing) + ' (100–4000).')
        # The shared pipeline also supports metadata-light web analysis; this CLI
        # requires an explicit account scale or complete PGN scale metadata.
        resolve_scale(headers, override=scale)
    except ValueError as exc:
        raise ValueError(f'{exc} Provide --elo and --rating-scale together to set both players\' ratings.') from exc
    return context


def failure_detail(exc):
    """Unwrap safe application diagnostics without printing provider response bodies."""
    current = exc
    for _ in range(5):
        if type(current).__module__.startswith('coach.') and isinstance(current, (ValueError, TimeoutError)):
            return str(current)
        current = current.__cause__
        if current is None:
            break
    return str(exc) if isinstance(exc, (ValueError, FileNotFoundError, TimeoutError)) else type(exc).__name__


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if not args.analysis_only and args.side is None:
            raise ValueError('Provide --side white or --side black to select the player to coach.')
        if args.coach_only and args.refresh_cache:
            raise ValueError('--refresh-cache requires game analysis; omit --coach-only or use --analysis-only.')
        if args.cache_only and (not args.analysis_only or args.refresh_cache):
            raise ValueError('--cache-only requires --analysis-only and cannot be combined with --refresh-cache.')
        if (args.elo is None) != (args.rating_scale is None):
            raise ValueError('Provide --elo and --rating-scale together, or omit both to use PGN WhiteElo/BlackElo, Site and TimeControl.')
        if args.elo is not None and not 100 <= args.elo <= 4000:
            raise ValueError('Actual Elo must be between 100 and 4000.')
        if not 1 <= args.threads_per_worker <= 256 or not 16 <= args.hash_mb <= 8192:
            raise ValueError('Threads per worker must be 1–256 and hash memory per worker 16–8192 MB.')
        if not 6 <= args.max_model_responses <= 100:
            raise ValueError('Model response budget must be 6–100.')
        if not 512 <= args.max_tokens <= 16000:
            raise ValueError('Output allowance must be 512–16000 tokens.')
        limits = Limits(**{key: getattr(args,key) for key in ('verify_ms','max_ms','depth','analysis_strategy')})
        game = load_game(args.pgn)
        metadata = declared_game_context(dict(game.headers), args.elo, args.rating_scale, game=game)
        model_id = codex_model(args)
        output = args.output_dir or args.pgn.parent / 'output' / f'{args.pgn.stem}-full'
        store = AnalysisStore(args.cache_dir)
        output.mkdir(parents=True, exist_ok=True)
        if (output/'game.pgn').resolve() != args.pgn.resolve():
            (output/'game.pgn').write_text(args.pgn.read_text(encoding='utf-8-sig'), encoding='utf-8')
        analysis = None
        if args.coach_only:
            analysis = store.load(output/'analysis.json')
            if analysis.get('schema_version') != ANALYSIS_VERSION:
                raise ValueError('Saved analysis uses an older schema or search method. Regenerate locally with --analysis-only first.')
            expected = identity([game.board().fen(), [m.uci() for m in game.mainline_moves()]])
            if analysis['game_id'] != expected:
                raise ValueError('Saved analysis belongs to a different game. Run without --coach-only.')
            # Chess measurements are reusable when the declared account context changes.
            analysis['headers'] = dict(game.headers)
            analysis['game'] = metadata
            refresh_performance(analysis)
            from analysis.accuracy.service import refresh_saved_curve
            refresh_saved_curve(analysis, output_dir=output)
            store.save(output/'analysis.json', analysis)
        if args.cache_only:
            from analysis.cache.session import CachedAnalysisSession
            print('Rebuilding analysis from saved measurements; engines disabled.', flush=True)
            context = CachedAnalysisSession(game, args.cache_dir)
        else:
            print('Preparing local analysis; engines run only for missing evidence.', flush=True)
            context = AnalysisSession(game.board().fen(), args.cache_dir, limits=limits, stockfish_path=args.stockfish_path,
                     checkpoint=args.maia_checkpoint, device=args.device, threads_per_worker=args.threads_per_worker,
                     hash_mb=args.hash_mb, refresh_cache=args.refresh_cache)
        with context as session:
            if analysis is None:
                analysis = analyze_game(game, session, actual_elo=args.elo,
                                        progress=lambda message: print(message, flush=True),
                                        accuracy_output_dir=output, rating_scale=args.rating_scale)
                store.save(output/'analysis.json', analysis)
            print(f'Analysis saved: {output / "analysis.json"}', flush=True)
            if not args.analysis_only:
                from coach.agent_runner import run_coach
                print(f'Coaching with {model_id or "Codex default model"}; evidence goes to Codex using your ChatGPT sign-in…', flush=True)
                run_coach(analysis, session, output, side=args.side, model_id=model_id,
                          max_model_responses=args.max_model_responses, max_tokens=args.max_tokens)
                print(f'Coaching saved: {output / "coaching.md"}', flush=True)
                print('Usage: ' + json.dumps(analysis['agent_run']['usage']), flush=True)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Local engines closed; completed analysis remains available.', file=sys.stderr)
        return 130
    except Exception as exc:
        # Avoid provider response bodies or credentials in terminal diagnostics.
        detail = failure_detail(exc)
        print(f'Coach failed: {detail.rstrip(".")}. Local engines closed.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
