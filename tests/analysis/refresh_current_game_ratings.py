"""Refresh existing saved ratings from each game's own cached evidence only.

This maintenance runner never loads other-game calibration, rebuilds engine
evidence, scores commercial labels, or compares withdrawn rating methods.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

import chess.pgn

from analysis.cache import write_json
from analysis.player_rating.context import saved_context
from analysis.player_rating.evidence import evidence_matches_game, validate_evidence
from analysis.player_rating.figures import export_figures
from analysis.player_rating.scale import rating_context
from analysis.player_rating.service import evidence_cache_path, fit_evidence, selected_method, store_elo_fit
from analysis.settings import CONFIG


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/shared-curve-affine-current-game-only'
RATING_FILES = {'fit.json', 'analysis.svg', 'prior.svg', 'method-explanation.svg'}


def discover_games(directory):
    """Include every direct PGN child in natural filename order, without a range limit."""
    paths = [path for path in Path(directory).iterdir() if path.is_file() and path.suffix.lower() == '.pgn']
    if not paths:
        raise ValueError(f'No PGN games found in {directory}.')
    return sorted(paths, key=lambda path: [int(part) if part.isdigit() else part.casefold()
                                          for part in re.split(r'(\d+)', path.name)])


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths):
    paths = sorted(set(map(Path, paths)))
    with ThreadPoolExecutor(max_workers=8) as executor:
        return {str(path): value for path, value in zip(paths, executor.map(file_hash, paths), strict=True)}


def non_rating_data(analysis):
    return {key: value for key, value in analysis.items()
            if not key.startswith('played_elo') and key != 'rating_fit'}


def audit_game(pgn, cache):
    """Validate the complete source pair before any game's output is changed."""
    with pgn.open(encoding='utf-8-sig') as stream:
        game = chess.pgn.read_game(stream)
    if game is None or game.errors or not list(game.mainline_moves()):
        raise ValueError(f'{pgn.name}: a valid played game is required.')
    directory = pgn.parent/'output'/(pgn.stem+'-full')
    path = directory/'analysis.json'
    analysis = json.loads(path.read_text(encoding='utf-8-sig'))
    if analysis.get('headers') != dict(game.headers):
        raise ValueError(f'{pgn.name}: current PGN headers and saved analysis differ.')
    moves = list(game.mainline_moves())
    if [row['played']['move'] for row in analysis['moves']] != [move.uci() for move in moves]:
        raise ValueError(f'{pgn.name}: saved analysis moves differ from the PGN.')
    board = game.board()
    for move, row in zip(moves, analysis['moves'], strict=True):
        if row['fen'] != board.fen():
            raise ValueError(f'{pgn.name}: saved before-position FEN differs at {row["ply"]}.')
        board.push(move)
    key = analysis['rating_fit']['evidence_key']
    evidence_path = evidence_cache_path(cache, key)
    evidence = validate_evidence(json.loads(evidence_path.read_text(encoding='utf-8-sig')))
    if not evidence_matches_game(evidence, game):
        raise ValueError(f'{pgn.name}: cached evidence does not match the game.')
    return {'name': pgn.stem, 'pgn': pgn, 'directory': directory, 'path': path,
            'analysis': analysis, 'cache': evidence_path, 'evidence_key': key,
            'evidence': evidence, 'positions': len(moves)}


def refit_case(case):
    """Always recompute the fit, including when its saved model signature is current."""
    analysis = case['analysis']
    context = rating_context(analysis.get('headers'), analysis.get('rating_scale_override'))
    fit = fit_evidence(saved_context(case['evidence'], analysis),
                       evidence_key=case['evidence_key'], rating_scale=context)
    store_elo_fit(analysis, fit)
    export_figures(fit, case['directory']/'player-rating')
    return fit


def run(games=ROOT/'games', output=OUTPUT):
    started = perf_counter()
    if selected_method() != 'shared_curve_affine':
        raise ValueError('This refresh requires the current-game-only shared_curve_affine method.')
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'tests/analysis/output'):
        raise ValueError('Verification output must remain in tests/analysis/output.')
    cache = Path(CONFIG['ANALYSIS']['CACHE_DIR'])
    cases = [audit_game(path, cache) for path in discover_games(games)]
    cache_paths = sorted(path for path in cache.rglob('*') if path.is_file())
    protected = [case['pgn'] for case in cases]
    for case in cases:
        protected.extend(path for path in case['directory'].rglob('*') if path.is_file()
                         and path != case['path']
                         and not (path.parent == case['directory']/'player-rating' and path.name in RATING_FILES))
    print(f'Preflight valid for {len(cases)} games; hashing {len(cache_paths)} cache files.', flush=True)
    cache_before = hash_files(cache_paths)
    other_before = hash_files(protected)
    rows = []
    for case in cases:
        case_started = perf_counter()
        analysis = case['analysis']
        unchanged = deepcopy(non_rating_data(analysis))
        fit = refit_case(case)
        if non_rating_data(analysis) != unchanged:
            raise AssertionError('Rating refresh changed unrelated game analysis.')
        if analysis['played_elo_method'] != 'shared_curve_affine':
            raise AssertionError('Refresh did not use the current-game estimator.')
        diagnostics = analysis['played_elo_diagnostics']
        if diagnostics.get('kind') != 'shared_curve_affine' or any(
                key in diagnostics for key in ('hierarchy', 'calibration', 'population')):
            raise AssertionError('A refreshed game retains population-model diagnostics.')
        if diagnostics['model']['accuracy_curve'] != diagnostics['curve']['shared_accuracy']:
            raise AssertionError('The estimator did not use its original game shared curve.')
        if analysis['rating_fit']['evidence_key'] != case['evidence_key']:
            raise AssertionError('Rating refresh changed the evidence identity.')
        write_json(case['path'], analysis)
        artifacts = {name: file_hash(case['directory']/'player-rating'/name) for name in sorted(RATING_FILES)}
        rows.append({'game': case['name'], 'positions': case['positions'],
                     'pgn_headers_and_moves_match': True, 'cached_evidence_matches_game': True,
                     'evidence_key': case['evidence_key'], 'evidence_path': str(case['cache']),
                     'method': analysis['played_elo_method'], 'rating_scale': analysis['played_elo_scale']['scale'],
                     'rating_scale_name': analysis['played_elo_scale']['name'],
                     'white': fit['players']['White']['estimate'], 'black': fit['players']['Black']['estimate'],
                     'recomputed': True, 'runtime_seconds': perf_counter()-case_started,
                     'no_population_diagnostics': True, 'non_rating_analysis_unchanged': True,
                     'analysis_sha256': file_hash(case['path']), 'player_rating_artifacts': artifacts})
        print(f'{case["name"]}: {rows[-1]["white"]} / {rows[-1]["black"]}; recomputed rating and 3 SVGs.', flush=True)
    cache_after_paths = sorted(path for path in cache.rglob('*') if path.is_file())
    cache_after = hash_files(cache_after_paths)
    other_after = hash_files(protected)
    if cache_before != cache_after or other_before != other_after:
        raise AssertionError('A cache file, PGN, or unrelated output changed during the refresh.')
    verification = {'created_utc': datetime.now(timezone.utc).isoformat(), 'method': selected_method(),
                    'scope': 'Each game uses only its own cached numeric engine evidence and supplied account ratings.',
                    'commercial_evaluation_performed': False, 'population_assets_used': False,
                    'game_count': len(rows), 'positions': sum(row['positions'] for row in rows),
                    'player_rating_artifact_count': len(rows)*len(RATING_FILES), 'svg_count': len(rows)*3,
                    'all_cache_files_unchanged': True, 'cache_file_count': len(cache_before),
                    'cache_sha256_before': cache_before, 'cache_sha256_after': cache_after,
                    'pgn_and_other_outputs_unchanged': True,
                    'protected_files_before': other_before, 'protected_files_after': other_after,
                    'games': rows, 'runtime_seconds': perf_counter()-started}
    write_json(output/'verification.json', verification)
    write_json(output/'ratings.json', {'method': selected_method(), 'game_count': len(rows),
               'games': [{key: row[key] for key in ('game', 'white', 'black', 'rating_scale',
                                                   'rating_scale_name', 'runtime_seconds')} for row in rows]})
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games-dir', type=Path, default=ROOT/'games')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.games_dir, args.output)
    print(json.dumps({key: result[key] for key in ('game_count', 'positions', 'svg_count',
                     'cache_file_count', 'all_cache_files_unchanged', 'runtime_seconds')}))
