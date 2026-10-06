"""Refit saved games, export production figures and compare reference labels.

No engine or language-model calls are made. Commercial estimates are read only
after fitting. Historic diagonal evidence can be migrated after validating its
conditioning, legal-move sequence and available saved Stockfish move qualities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from analysis.cache import identity, write_json
from analysis.lichess_accuracy import INITIAL_CP, move_accuracy, win_percent
from analysis.player_rating.context import saved_context
from analysis.player_rating.evidence import evidence_matches_game, validate_evidence
from analysis.player_rating.figures import export_figures
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.service import evidence_cache_path, fit_evidence, get_estimator, store_elo_fit
from analysis.player_rating.scale import rating_context
from analysis.elo_convert import convert
from analysis.game.performance import refresh_performance
from analysis.position_evaluation import centipawns
from analysis.settings import CONFIG

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'tests/analysis/output/selected-rating-production'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def historical_evidence(number):
    """Read archived equal-rating evidence, never relabel conditioned evidence."""
    if number <= 10:
        path = ROOT / 'tests/coach/output/refreshed-strength-11/audit/equal-opponent-evidence.json'
        offset = number * 2
    elif number <= 15:
        path = ROOT / 'tests/coach/output/trend-new-games-11-15/equal-evidence.json'
        offset = (number-11) * 2
    else:
        raise ValueError('No compatible cached evidence; run local full-game analysis first.')
    source = read_json(path)
    condition = source['conditioning']
    if (condition['method'] != 'equal_skill' or condition['rating_grid'] != list(RATINGS)
            or condition['account_ratings_used'] or condition['interpolation'] != 'none; exact diagonal profiles'):
        raise ValueError('Archived evidence does not identify exact equal-opponent policies.')
    records = source['records'][offset:offset+2]
    evidence = validate_evidence({side: {**record, 'conditioning': 'equal_opponent',
                                       'rating_grid': list(RATINGS)}
                                  for side, record in zip(('White', 'Black'), records, strict=True)})
    return evidence, path


def audit_evidence(game, analysis, evidence):
    """Check archived measurements against legal moves and available saved scores."""
    if not evidence_matches_game(evidence, game):
        raise ValueError('Cached rating evidence does not match the PGN.')
    moves = list(game.mainline_moves())
    rows = analysis['moves']
    if len(moves) != len(rows):
        raise ValueError('Saved analysis does not match the PGN length.')
    board = game.board()
    previous_score = INITIAL_CP if board.board_fen() == chess.Board().board_fen() else centipawns(rows[0]['position_eval'])
    offsets = {'White': 0, 'Black': 0}
    errors = []
    comparisons = 0
    for index, (move, row) in enumerate(zip(moves, rows, strict=True)):
        if row['fen'] != board.fen() or row['played']['move'] != move.uci():
            raise ValueError('Saved analysis contains a different position or move.')
        side = 'White' if board.turn else 'Black'
        observation = evidence[side]['observations'][offsets[side]]
        offsets[side] += 1
        legal = sorted(m.uci() for m in board.legal_moves)
        after_score = centipawns(rows[index+1]['position_eval']) if index+1 < len(rows) else centipawns(row['played']['eval'])
        child = board.copy()
        child.push(move)
        if child.is_checkmate():
            after_score = '#-0' if child.turn else '#0'
        elif child.is_game_over(claim_draw=False):
            after_score = 0
        for candidate in row['candidate_moves']:
            root_after = centipawns(candidate['eval'])
            candidate_child = board.copy()
            candidate_child.push_uci(candidate['move'])
            if candidate_child.is_checkmate():
                root_after = '#-0' if candidate_child.turn else '#0'
            elif candidate_child.is_game_over(claim_draw=False):
                root_after = 0
            for view, before, after in (
                ('root', centipawns(row['position_eval']), root_after),
                ('position', previous_score, after_score if candidate['move'] == move.uci() else root_after),
            ):
                before, after = win_percent(before), win_percent(after)
                quality = move_accuracy(before, after) if board.turn else move_accuracy(after, before)
                errors.append(abs(quality - observation['qualities'][view][legal.index(candidate['move'])]))
                comparisons += 1
        previous_score = after_score
        board.push(move)
    maximum = max(errors, default=0.)
    if maximum > 1e-6:
        raise ValueError(f'Cached quality differs from saved analysis: {maximum:.8f} accuracy points.')
    return {'candidate_quality_checks': comparisons, 'maximum_difference': maximum,
            'scope': 'All played moves and saved candidate moves; full legal order and policy schema validated.'}


def load_evidence(game, analysis, number, cache):
    metadata = analysis.get('rating_fit', {})
    key = metadata.get('evidence_key')
    path = evidence_cache_path(cache, key) if key else None
    if path is not None and path.is_file():
        evidence = validate_evidence(read_json(path))
    else:
        evidence, path = historical_evidence(number)
        key = identity(['validated_equal_opponent_evidence', evidence])
    audit = audit_evidence(game, analysis, evidence)
    target = evidence_cache_path(cache, key)
    if not target.exists():
        write_json(target, evidence)
    return evidence, key, {'source': str(path.resolve()),
                           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'cache': str(target.resolve()), 'audit': audit}


def metrics(rows, key):
    errors = [row[key]-row['reference'] for row in rows if row['reference'] is not None and row[key] is not None]
    if not errors:
        return {'players': 0, 'mean_absolute_error': None, 'maximum_absolute_error': None,
                'root_mean_square_error': None, 'mean_signed_error': None}
    absolute = np.abs(errors)
    return {'players': len(errors), 'mean_absolute_error': float(np.mean(absolute)),
            'maximum_absolute_error': float(np.max(absolute)),
            'root_mean_square_error': float(np.sqrt(np.mean(np.square(errors)))),
            'mean_signed_error': float(np.mean(errors))}


def comparison_figure(rows, output, *, method_name=None, method_id=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(12, 10), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1)
    fig.suptitle('Saved-game rating estimates versus commercial PGN references', fontsize=16)
    x = np.arange(len(rows))
    ref = np.array([row['reference'] for row in rows], dtype=float)
    # Keep archived comparison files readable without relabeling their method.
    current_key = 'selected_rating' if any('selected_rating' in row for row in rows) else 'current_shared_curve'
    method_name = method_name or next((row.get('method_name') for row in rows if row.get('method_name')), None)
    method_id = method_id or next((row.get('method_id') for row in rows if row.get('method_id')), None)
    label = method_name or ('Current shared curve' if current_key == 'current_shared_curve' else 'Selected estimator')
    if method_id:
        label += f' ({method_id})'
    methods = [('previous_saved', 'Previous saved estimate', '#a9aeb6'),
               (current_key, label, '#16828a')]
    for method, label, color in methods:
        if any(method not in row for row in rows):
            continue
        values = np.array([row[method] for row in rows], dtype=float)
        axes[0].plot(x, values, marker='o', ms=3, lw=1, color=color, label=label)
        axes[1].plot(x, values-ref, marker='o', ms=3, lw=1, color=color, label=label)
    axes[0].plot(x, ref, marker='x', ms=5, ls='none', color='#292e37', label='Commercial reference')
    scales = sorted({row.get('rating_scale_name', 'Legacy unspecified scale') for row in rows})
    axis_scale = scales[0] if len(scales) == 1 else 'Each game\'s declared scale'
    axes[0].set_ylabel(f'Estimated rating — {axis_scale}')
    axes[1].set_ylabel(f'Estimate minus reference — {axis_scale}')
    axes[1].axhline(0, color='#292e37', lw=.8)
    for axis in axes:
        axis.set_xticks(x, [f'{r["game"][4:]}{r["side"][0]}' for r in rows], rotation=60)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=9)
        axis.set_xlabel('Game number and player (W / B)')
    fig.savefig(output/'commercial-comparison.svg', facecolor='white')
    (output/'commercial-comparison.png').unlink(missing_ok=True)
    fig.clear()


def figures_only(games_dir, output):
    """Re-render saved fit data, leaving game analysis and evidence untouched."""
    paths = sorted(games_dir.glob('game*.pgn'), key=lambda p: int(p.stem[4:]))
    if not paths:
        raise ValueError(f'No game PGNs found in {games_dir}.')
    saved = []
    for pgn in paths:
        game_output = pgn.parent/'output'/f'{pgn.stem}-full'
        analysis_path = game_output/'analysis.json'
        fit_path = game_output/'player-rating'/'fit.json'
        fit = read_json(fit_path)
        saved.append((pgn, analysis_path, hashlib.sha256(analysis_path.read_bytes()).digest(), fit))
    result = {}
    for pgn, analysis_path, original_hash, fit in saved:
        result[pgn.stem] = export_figures(
            fit, analysis_path.parent/'player-rating', title=f'{pgn.stem} — played-strength estimate')
        if hashlib.sha256(analysis_path.read_bytes()).digest() != original_hash:
            raise RuntimeError(f'{analysis_path} changed while rendering its figures.')
        print(f'{pgn.stem}: saved fit rendered; analysis unchanged', flush=True)
    comparison_path = output/'comparison.json'
    if comparison_path.is_file():
        comparison = read_json(comparison_path)
        comparison_figure(comparison['players'], output, method_name=comparison.get('method_name'),
                          method_id=comparison.get('method_id'))
        print('Commercial comparison rendered from saved results.', flush=True)
    return result


def run(games_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(CONFIG['ANALYSIS']['CACHE_DIR'])
    estimator = get_estimator()
    games, rows = [], []
    previous_path = output/'comparison.json'
    previous = {item['game']: item['provenance'] for item in read_json(previous_path)['games']} if previous_path.exists() else {}
    paths = sorted(games_dir.glob('game*.pgn'), key=lambda p: int(p.stem[4:]))
    if not paths:
        raise ValueError(f'No game PGNs found in {games_dir}.')
    for pgn in paths:
        with pgn.open(encoding='utf-8-sig') as stream:
            game = chess.pgn.read_game(stream)
        if game is None or game.errors:
            raise ValueError(f'{pgn.name} is not a valid game.')
        game_output = pgn.parent / 'output' / f'{pgn.stem}-full'
        analysis_path = game_output / 'analysis.json'
        analysis = read_json(analysis_path)
        backup = output / 'original-analysis' / f'{pgn.stem}.json'
        if not backup.exists():
            write_json(backup, analysis)
        original = read_json(backup)
        evidence, key, provenance = load_evidence(game, analysis, int(pgn.stem[4:]), cache)
        earlier = previous.get(pgn.stem, provenance)
        provenance['origin'] = earlier.get('origin', {field: earlier[field] for field in ('source', 'source_sha256')})
        # This corpus command evaluates the latest PGN declarations, not stale
        # interactive overrides saved during earlier coaching runs.
        analysis['headers'] = dict(game.headers)
        analysis['rating_account_overrides'] = {}
        analysis.pop('rating_scale_override', None)
        selected = analysis.get('selected_player', {})
        side = str(selected.get('side', '')).title()
        if side in ('White', 'Black'):
            selected['actual_elo'] = int(game.headers[side+'Elo']) if game.headers.get(side+'Elo', '').isdigit() else None
        context = rating_context(game.headers)
        fit = fit_evidence(saved_context(evidence, analysis), evidence_key=key, rating_scale=context)
        store_elo_fit(analysis, fit)
        refresh_performance(analysis)
        write_json(analysis_path, analysis)
        export_figures(fit, game_output/'player-rating', title=f'{pgn.stem} — played-strength estimate')
        game_rows = []
        # Reference labels enter only after the completed estimator call.
        for side in ('White', 'Black'):
            raw_reference = game.headers.get(f'{side}EloEstimate')
            reference = int(raw_reference) if raw_reference and raw_reference.isdigit() else None
            interval = fit['players'][side]['interval']
            old_scale = original.get('played_elo_scale', {}).get('scale', 'lb')
            old_point = original['played_elo'][side.lower()].get('unrounded_estimate',
                        original['played_elo'][side.lower()]['estimate'])
            previous_point = None if old_point is None else convert(old_point, old_scale, context['scale'], extrapolate=True)
            row = {'game': pgn.stem, 'side': side, 'method_id': fit['method_id'],
                   'method_name': fit['name'], 'reference': reference,
                   'rating_scale': context['scale'], 'rating_scale_name': context['name'],
                   'actual_rating': game.headers.get(f'{side}Elo'),
                   'selected_rating': fit['players'][side]['estimate'],
                   'interval_low': interval[0] if interval is not None else None,
                   'interval_high': interval[1] if interval is not None else None,
                   'previous_saved': None if previous_point is None else int(np.floor(previous_point+.5))}
            rows.append(row)
            game_rows.append(row)
        game_result = {'game': pgn.stem, 'provenance': provenance, 'players': fit['players'],
                       'method_id': fit['method_id'], 'method_name': fit['name'],
                       'previous_saved_method': original['played_elo_method']}
        for field in ('selected_rating', 'previous_saved'):
            comparable = all(row['reference'] is not None and row[field] is not None for row in game_rows)
            if comparable:
                reference_gap = game_rows[0]['reference']-game_rows[1]['reference']
                fitted_gap = game_rows[0][field]-game_rows[1][field]
                matched = abs(fitted_gap) < 50 if reference_gap == 0 else np.sign(fitted_gap) == np.sign(reference_gap)
                game_result[f'{field}_order_matches_reference'] = bool(matched)
            else:
                game_result[f'{field}_order_matches_reference'] = None
        games.append(game_result)
        print(f'{pgn.stem}: White {game_rows[0]["selected_rating"]}, Black {game_rows[1]["selected_rating"]}; {fit["name"]}; cached evidence verified', flush=True)
    summary = {key: metrics(rows, key) for key in ('selected_rating', 'previous_saved')}
    for key in ('selected_rating', 'previous_saved'):
        matches = [game[f'{key}_order_matches_reference'] for game in games
                   if game[f'{key}_order_matches_reference'] is not None]
        summary[f'{key}_order'] = {'matched': sum(matches), 'games': len(matches)}
    result = {'method_id': estimator.id, 'method_name': estimator.name,
              'method_parameters': estimator.parameters, 'summary': summary, 'players': rows, 'games': games,
              'evaluation_only': 'Commercial PGN labels are read only after inference. This repeated exploratory comparison is not independent validation.',
              'evidence': 'Existing saved engine measurements; no Stockfish, Maia or language-model calls. Legacy exact-diagonal evidence was validated and migrated where necessary.',
              'baseline': 'Previous estimates are read from preserved analysis snapshots and converted to the current PGN scale; unlabeled historical fits are native Lichess Blitz. No retired estimator is recomputed.'}
    result['ordering_rule'] = 'An exact commercial tie accepts a fitted difference below 50 Elo; every nonzero commercial difference requires the same fitted sign.'
    write_json(output/'comparison.json', result)
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison_figure(rows, output, method_name=estimator.name, method_id=estimator.id)
    print(json.dumps(summary, indent=2))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--games-dir', type=Path, default=ROOT/'games')
    cli.add_argument('--output-dir', type=Path, default=OUTPUT)
    cli.add_argument('--figures-only', action='store_true',
                     help='Render saved fit.json files without refitting or changing game analyses.')
    args = cli.parse_args()
    if args.figures_only:
        figures_only(args.games_dir, args.output_dir)
    else:
        run(args.games_dir, args.output_dir)


if __name__ == '__main__':
    main()
