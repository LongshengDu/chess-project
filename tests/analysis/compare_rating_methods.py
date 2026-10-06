"""Compare both frozen rating methods using existing full-game engine evidence.

All requested PGNs, analyses and evidence are validated before any output is
written. Both historical methods export to the comparison directory, preserving
the selected production method in normal game outputs. No engines or language models run.
Reference labels enter scoring only after both methods have finished fitting.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import chess.pgn
import numpy as np

from analysis.cache import write_json
from analysis.player_rating.calibration import evidence_fingerprint, load_calibration
from analysis.player_rating.context import saved_context
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.figures import export_figures
from analysis.player_rating.service import evidence_cache_path, fit_evidence
from analysis.settings import CONFIG
from tests.analysis.compare_shared_curve_games import audit_evidence


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/rating-methods-games0-18'
METHODS = ('uncertainty_ensemble', 'bayesian_shared_curve')
SIDES = ('White', 'Black')
TABLE_COLUMNS = ('Game', 'Commercial W / B', 'Uncertainty ensemble W / B', 'Bayesian shared curve W / B')


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def load_cases(games_dir, numbers, cache):
    """Read and audit every input without migration or other filesystem writes."""
    cases = []
    for number in numbers:
        pgn = games_dir/f'game{number}.pgn'
        pgn_bytes = pgn.read_bytes()
        with pgn.open(encoding='utf-8-sig') as stream:
            game = chess.pgn.read_game(stream)
        if game is None or game.errors or not list(game.mainline_moves()):
            raise ValueError(f'{pgn.name} must contain a valid played game.')
        analysis_path = games_dir/'output'/f'{pgn.stem}-full'/'analysis.json'
        analysis_bytes = analysis_path.read_bytes()
        analysis = json.loads(analysis_bytes.decode('utf-8-sig'))
        key = analysis.get('rating_fit', {}).get('evidence_key')
        evidence_path = evidence_cache_path(cache, key)
        evidence_bytes = evidence_path.read_bytes()
        evidence = validate_evidence(json.loads(evidence_bytes.decode('utf-8-sig')))
        audit = audit_evidence(game, analysis, evidence)
        # The PGN supplies current actual Elo; explicit saved overrides retain
        # precedence. saved_context reads only those declared ratings and scores.
        contextual_analysis = {**analysis, 'headers': dict(game.headers)}
        contextual = validate_evidence(saved_context(evidence, contextual_analysis))
        cases.append({'number': number, 'name': pgn.stem, 'game': game,
                      'analysis': contextual_analysis, 'evidence': contextual, 'key': key,
                      'analysis_path': analysis_path, 'original_analysis_bytes': analysis_bytes,
                      'inputs': {pgn: _hash(pgn_bytes), analysis_path: _hash(analysis_bytes),
                                 evidence_path: _hash(evidence_bytes)}, 'audit': audit})
    return cases


def _fit(evidence, key, method):
    """Select the public service in memory so its validation/signature stay exact."""
    with patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD=method):
        return fit_evidence(evidence, evidence_key=key)


def _reference(game, side):
    value = game.headers.get(side+'EloEstimate')
    return int(value) if value and value.isdigit() else None


def method_metrics(rows, method):
    """Score displayed integers; retain unrounded errors and distinguish ties."""
    selected = [row for row in rows if row['method_id'] == method]
    comparable = [row for row in selected if row['reference'] is not None and row['unrounded_estimate'] is not None]
    errors = np.asarray([row['unrounded_estimate']-row['reference'] for row in comparable])
    rounded = np.asarray([row['estimate']-row['reference'] for row in comparable])
    paired = {}
    for row in selected:
        paired.setdefault(row['game'], {})[row['side']] = row
    order_checks, decisive_checks, ties = [], [], []
    for game, pair in paired.items():
        if set(pair) == set(SIDES) and all(pair[side]['reference'] is not None and
                                         pair[side]['unrounded_estimate'] is not None for side in SIDES):
            reference = np.sign(pair['White']['reference']-pair['Black']['reference'])
            raw_difference = pair['White']['unrounded_estimate']-pair['Black']['unrounded_estimate']
            displayed_difference = pair['White']['estimate']-pair['Black']['estimate']
            check = (np.sign(raw_difference) == reference, np.sign(displayed_difference) == reference)
            order_checks.append(check)
            if reference:
                decisive_checks.append(check)
            else:
                ties.append({'game': game, 'reference': pair['White']['reference'],
                             'white_minus_black': displayed_difference,
                             'absolute_difference': abs(displayed_difference),
                             'unrounded_white_minus_black': raw_difference})
    largest = sorted(comparable, key=lambda row: abs(row['estimate']-row['reference']), reverse=True)[:5]
    return {'players': len(comparable), 'games': len(paired),
            'point_values_for_primary_metrics': 'displayed_integer_estimates',
            'mean_absolute_error': float(np.mean(abs(rounded))) if rounded.size else None,
            'maximum_absolute_error': float(np.max(abs(rounded))) if rounded.size else None,
            'root_mean_square_error': float(np.sqrt(np.mean(rounded**2))) if rounded.size else None,
            'mean_signed_error': float(np.mean(rounded)) if rounded.size else None,
            'unrounded_mean_absolute_error': float(np.mean(abs(errors))) if errors.size else None,
            'unrounded_maximum_absolute_error': float(np.max(abs(errors))) if errors.size else None,
            'ordering': {'strict_matched_including_reference_ties': sum(bool(pair[1]) for pair in order_checks),
                         'games': len(order_checks),
                         'unrounded_strict_matched': sum(bool(pair[0]) for pair in order_checks),
                         'decisive_matched': sum(bool(pair[1]) for pair in decisive_checks),
                         'decisive_games': len(decisive_checks),
                         'unrounded_decisive_matched': sum(bool(pair[0]) for pair in decisive_checks),
                         'reference_ties': ties},
            'largest_errors': [{'game': row['game'], 'side': row['side'], 'reference': row['reference'],
                                'estimate': row['estimate'], 'unrounded_estimate': row['unrounded_estimate'],
                                'absolute_error': abs(row['estimate']-row['reference']),
                                'unrounded_absolute_error': abs(row['unrounded_estimate']-row['reference'])} for row in largest]}


def _pair(values):
    return ' / '.join('—' if values[side] is None else str(values[side]) for side in SIDES)


def table_rows(games):
    return [dict(zip(TABLE_COLUMNS, [game['game'], _pair(game['reference']),
                                     *(_pair(game['methods'][method]) for method in METHODS)], strict=True))
            for game in games]


def markdown_table(games):
    lines = ['| '+' | '.join(TABLE_COLUMNS)+' |',
             '| --- | --- | --- | --- |']
    for row in table_rows(games):
        lines.append('| '+' | '.join(row.values())+' |')
    return '\n'.join(lines)+'\n'


def run(games_dir, output, *, start=0, end=18, cache=None):
    """Fit both methods offline, score by cohort, then write the validated run."""
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError('Game bounds must be nonnegative integers with start <= end.')
    games_dir, output = Path(games_dir), Path(output)
    calibration = load_calibration()
    cases = load_cases(games_dir, range(start, end+1), Path(cache or CONFIG['ANALYSIS']['CACHE_DIR']))
    fits = {}
    for case in cases:
        fits[case['name']] = {method: _fit(case['evidence'], case['key'], method) for method in METHODS}
        result = fits[case['name']]['uncertainty_ensemble']
        if result['parameters']['calibration_hash'] != calibration.content_hash:
            raise RuntimeError('The frozen calibration changed during fitting; no output was written.')
        expected_exclusions = 1 if case['number'] <= 15 else 0
        excluded = result['diagnostics']['calibration']['contexts_excluded']
        if excluded != expected_exclusions:
            raise ValueError(f'{case["name"]}: expected {expected_exclusions} excluded calibration contexts, '
                             f'found {excluded}; numeric evidence does not match the intended study split. No output was written.')
        print(f'{case["name"]}: both methods fitted from the same cached evidence.', flush=True)
    # Read reference labels only after every fit has completed.
    games, rows = [], []
    for case in cases:
        game_fits = fits[case['name']]
        references = {side: _reference(case['game'], side) for side in SIDES}
        calibration_details = game_fits['uncertainty_ensemble']['diagnostics']['calibration']
        expected_exclusions = 1 if case['number'] <= 15 else 0
        games.append({'game': case['name'], 'number': case['number'], 'reference': references,
                      'methods': {method: {side: fit['players'][side]['estimate'] for side in SIDES}
                                  for method, fit in game_fits.items()},
                      'calibration': {**calibration_details,
                                      'target_fingerprint': evidence_fingerprint(case['evidence']),
                                      'expected_exclusions_for_study_split': expected_exclusions,
                                      'matches_study_split': calibration_details['contexts_excluded'] == expected_exclusions},
                      'evidence_key': case['key'], 'audit': case['audit'],
                      'input_sha256': {str(path.resolve()): digest for path, digest in case['inputs'].items()}})
        for method, fit in game_fits.items():
            for side in SIDES:
                player = fit['players'][side]
                if 'unrounded_estimate' not in player:
                    raise ValueError(f'{method} did not provide an unrounded estimate; no output was written.')
                rows.append({'game': case['name'], 'number': case['number'], 'side': side,
                             'method_id': method, 'method_name': fit['name'], 'reference': references[side],
                             'actual_rating': case['evidence'][side].get('actual_rating'),
                             'estimate': player['estimate'], 'unrounded_estimate': player['unrounded_estimate'],
                             'interval': player['interval'], 'uncertainty': player['uncertainty']})
    cohorts = {'all_games': rows, 'original_games_0_15': [row for row in rows if row['number'] <= 15],
               'new_games_16_18': [row for row in rows if 16 <= row['number'] <= 18]}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'methods': {
                  method: {'name': fits[cases[0]['name']][method]['name'],
                           'parameters': fits[cases[0]['name']][method]['parameters']} for method in METHODS},
              'calibration_hash': calibration.content_hash, 'calibration_contexts': len(calibration.records),
              'summary': {cohort: {method: method_metrics(items, method) for method in METHODS}
                          for cohort, items in cohorts.items()}, 'table': table_rows(games), 'games': games, 'players': rows,
              'scope': 'Frozen methods and calibration; reference labels are scoring data only. Original games were used in method exploration, so their repeated comparison is not independent validation. Newly added games are reported separately. No engine or language-model calls.'}
    # Avoid overwriting concurrent analysis or comparing changed source inputs.
    for case in cases:
        for path, digest in case['inputs'].items():
            if _hash(path.read_bytes()) != digest:
                raise RuntimeError(f'Input changed during fitting: {path}. No output was written.')
    if load_calibration().content_hash != calibration.content_hash:
        raise RuntimeError('The frozen calibration changed during fitting; no output was written.')
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        backup = output/'original-analysis'/(case['name']+'.json')
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(case['original_analysis_bytes'])
        for method, fit in fits[case['name']].items():
            export_figures(fit, output/case['name']/method,
                           title=f'{case["name"]} — {fit["name"]}')
    write_json(output/'comparison.json', result)
    (output/'comparison.md').write_text(markdown_table(games), encoding='utf-8')
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(result['table'])
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--games-dir', type=Path, default=ROOT/'games')
    cli.add_argument('--output-dir', type=Path, default=OUTPUT)
    cli.add_argument('--start', type=int, default=0)
    cli.add_argument('--end', type=int, default=18)
    args = cli.parse_args()
    result = run(args.games_dir, args.output_dir, start=args.start, end=args.end)
    print(markdown_table(result['games']))
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
