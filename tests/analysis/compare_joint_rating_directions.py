"""Rescore saved estimates against paired and account-relative directions.

This evaluator reads completed experiments and PGN labels only. It never calls
an estimator, fits a parameter, runs an engine or modifies production outputs.
The 19-game and historical 16-game cohorts remain separate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import chess.pgn


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/joint-rating-directions'
SIDES = ('White', 'Black')


def sign(value):
    return int(value > 0)-int(value < 0)


def matches_pair(reference_gap, fitted_gap):
    """Reference ties require |gap| <50; non-ties require sign, at any magnitude."""
    return abs(fitted_gap) < 50 if reference_gap == 0 else sign(reference_gap) == sign(fitted_gap)


def saved_rows(data):
    """Read the three archived prediction schemas without reading fitted scores."""
    if not isinstance(data, dict):
        return []
    for key in ('players', 'rows'):
        rows = data.get(key)
        if isinstance(rows, list) and rows and all(
                isinstance(row, dict) and {'game', 'side', 'method', 'estimate'} <= row.keys() for row in rows):
            return rows
    rows = []
    for game in data.get('games', []):
        if not isinstance(game, dict) or 'game' not in game:
            continue
        for method, pair in game.get('predictions', {}).items():
            for side in SIDES:
                rows.append({'game': game['game'], 'side': side, 'method': method,
                             'estimate': pair[side], 'reference': game.get('reference', {}).get(side)})
        if 'white_estimate' in game and 'black_estimate' in game and isinstance(data.get('method'), str):
            for side in SIDES:
                rows.append({'game': game['game'], 'side': side, 'method': data['method'],
                             'estimate': game[side.lower()+'_estimate'],
                             'reference': game.get(side.lower()+'_reference'),
                             'actual': game.get(side.lower()+'_actual')})
    return rows


def score(rows, *, rounded):
    """Score fixed estimates, with no tolerance for account-relative signs."""
    failures, pairs, errors = [], defaultdict(dict), []
    for row in rows:
        value = math.floor(row['unrounded_estimate']+.5) if rounded else row['unrounded_estimate']
        pairs[row['game']][row['side']] = {**row, 'value': value}
        errors.append(abs(value-row['reference']))
        if sign(value-row['actual']) != sign(row['reference']-row['actual']):
            failures.append({'game': row['game'], 'side': row['side'], 'estimate': value,
                             'actual': row['actual'], 'reference': row['reference'],
                             'estimate_minus_actual': value-row['actual'],
                             'reference_minus_actual': row['reference']-row['actual']})
    pair_failures = []
    joint_games = 0
    for game, pair in pairs.items():
        if set(pair) != set(SIDES):
            raise ValueError(f'{game} is missing one player.')
        white, black = pair['White'], pair['Black']
        reference_gap = white['reference']-black['reference']
        fitted_gap = white['value']-black['value']
        ok = matches_pair(reference_gap, fitted_gap)
        if not ok:
            pair_failures.append({'game': game, 'reference_gap': reference_gap, 'estimate_gap': fitted_gap})
        if ok and all(sign(row['value']-row['actual']) == sign(row['reference']-row['actual'])
                      for row in pair.values()):
            joint_games += 1
    return {'white_black_matches': len(pairs)-len(pair_failures), 'games': len(pairs),
            'account_direction_matches': len(rows)-len(failures), 'players': len(rows),
            'joint_game_matches': joint_games, 'all_joint_requirements': not failures and not pair_failures,
            'mae': sum(errors)/len(errors), 'maximum_error': max(errors),
            'account_direction_failures': failures, 'white_black_failures': pair_failures}


def canonical_rows(rows, labels):
    """Validate archived labels against current PGNs and retain raw estimates."""
    output, seen = [], set()
    for row in rows:
        key = (row['game'], row['side'])
        if key in seen:
            raise ValueError(f'Duplicate player prediction: {key}.')
        seen.add(key)
        reference, actual = labels[key]['reference'], labels[key]['actual']
        for field, expected in (('reference', reference), ('actual', actual)):
            if row.get(field) is not None and float(row[field]) != expected:
                raise ValueError(f'{key}: saved {field} does not match current PGN.')
        value = row.get('unrounded_estimate', row['estimate'])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key}: estimate is not a finite number.')
        output.append({'game': key[0], 'side': key[1], 'unrounded_estimate': float(value),
                       'reference': reference, 'actual': actual})
    return sorted(output, key=lambda row: (int(row['game'][4:]), row['side']))


def run(output=OUTPUT):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'tests/analysis/output'):
        raise ValueError('Diagnostic output must stay under tests/analysis/output.')
    labels, protected = {}, {}
    for number in range(19):
        path = ROOT/f'games/game{number}.pgn'
        protected[str(path)] = sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig') as stream:
            headers = chess.pgn.read_headers(stream)
        for side in SIDES:
            labels[(f'game{number}', side)] = {'reference': int(headers[side+'EloEstimate']),
                                             'actual': int(headers[side+'Elo'])}
    primary = ROOT/'tests/analysis/output/simple-rating-restart/comparison.json'
    sources = [primary, *sorted((ROOT/'tests/analysis/output/universal-rating-methods').rglob('*.json'))]
    found, ignored = {}, []
    for path in sources:
        protected[str(path)] = sha256(path.read_bytes()).hexdigest()
        source = str(path.relative_to(ROOT)).replace('\\', '/')
        data = json.loads(path.read_text(encoding='utf-8'))
        rows = saved_rows(data)
        if not rows:
            ignored.append({'source': source, 'reason': 'No per-player predictions in this artifact.'})
            continue
        by_method = defaultdict(list)
        for row in rows:
            by_method[row['method']].append(row)
        for method, selected in by_method.items():
            canonical = canonical_rows(selected, labels)
            games = tuple(sorted({r['game'] for r in canonical}, key=lambda name: int(name[4:])))
            if games not in (tuple(f'game{i}' for i in range(19)), tuple(f'game{i}' for i in range(16))):
                raise ValueError(f'{source}: unexpected cohort for {method}: {games}.')
            fingerprint = sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
            key = (method, fingerprint)
            if key in found:
                found[key]['sources'].append(source)
                continue
            found[key] = {'method': method, 'prediction_hash': fingerprint, 'cohort': f'games0-{len(games)-1}',
                          'sources': [source], 'rounded': score(canonical, rounded=True),
                          'unrounded': score(canonical, rounded=False)}
    cohorts = {}
    for cohort in ('games0-18', 'games0-15'):
        methods = [row for row in found.values() if row['cohort'] == cohort]
        methods.sort(key=lambda row: (
            row['rounded']['white_black_matches'] != row['rounded']['games'],
            -row['rounded']['account_direction_matches'], -row['rounded']['white_black_matches'],
            row['rounded']['mae'], row['method']))
        cohorts[cohort] = {'method_versions': len(methods), 'distinct_method_names': len({r['method'] for r in methods}),
                           'perfect_rounded': [r['method'] for r in methods if r['rounded']['all_joint_requirements']],
                           'perfect_unrounded': [r['method'] for r in methods if r['unrounded']['all_joint_requirements']],
                           'ranking': methods}
    after = {path: sha256(Path(path).read_bytes()).hexdigest() for path in protected}
    if after != protected:
        raise RuntimeError('An input archive or PGN changed during read-only rescoring.')
    result = {
        'scope': 'Saved predictions only; no estimator or engine was run and no parameters were fit. Historical16-game and full19-game cohorts are scored separately. Duplicate identical method versions are collapsed, with all source files retained. Selecting among already evaluated methods is retrospective and not independent validation.',
        'rules': {'white_black': 'Exactly equal commercial ratings allow a fitted absolute gap <50; otherwise require the same nonzero sign.',
                  'relative_to_actual': 'Require sign(fitted - actual) = sign(reference - actual), including exact zero; no tolerance.',
                  'primary': 'Displayed half-up integer estimates; unrounded decisions reported separately.',
                  'actual_source': 'Current PGN WhiteElo/BlackElo, checked against archived actual values where present.'},
        'cohorts': cohorts, 'ignored_artifacts': ignored, 'input_sha256': protected,
        'protected_inputs_unchanged': len(protected)}
    output.mkdir(parents=True, exist_ok=True)
    (output/'archive-ranking.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    table = []
    for cohort, data in cohorts.items():
        for row in data['ranking']:
            rounded, raw = row['rounded'], row['unrounded']
            table.append({'cohort': cohort, 'method': row['method'], 'versions_hash': row['prediction_hash'],
                          'white_black': f"{rounded['white_black_matches']}/{rounded['games']}",
                          'vs_actual': f"{rounded['account_direction_matches']}/{rounded['players']}",
                          'joint_games': f"{rounded['joint_game_matches']}/{rounded['games']}",
                          'mae': rounded['mae'], 'unrounded_vs_actual': raw['account_direction_matches'],
                          'perfect': rounded['all_joint_requirements']})
    with (output/'archive-ranking.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for cohort, data in cohorts.items():
        print(cohort, 'method versions', data['method_versions'], 'perfect', data['perfect_rounded'])
        for row in data['ranking'][:8]:
            score_row = row['rounded']
            print(row['method'], score_row['white_black_matches'], score_row['account_direction_matches'], round(score_row['mae'], 2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
