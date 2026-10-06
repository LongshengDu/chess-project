"""Compare fitted/commercial direction relative to actual Elo, without refitting.

The primary comparison uses displayed fitted ratings and exact signs. No Elo
tolerance turns a small positive or negative difference into an equality. Raw
fitted points are checked separately to expose any display-rounding effects.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'tests/analysis/output/simple-rating-restart/comparison.json'
OUTPUT = ROOT/'tests/analysis/output/arithmetic-coverage-production'
METHOD = 'arithmetic_coverage_common_account'
LABELS = {-1: 'below', 0: 'equal', 1: 'above'}


def _number(value, field):
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValueError(f'{field} must be a supplied finite number.') from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field} must be a supplied finite number.')
    return value


def _sign(value):
    return int(value > 0)-int(value < 0)


def compare(data, *, method=METHOD):
    """Read one saved method's rows; require actual/reference ratings explicitly."""
    aliases = {METHOD, 'arithmetic_coverage'} if method in (METHOD, 'arithmetic_coverage') else {method}
    # The production comparison keeps full unrounded player summaries at game
    # level, while its flat CSV-compatible rows expose selected_rating integers.
    saved_players = {}
    for game in data.get('games', []):
        if isinstance(game, dict) and isinstance(game.get('players'), dict):
            for side, player in game['players'].items():
                if isinstance(player, dict):
                    saved_players[game['game'], side] = player
    rows, identities, source_methods = [], set(), set()
    for saved in data['players']:
        saved_method = saved.get('method', saved.get('method_id'))
        if saved_method not in aliases:
            continue
        source_methods.add(saved_method)
        game, side = saved['game'], saved['side']
        if side not in ('White', 'Black') or (game, side) in identities:
            raise ValueError('Each game must have at most one valid row per side for the selected method.')
        identities.add((game, side))
        actual = _number(saved.get('actual', saved.get('actual_rating')), 'Actual Elo')
        reference = _number(saved['reference'], 'Commercial reference')
        fitted = _number(saved.get('estimate', saved.get('selected_rating')), 'Fitted Elo')
        raw = saved.get('unrounded_estimate', saved_players.get((game, side), {}).get('unrounded_estimate'))
        raw = _number(raw, 'Unrounded fitted Elo') if raw is not None else None
        reference_sign, fitted_sign = (_sign(value-actual) for value in (reference, fitted))
        raw_sign = _sign(raw-actual) if raw is not None else None
        rows.append({'game': game, 'side': side, 'actual': actual, 'reference': reference,
                     'estimate': fitted, 'unrounded_estimate': raw,
                     'reference_minus_actual': reference-actual, 'fitted_minus_actual': fitted-actual,
                     'unrounded_minus_actual': raw-actual if raw is not None else None,
                     'reference_direction': LABELS[reference_sign], 'fitted_direction': LABELS[fitted_sign],
                     'unrounded_direction': LABELS[raw_sign] if raw_sign is not None else None,
                     'matches': reference_sign == fitted_sign,
                     'unrounded_matches': reference_sign == raw_sign if raw_sign is not None else None})
    if not rows:
        raise ValueError(f'No saved player rows found for {method}.')
    rows.sort(key=lambda row: (int(row['game'][4:]), row['side'] == 'Black'))
    matrix = {reference: {fitted: 0 for fitted in LABELS.values()} for reference in LABELS.values()}
    for row in rows:
        matrix[row['reference_direction']][row['fitted_direction']] += 1
    matches = sum(row['matches'] for row in rows)
    return {'method': next(iter(source_methods)) if len(source_methods) == 1 else method,
            'requested_method': method, 'source_method_ids': sorted(source_methods),
            'players': len(rows), 'games': len({row['game'] for row in rows}),
            'criterion': 'Exact sign of displayed fitted Elo minus supplied actual Elo equals exact sign of commercial Elo minus supplied actual Elo; no tolerance.',
            'matches': matches, 'agreement_fraction': matches/len(rows),
            'unrounded_players': sum(row['unrounded_matches'] is not None for row in rows),
            'unrounded_matches': sum(row['unrounded_matches'] is True for row in rows),
            'confusion_matrix': {'rows': 'Commercial direction relative to actual Elo',
                                 'columns': 'Fitted direction relative to actual Elo', 'counts': matrix},
            'mismatches': [row for row in rows if not row['matches']],
            'rounding_direction_changes': [row for row in rows if row['unrounded_direction'] is not None and
                                           row['fitted_direction'] != row['unrounded_direction']],
            'nearest_to_actual': sorted(rows, key=lambda row: min(abs(row['reference_minus_actual']),
                                                                abs(row['fitted_minus_actual'])))[:5],
            'players_detail': rows}


def run(source=INPUT, output=OUTPUT, *, method=METHOD, expected_players=38):
    source, output = Path(source), Path(output)
    content = source.read_bytes()
    result = compare(json.loads(content.decode('utf-8-sig')), method=method)
    if expected_players is not None and result['players'] != expected_players:
        raise ValueError(f'Expected {expected_players} selected player rows, found {result["players"]}.')
    result.update(source=str(source.resolve()), source_sha256=sha256(content).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    (output/'direction.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    with (output/'direction.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['players_detail'][0]))
        writer.writeheader()
        writer.writerows(result['players_detail'])
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, default=INPUT)
    cli.add_argument('--output-dir', type=Path, default=OUTPUT)
    cli.add_argument('--method', default=METHOD)
    cli.add_argument('--expected-players', type=int, default=38)
    args = cli.parse_args()
    result = run(args.input, args.output_dir, method=args.method, expected_players=args.expected_players)
    print(json.dumps({key: value for key, value in result.items() if key != 'players_detail'}, indent=2))


if __name__ == '__main__':
    main()
