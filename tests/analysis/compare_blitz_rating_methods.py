"""Compare three fixed rating methods with the current PGN Blitz-scale labels.

Reuse audited engine evidence, but explicitly replace historical account-rating
overrides with the revised PGN ratings. References enter scoring after fitting.
Normal game analyses and production settings are preserved. Missing analyses
must be prepared with the shared full-game pipeline before running this script.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from analysis.player_rating.bayesian_shared_curve import ARGS, Rating as BayesianRating, SharedCurve
from analysis.player_rating.calibration import load_calibration
from analysis.player_rating.context import saved_context
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.figures import export_figures
from analysis.settings import CONFIG
from tests.analysis.compare_joint_rating_directions import matches_pair, sign
from tests.analysis.compare_rating_methods import load_cases


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/lichess-blitz-methods'
SIDES = ('White', 'Black')
METHOD_NAMES = {
    'shared_curve_intersection': 'Shared curve intersection',
    'bayesian_shared_curve': 'Bayesian shared curve',
    'uncertainty_weighted_shared_curve': 'Uncertainty-weighted shared curve',
}


def intersection(knots, accuracy):
    """Return an actual crossing on 0--3200, never an endpoint-clipped estimate."""
    curve = SharedCurve(knots)
    low, high = ARGS.rating_range
    native_low, native_high = np.asarray(curve([600., 2600.]), dtype=float)
    result = {'estimate': None, 'intersection_status': 'missing', 'intersection_bound': None,
              'intersection_interval': None, 'edge': None, 'native_accuracy_min': float(native_low),
              'native_accuracy_max': float(native_high)}
    if accuracy is None:
        return result
    accuracy = float(accuracy)
    if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError('Observed accuracy must be finite and lie in [0, 100].')
    result['edge'] = not native_low <= accuracy <= native_high
    start, end = map(float, curve([low, high]))
    if accuracy > end:
        return {**result, 'intersection_status': 'above_support', 'intersection_bound': high}
    if accuracy < start:
        return {**result, 'intersection_status': 'below_support', 'intersection_bound': low}
    if end-start < 1e-10:
        return {**result, 'intersection_status': 'flat', 'intersection_interval': [low, high]}
    nodes = np.r_[low, np.arange(600., 2601., 100.), high]
    matching = np.isclose(curve(nodes), accuracy, atol=1e-12, rtol=0)
    flat = np.flatnonzero(matching[:-1] & matching[1:])
    if len(flat):
        return {**result, 'intersection_status': 'nonunique',
                'intersection_interval': [float(nodes[flat[0]]), float(nodes[flat[-1]+1])]}
    estimate = float(brentq(lambda rating: float(curve(rating))-accuracy, low, high, xtol=1e-9))
    return {**result, 'estimate': estimate,
            'intersection_status': 'extrapolated' if result['edge'] else 'native'}


def current_pgn_context(case):
    """Current explicit user revisions supersede saved account overrides here."""
    analysis = {**case['analysis'], 'headers': dict(case['game'].headers), 'rating_account_overrides': {}}
    evidence = validate_evidence(saved_context(case['evidence'], analysis))
    for side in SIDES:
        value = case['game'].headers.get(side+'Elo')
        if value is None or not value.isdigit() or evidence[side]['actual_rating'] != int(value):
            raise ValueError(f'{case["name"]} {side}: a valid revised actual Elo is required.')
    return evidence


def metrics(rows):
    available = [row for row in rows if row['estimate'] is not None]
    errors = [abs(row['estimate']-row['reference']) for row in available]
    native = [abs(row['estimate']-row['reference']) for row in available if not row['edge']]
    edges = [abs(row['estimate']-row['reference']) for row in available if row['edge']]
    by_game = {}
    for row in rows:
        by_game.setdefault(row['game'], {})[row['side']] = row
    pairs = [pair for pair in by_game.values()
             if all(side in pair and pair[side]['estimate'] is not None for side in SIDES)]
    return {
        'estimated_players': len(available), 'missing_players': len(rows)-len(available),
        'mae': float(np.mean(errors)) if errors else None,
        'maximum_error': max(errors) if errors else None,
        'native_mae': float(np.mean(native)) if native else None,
        'edge_mae': float(np.mean(edges)) if edges else None,
        'edge_players': sum(bool(row['edge']) for row in rows),
        'edge_estimated_players': sum(bool(row['edge']) for row in available),
        'white_black_comparable_games': len(pairs),
        'white_black_matches': sum(matches_pair(pair['White']['reference']-pair['Black']['reference'],
                                                pair['White']['display_estimate']-pair['Black']['display_estimate'])
                                   for pair in pairs),
        'above_below_matches': sum(sign(row['display_estimate']-row['actual']) ==
                                   sign(row['reference']-row['actual']) for row in available),
    }


def intersection_ordering(rows):
    """Distinguish numeric rating comparisons from direction without a crossing."""
    pairs = {}
    for row in rows:
        if row['method'] == 'shared_curve_intersection':
            pairs.setdefault(row['game'], {})[row['side']] = row
    checks = []
    for game, pair in sorted(pairs.items(), key=lambda item: int(item[0][4:])):
        white, black = pair['White'], pair['Black']
        reference_gap = white['reference']-black['reference']
        finite = all(row['estimate'] is not None for row in pair.values())
        accuracy_gap = (white['average_accuracy']-black['average_accuracy']
                        if all(row['average_accuracy'] is not None for row in pair.values()) else None)
        gap = white['estimate']-black['estimate'] if finite else None
        if finite:
            matched = matches_pair(reference_gap, gap)
            displayed = matches_pair(reference_gap, white['display_estimate']-black['display_estimate'])
        else:
            # A commercial tie needs a numerical gap; accuracy alone cannot
            # establish that |White Elo - Black Elo| is below 50.
            matched = sign(accuracy_gap) == sign(reference_gap) if reference_gap and accuracy_gap is not None else None
            displayed = None
        checks.append({'game': game, 'reference_white': white['reference'], 'reference_black': black['reference'],
                       'reference_gap': reference_gap, 'white_intersection': white['estimate'],
                       'black_intersection': black['estimate'], 'intersection_gap': gap,
                       'white_status': white['intersection_status'], 'black_status': black['intersection_status'],
                       'accuracy_gap': accuracy_gap,
                       'basis': 'finite_intersections' if finite else 'accuracy_direction_without_finite_pair',
                       'ordering_match': matched, 'display_ordering_match': displayed})
    finite = [row for row in checks if row['basis'] == 'finite_intersections']
    directional = [row for row in checks if row['basis'] != 'finite_intersections']
    return {'finite_games': len(finite), 'finite_matches': sum(row['ordering_match'] for row in finite),
            'direction_only_games': len(directional),
            'direction_only_matches': sum(row['ordering_match'] is True for row in directional),
            'unresolved_games': sum(row['ordering_match'] is None for row in checks),
            'scope': 'Finite intersections are tested numerically. Missing-crossing cases compare observed accuracy direction only; they are not fabricated Elo estimates and cannot verify a commercial tie tolerance.',
            'games': checks}


def run(output=OUTPUT, *, end=None):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    # Kept as an unreachable historical record; these implementations were removed.
    from analysis.player_rating.uncertainty_ensemble import Rating as UncertaintyRating
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'tests/analysis/output'):
        raise ValueError('Comparison output must stay under tests/analysis/output.')
    numbers = sorted(int(match[1]) for path in (ROOT/'games').glob('*.pgn')
                     if (match := re.fullmatch(r'game(\d+)\.pgn', path.name)))
    if end is not None:
        numbers = [number for number in numbers if number <= end]
    if not numbers:
        raise ValueError('No game PGNs found.')
    cases = load_cases(ROOT/'games', numbers, Path(CONFIG['ANALYSIS']['CACHE_DIR']))
    protected = {path for case in cases for path in case['inputs']}
    protected.update((ROOT/'analysis').rglob('*.py'))
    protected.update((ROOT/'config.yaml', ROOT/'analysis/player_rating/data/maia_accuracy_calibration.json'))
    for case in cases:
        protected.update((case['analysis_path'].parent/'player-rating').glob('*'))
    before = {str(path): sha256(path.read_bytes()).hexdigest() for path in protected if path.is_file()}
    calibration = load_calibration()
    estimators = {'bayesian_shared_curve': BayesianRating(),
                  'uncertainty_weighted_shared_curve': UncertaintyRating(calibration=calibration)}
    prepared, started = [], perf_counter()
    for case in cases:
        evidence = current_pgn_context(case)
        fits = {method: estimator.fit(evidence) for method, estimator in estimators.items()}
        bayesian = fits['bayesian_shared_curve']
        knots = bayesian['diagnostics']['curve']['monotone_expected_accuracy']
        intersections = {side: intersection(knots, bayesian['players'][side]['average_accuracy']) for side in SIDES}
        prepared.append((case, evidence, fits, intersections))
        print(f'{case["name"]}: three methods calculated with current PGN ratings.', flush=True)
    # Reference labels are first extracted after all three predictions finish.
    rows, provenance = [], []
    for case, evidence, fits, intersections in prepared:
        provenance.append({'game': case['name'], 'evidence_key': case['key'], 'audit': case['audit'],
                           'ignored_saved_rating_overrides': case['analysis'].get('rating_account_overrides', {}),
                           'actual_ratings': {side: evidence[side]['actual_rating'] for side in SIDES}})
        for side in SIDES:
            reference = case['game'].headers.get(side+'EloEstimate')
            if reference is None or not reference.isdigit():
                raise ValueError(f'{case["name"]} {side}: revised commercial estimate is required.')
            for method in METHOD_NAMES:
                cross = intersections[side]
                point = cross['estimate'] if method == 'shared_curve_intersection' else fits[method]['players'][side]['unrounded_estimate']
                rows.append({'game': case['name'], 'number': case['number'], 'side': side, 'method': method,
                             'actual': evidence[side]['actual_rating'], 'reference': int(reference),
                             'estimate': point, 'display_estimate': int(np.floor(point+.5)) if point is not None else None,
                             'average_accuracy': fits['bayesian_shared_curve']['players'][side]['average_accuracy'],
                             'edge': cross['edge'], 'intersection_status': cross['intersection_status'],
                             'intersection_bound': cross['intersection_bound'] if method == 'shared_curve_intersection' else None,
                             'intersection_interval': cross['intersection_interval'] if method == 'shared_curve_intersection' else None})
    common = {(row['game'], row['side']) for row in rows
              if row['method'] == 'shared_curve_intersection' and row['estimate'] is not None}
    summary = {method: metrics([row for row in rows if row['method'] == method]) for method in METHOD_NAMES}
    for method in METHOD_NAMES:
        subset = [row for row in rows if row['method'] == method and (row['game'], row['side']) in common]
        summary[method]['common_intersection_subset_mae'] = metrics(subset)['mae'] if subset else None
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'games': len(cases), 'game_numbers': numbers,
        'rating_scale': 'Lichess Blitz (revised actual and commercial PGN headers supplied by the user)',
        'method_order': list(METHOD_NAMES), 'method_names': METHOD_NAMES,
        'implementation': {'shared_curve_intersection': 'Direct inverse of the Bayesian arithmetic curve on 0--3200; no prior or account blend.',
                           'bayesian_shared_curve': 'analysis.player_rating.bayesian_shared_curve.Rating',
                           'uncertainty_weighted_shared_curve': 'analysis.player_rating.uncertainty_ensemble.Rating'},
        'parameters': {method: estimator.parameters for method, estimator in estimators.items()},
        'method_versions': {method: estimator.version for method, estimator in estimators.items()},
        'scope': 'Fixed methods; current PGN account ratings supersede historical saved overrides. Shared audited engine evidence. No reference-fitted parameters or calibration changes. Graph MAE uses unrounded estimates. Undefined intersections are excluded from errors and marked explicitly, never replaced with boundary ratings.',
        'edge_definition': 'Observed arithmetic accuracy outside the measured shared curve at rating600--2600; applies equally to all methods and includes new edge players.',
        'ordering_rule': 'Commercial equality requires |fitted W-B|<50. Otherwise require the same sign, regardless of magnitude.',
        'players': rows, 'summary': summary, 'provenance': provenance,
        'shared_curve_ordering': intersection_ordering(rows),
        'elapsed_fitting_seconds': perf_counter()-started,
        'input_sha256': before, 'calibration_hash': calibration.content_hash,
    }
    after = {path: sha256(Path(path).read_bytes()).hexdigest() for path in before}
    if after != before:
        raise RuntimeError('A protected game, evidence, configuration or implementation changed while fitting.')
    result['protected_inputs_unchanged'] = len(before)
    output.mkdir(parents=True, exist_ok=True)
    for case, evidence, fits, intersections in prepared:
        for method, fit in fits.items():
            export_figures(fit, output/case['name']/method, title=f'{case["name"]} — Lichess Blitz rating scale')
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output/'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ordering = result['shared_curve_ordering']['games']
    with (output/'intersection-ordering.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordering[0]))
        writer.writeheader()
        writer.writerows(ordering)
    from tests.analysis.blitz_rating_figures import export
    artifact = export(result, output)
    print(json.dumps(summary, indent=2))
    print(f'Combined SVG: {artifact}', flush=True)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output-dir', type=Path, default=OUTPUT)
    cli.add_argument('--end', type=int, help='Optional highest game number for a limited diagnostic run.')
    args = cli.parse_args()
    run(args.output_dir, end=args.end)
