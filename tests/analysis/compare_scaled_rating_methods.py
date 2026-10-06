"""Evaluate frozen rating rules with explicit source/canonical scale conversion.

Current PGN Site and TimeControl supply the rating scale; current actual Elo
headers supply accounts. Candidate inference always takes full-precision Lichess
Blitz accounts and unchanged canonical Maia evidence. Commercial references are
joined only after predictions and four-scale representation checks are saved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
import json
import math
from pathlib import Path
import re
from time import perf_counter

import chess.pgn
import numpy as np

from analysis import elo_convert
from analysis.player_rating.bayesian_shared_curve import SharedCurve, fit_pair
from analysis.player_rating.context import attach_context
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.service import evidence_cache_path
from analysis.settings import CONFIG
from tests.analysis.compare_blitz_rating_methods import current_pgn_context, intersection
from tests.analysis.compare_intuitive_curves import DEFAULT_MODULES, dump, metrics
from tests.analysis.compare_shared_curve_games import audit_evidence


ROOT = Path(__file__).resolve().parents[2]
SIDES = ('White', 'Black')
SCALES = ('lb', 'lr', 'cb', 'cr')
CANONICAL = 'lb'
# A legacy local-noise candidate uses a 0.001-Elo finite-difference derivative.
# Cancellation can magnify sub-picorating input changes into ~1e-6 Elo at its
# final point. Retain the measured discrepancy, allowing 1e-5 for inference and
# the much stricter 1e-8 for conversion alone; no candidate formula is changed.
AUDIT_TOLERANCE = 1e-5
POPULATION_CONTROLS = frozenset({
    'population_affine_common_account5', 'population_affine_account_prior',
    'population_percentile_account_prior', 'simple_global_population_inverse',
    'simple_global_population_elo_odds',
})
BASELINES = frozenset({'shared_curve_intersection', 'bayesian_shared_curve',
                        'arithmetic_coverage', 'uncertainty_ensemble'})


def scope(method):
    """Declare mathematical scope, independently of benchmark performance."""
    return ('population_control' if method in POPULATION_CONTROLS else
            'baseline' if method in BASELINES else 'target_curve')


def analysis_matches_game(analysis, game):
    """Identify a saved game from its complete positions and moves, not filename."""
    moves = list(game.mainline_moves())
    rows = analysis.get('moves', [])
    if len(rows) != len(moves):
        return False
    board = game.board()
    for move, row in zip(moves, rows, strict=True):
        if row.get('fen') != board.fen() or row.get('played', {}).get('move') != move.uci():
            return False
        board.push(move)
    return True


def load_cases(games_dir, numbers, cache):
    """Resolve renamed PGNs to exact existing analyses, then audit their evidence.

This does not repair or relabel any saved file. Source paths and bytes remain
available for an explicit later production refresh, after the comparison ends.
"""
    sources = {}
    for path in sorted((games_dir/'output').glob('game*-full/analysis.json')):
        content = path.read_bytes()
        sources[path] = (content, json.loads(content.decode('utf-8-sig')))
    cases = []
    for number in numbers:
        path = games_dir/f'game{number}.pgn'
        pgn_bytes = path.read_bytes()
        with path.open(encoding='utf-8-sig') as stream:
            game = chess.pgn.read_game(stream)
        if game is None or game.errors or not list(game.mainline_moves()):
            raise ValueError(f'{path.name} must contain a valid played game.')
        expected = games_dir/'output'/f'{path.stem}-full/analysis.json'
        if expected in sources and analysis_matches_game(sources[expected][1], game):
            analysis_path = expected
        else:
            matches = [source for source, (_, analysis) in sources.items() if analysis_matches_game(analysis, game)]
            if len(matches) != 1:
                raise ValueError(f'{path.name}: expected one exact saved analysis; found {len(matches)}.')
            analysis_path = matches[0]
        analysis_bytes, analysis = sources[analysis_path]
        key = analysis.get('rating_fit', {}).get('evidence_key')
        evidence_path = evidence_cache_path(cache, key)
        evidence_bytes = evidence_path.read_bytes()
        evidence = validate_evidence(json.loads(evidence_bytes.decode('utf-8-sig')))
        audit = audit_evidence(game, analysis, evidence)
        inputs = {path: sha256(pgn_bytes).hexdigest(), analysis_path: sha256(analysis_bytes).hexdigest(),
                  evidence_path: sha256(evidence_bytes).hexdigest()}
        cases.append({'number': number, 'name': path.stem, 'game': game, 'analysis': analysis,
                      'evidence': evidence, 'key': key, 'analysis_path': analysis_path,
                      'original_analysis_bytes': analysis_bytes, 'inputs': inputs,
                      'audit': {**audit, 'source_analysis': str(analysis_path),
                                'filename_remapped': analysis_path != expected}})
    return cases


def convert_point(value, source, target):
    """Keep missing roots missing; extend the supplied converter explicitly."""
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError('A finite rating or missing estimate is required.')
    return float(elo_convert.convert(float(value), source, target, extrapolate=True))


def converted_metadata(value, source, target):
    canonical = convert_point(value, source, CANONICAL)
    return {'value': convert_point(value, source, target), 'canonical_lb': canonical,
            'extrapolated': canonical is not None and not elo_convert.LB_MIN <= canonical <= elo_convert.LB_MAX}


def canonical_context(evidence, actuals, source):
    """Copy context; never round accounts or change qualities/policy ratings."""
    canonical = {side: convert_point(actuals[side], source, CANONICAL) for side in SIDES}
    return attach_context(evidence, canonical), canonical


def baseline_predictions(evidence):
    curve = fit_pair(evidence['White'], evidence['Black'])
    accuracies = {side: curve['players'][i]['average_accuracy'] for i, side in enumerate(SIDES)}
    crossings = {side: intersection(curve['monotone_expected_accuracy'], accuracies[side]) for side in SIDES}
    predictions = {
        'shared_curve_intersection': {side: crossings[side]['estimate'] for side in SIDES},
        'bayesian_shared_curve': {side: curve['players'][i]['unrounded_estimate'] for i, side in enumerate(SIDES)},
    }
    return predictions, curve, accuracies, crossings


def prepare_predictors(modules, evidence, actuals):
    """Reuse only declared account-independent preparation, as the old runner did."""
    predictors = []
    for module in modules:
        if hasattr(module, 'prepare') and hasattr(module, 'predict_prepared'):
            prepared = module.prepare(evidence, actuals)
            predictors.append(lambda ratings, module=module, prepared=prepared:
                              module.predict_prepared(prepared, ratings))
        else:
            predictors.append(lambda ratings, module=module:
                              module.predict(attach_context(evidence, ratings), ratings))
    return predictors


def predict_all(evidence, actuals, predictors):
    predictions, curve, accuracies, crossings = baseline_predictions(evidence)
    for predict in predictors:
        values = predict(actuals)
        if predictions.keys() & values.keys():
            raise ValueError('Duplicate candidate method names.')
        for pair in values.values():
            if set(pair) != set(SIDES) or any(value is not None and
                    (isinstance(value, bool) or not math.isfinite(value)) for value in pair.values()):
                raise ValueError('Every method must return two finite or missing player estimates.')
        predictions.update(values)
    return predictions, curve, accuracies, crossings


def check_representations(evidence, actuals, predictors, base):
    """Re-express identical accounts in four scales, normalize, and refit.

This is coordinate equivariance, not statistical validity of the empirical
conversion model. Prepared Maia-only data are reused; no +/-200 grid is run.
"""
    checks = []
    for scale in SCALES:
        represented = {side: convert_point(actuals[side], CANONICAL, scale) for side in SIDES}
        normalized_evidence, normalized = canonical_context(evidence, represented, scale)
        predictions, _, _, _ = predict_all(normalized_evidence, normalized, predictors)
        if predictions.keys() != base.keys():
            raise ValueError('Method inventory changed with rating-scale representation.')
        for method, pair in base.items():
            for side in SIDES:
                original, repeated = pair[side], predictions[method][side]
                if (original is None) != (repeated is None):
                    raise ValueError('Missing-estimate status changed with scale representation.')
                direct = convert_point(original, CANONICAL, scale)
                repeated_target = convert_point(repeated, CANONICAL, scale)
                roundtrip = convert_point(direct, scale, CANONICAL)
                checks.append({'scale': scale, 'method': method, 'side': side,
                               'represented_actual': represented[side],
                               'actual_roundtrip_error': abs(normalized[side]-actuals[side]),
                               'canonical_prediction_error': None if original is None else abs(repeated-original),
                               'target_commutation_error': None if original is None else abs(repeated_target-direct),
                               'point_roundtrip_error': None if original is None else abs(roundtrip-original)})
    return checks


def audit_summary(checks):
    keys = ('actual_roundtrip_error', 'canonical_prediction_error',
            'target_commutation_error', 'point_roundtrip_error')
    maxima = {name: max((row[name] for row in checks if row[name] is not None), default=0.) for name in keys}
    tolerances = {name: 1e-8 if name in ('actual_roundtrip_error', 'point_roundtrip_error') else AUDIT_TOLERANCE for name in keys}
    return {'scales': list(SCALES), 'checks': len(checks), 'tolerance': AUDIT_TOLERANCE, 'tolerances': tolerances,
            'maximum_errors': maxima, 'passed': all(value <= tolerances[name] for name, value in maxima.items()),
            'strict_1e_7_passed': all(value <= 1e-7 for value in maxima.values()),
            'numerical_note': 'The fixed local-noise-prior candidate estimates its curve slope by a 0.001-Elo finite difference. Roundoff cancellation can amplify ~1e-13 account changes into ~1e-6 output Elo; formulas are unchanged.',
            'interpretation': 'Equivalent coordinate representations of the same canonical accounts; not validation of conversion accuracy.'}


def method_metadata(methods, declarations):
    result = {}
    for method in methods:
        description = next((declaration['methods'][method] for declaration in declarations.values()
                            if isinstance(declaration['methods'].get(method), str)), '')
        result[method] = {'scope': scope(method), 'label': method.replace('_', ' ').capitalize(),
                          'description': description}
    result['shared_curve_intersection']['description'] = 'Unregularized crossing of observed arithmetic accuracy and the target shared curve; unavailable if no crossing exists.'
    result['bayesian_shared_curve']['description'] = 'Original conditional posterior median using the target arithmetic shared curve and existing tapered rating prior.'
    return result


def run(names, output, *, figures=True):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'tests/analysis/output'):
        raise ValueError('Research output must remain under tests/analysis/output.')
    output.mkdir(parents=True, exist_ok=True)
    modules = [import_module('tests.analysis.'+name) for name in names]
    declarations = {module.__name__: {'methods': module.describe(),
                    'sha256': sha256(Path(module.__file__).read_bytes()).hexdigest()} for module in modules}
    numbers = sorted(int(match[1]) for path in (ROOT/'games').glob('game*.pgn')
                     if (match := re.fullmatch(r'game(\d+)\.pgn', path.name)))
    cases = load_cases(ROOT/'games', numbers, Path(CONFIG['ANALYSIS']['CACHE_DIR']))
    protected = {path: value for case in cases for path, value in case['inputs'].items()}
    # Guard numerical dependencies, scoring and rendering, not unrelated app
    # integration code or unit tests being developed concurrently.
    numerical_sources = [path for path in (ROOT/'analysis/player_rating').glob('*.py')
                         if path.name not in {'__init__.py', 'service.py', 'scale.py', 'figures.py'}]
    research_sources = [Path(module.__file__) for module in modules]
    research_sources += [ROOT/'tests/analysis'/name for name in (
        'simple_curve_account.py', 'compare_scaled_rating_methods.py', 'scaled_rating_figures.py',
        'compare_intuitive_curves.py', 'compare_joint_rating_directions.py', 'compare_blitz_rating_methods.py')]
    for path in [ROOT/'config.yaml', *numerical_sources, *research_sources,
                 *(ROOT/'analysis'/name for name in ('elo_convert.py', 'lichess_accuracy.py', 'position_evaluation.py', 'cache.py', 'settings.py')),
                 ROOT/'analysis/player_rating/data/maia_accuracy_calibration.json']:
        protected[path] = sha256(path.read_bytes()).hexdigest()
    design = {'created_utc': datetime.now(timezone.utc).isoformat(), 'declarations': declarations,
              'conversion_model': elo_convert.MODEL_VERSION, 'canonical_scale': CANONICAL,
              'extrapolate': True, 'scope_rule': 'Population-only controls are separate from target-curve methods; every declared method is retained.',
              'source_guard_scope': 'Current PGNs, exact source analyses, evidence caches, calibration asset, numerical models, converter, candidate and scoring/plotting modules; unrelated app integration code and unit tests are excluded.',
              'rule': 'Frozen formulas and conversion coefficients. Current reference headers are joined after predictions. Actual PGN ratings replace stale saved overrides.'}
    dump(output/'design.json', design)
    predicted, contexts, checks = [], [], []
    started = perf_counter()
    for case in cases:
        scale = elo_convert.resolve_scale(dict(case['game'].headers))
        raw_evidence = current_pgn_context(case)
        original_actual = {side: raw_evidence[side]['actual_rating'] for side in SIDES}
        evidence, actual = canonical_context(raw_evidence, original_actual, scale['scale'])
        predictors = prepare_predictors(modules, evidence, actual)
        predictions, curve, accuracies, crossings = predict_all(evidence, actual, predictors)
        checks.extend({'game': case['name'], **row} for row in check_representations(evidence, actual, predictors, predictions))
        canonical_axis = np.arange(0., 3201., 10.)
        contexts.append({'game': case['name'], 'number': case['number'], 'scale': scale,
                         'actual': original_actual, 'canonical_actual': actual, 'accuracy': accuracies,
                         'knots': curve['monotone_expected_accuracy'],
                         'canonical_native_axis': list(range(600, 2601, 100)),
                         'source_native_axis': [convert_point(value, CANONICAL, scale['scale']) for value in range(600, 2601, 100)],
                         'canonical_axis': canonical_axis.tolist(),
                         'source_axis': [convert_point(float(value), CANONICAL, scale['scale']) for value in canonical_axis],
                         'shared_accuracy': SharedCurve(curve['monotone_expected_accuracy'])(canonical_axis).tolist(),
                         'variance': curve['likelihood']['accuracy_variance'], 'intersection': crossings})
        for method, pair in predictions.items():
            for side in SIDES:
                point = converted_metadata(pair[side], CANONICAL, scale['scale'])
                crossing = crossings[side]
                predicted.append({'game': case['name'], 'number': case['number'], 'side': side, 'method': method,
                                  'source_scale': scale['scale'], 'scale_name': scale['name'],
                                  'actual': original_actual[side], 'canonical_actual': actual[side],
                                  'actual_conversion_extrapolated': not elo_convert.LB_MIN <= actual[side] <= elo_convert.LB_MAX,
                                  'accuracy': accuracies[side], 'estimate': point['value'],
                                  'canonical_estimate': pair[side], 'conversion_extrapolated': point['extrapolated'],
                                  'edge': crossing['edge'], 'intersection_status': crossing['intersection_status'],
                                  'canonical_intersection_bound': crossing['intersection_bound'],
                                  'source_intersection_bound': convert_point(crossing['intersection_bound'], CANONICAL, scale['scale']),
                                  'own_rating_change': None, 'both_ratings_change': None})
        print(f'{case["name"]}: {scale["name"]}; {len(predictions)} canonical methods and four-scale checks finished.', flush=True)
    dump(output/'predictions.json', predicted)
    dump(output/'contexts.json', contexts)
    invariance = audit_summary(checks)
    dump(output/'scale-invariance.json', {**invariance, 'rows': checks})
    if not invariance['passed']:
        raise RuntimeError('Scale representation audit exceeded declared floating-point tolerance.')
    # First reference-header access in this runner: no reference reaches a predictor.
    references = {case['name']: {side: int(case['game'].headers[side+'EloEstimate']) for side in SIDES} for case in cases}
    rows = [{**row, 'reference': references[row['game']][row['side']]} for row in predicted]
    methods = list(dict.fromkeys(row['method'] for row in rows))
    scales = list(dict.fromkeys(row['source_scale'] for row in rows))
    summaries = {scale: {method: metrics([row for row in rows if row['method'] == method and row['source_scale'] == scale])
                         for method in methods} for scale in scales}
    for path, digest in protected.items():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Protected input changed: {path}')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'games': len(cases),
              'method_order': methods, 'methods': method_metadata(methods, declarations),
              'source_scales': scales, 'canonical_scale': CANONICAL, 'conversion_model': elo_convert.MODEL_VERSION,
              'conversion_ranges': elo_convert.valid_ranges(), 'extrapolate': True,
              'summary_by_scale': summaries, 'summary': summaries[scales[0]] if len(scales) == 1 else {},
              'players': rows, 'contexts': contexts, 'scale_invariance': invariance, 'design': design,
              'runtime_seconds': perf_counter()-started, 'sensitivity_evaluated': False,
              'evidence_audits': {case['name']: case['audit'] for case in cases},
              'input_sha256': {str(path): digest for path, digest in protected.items()},
              'protected_files_unchanged': len(protected)}
    dump(output/'comparison.json', result)
    with (output/'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if figures:
        from tests.analysis.scaled_rating_figures import export
        export(result, output)
    print(json.dumps({'games': len(cases), 'methods': len(methods), 'scales': scales,
                      'scale_invariance': invariance, 'runtime_seconds': result['runtime_seconds']}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modules', nargs='+', default=list(DEFAULT_MODULES))
    parser.add_argument('--output', type=Path, default=ROOT/'tests/analysis/output/rating-scales')
    parser.add_argument('--no-figures', action='store_true')
    options = parser.parse_args()
    run(options.modules, options.output, figures=not options.no_figures)
