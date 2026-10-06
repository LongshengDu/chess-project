"""Pluggable played-strength estimators over one common chess-evidence contract.

Applications prepare evidence and call this module; they do not select algorithm
branches. METHOD loads analysis/player_rating/<METHOD>.py, whose Rating class implements
PlayerRating.fit. Adding a method requires only that file and a YAML change.
"""
from __future__ import annotations

import hashlib
from importlib import import_module
import inspect
import json
import math
from pathlib import Path
import re

from analysis.player_rating.interface import PlayerRating
from analysis.cache import write_json
from analysis.player_rating.evidence import SCHEMA_VERSION, collect_evidence, evidence_matches_game, validate_evidence
from analysis.player_rating.context import account_ratings, attach_context, game_scores, saved_context, saved_ratings
from analysis.settings import CONFIG
from analysis.player_rating import scale
from analysis import elo_convert


def get_estimator(method=None, *, args=None) -> PlayerRating:
    """Load the configured module's concrete Rating class, without a registry.

    Python caches module imports; each call returns a fresh, stateless instance.
    Keep constructors inexpensive and implement calculation in ``fit``.
    """
    method = CONFIG['ANALYSIS']['PLAYER_RATING']['METHOD'] if method is None else method
    if not isinstance(method, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', method):
        raise ValueError('ANALYSIS.PLAYER_RATING.METHOD must be a lowercase module filename without a path or .py suffix.')
    module_name = f'{__package__}.{method}'
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(f'Unknown rating method {method!r}: create analysis/player_rating/{method}.py with a Rating(PlayerRating) class.') from exc
        raise ValueError(f'Cannot load rating method {method!r}: missing dependency {exc.name!r}.') from exc
    if hasattr(module, '__path__') or Path(getattr(module, '__file__', '')).name != f'{method}.py':
        raise ValueError(f'Rating method {method!r} must be implemented in analysis/player_rating/{method}.py.')
    implementation = getattr(module, 'Rating', None)
    if (not inspect.isclass(implementation) or not issubclass(implementation, PlayerRating)
            or implementation.__module__ != module_name):
        raise ValueError(f'analysis/player_rating/{method}.py must define its own Rating(PlayerRating) class.')
    if inspect.isabstract(implementation):
        raise ValueError(f'Rating in analysis/player_rating/{method}.py must implement the abstract fit method.')
    try:
        estimator = implementation() if args is None else implementation(args=args)
    except TypeError as exc:
        requirement = 'without arguments' if args is None else 'with the supplied method arguments'
        raise ValueError(f'Rating in analysis/player_rating/{method}.py must be constructible {requirement}.') from exc
    if estimator.id != method:
        raise ValueError('Rating method identity must match its module filename.')
    if not isinstance(estimator.name, str) or not estimator.name.strip():
        raise ValueError('Estimator display name must be nonempty text.')
    if type(estimator.version) is not int or estimator.version < 1:
        raise ValueError('Estimator version must be a positive integer.')
    if not callable(estimator.fit) or inspect.iscoroutinefunction(estimator.fit) or inspect.isasyncgenfunction(estimator.fit):
        raise ValueError('Rating must implement a synchronous fit(evidence) method.')
    try:
        inspect.signature(estimator.fit).bind({})
    except (TypeError, ValueError) as exc:
        raise ValueError('Rating must implement fit(evidence) without other required arguments.') from exc
    if not isinstance(estimator.parameters, dict):
        raise ValueError('Estimator parameters must be a finite JSON mapping.')
    try:
        json.dumps(estimator.parameters, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('Estimator parameters must be a finite JSON mapping.') from exc
    return estimator


def selected_method():
    return get_estimator().id


def rating_signature(*, args=None):
    """Change saved fit identity when algorithm version or method arguments change."""
    return _estimator_signature(get_estimator(args=args))


def _estimator_signature(estimator):
    payload = [SCHEMA_VERSION, estimator.id, estimator.name, estimator.version, estimator.parameters,
               scale.VERSION, elo_convert.MODEL_VERSION]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return {'method': estimator.id, 'method_version': estimator.version,
            'model_signature': digest, 'evidence_schema_version': SCHEMA_VERSION}


def _validate_result(result, evidence):
    if not isinstance(result, dict) or not isinstance(result.get('players'), dict) or set(result['players']) != {'White', 'Black'}:
        raise ValueError('An estimator must return both White and Black player summaries.')
    if not isinstance(result.get('prior'), dict) or not isinstance(result.get('interval_scope'), str) or not result['interval_scope'].strip():
        raise ValueError('An estimator must describe its prior and interval scope.')
    if 'diagnostics' in result and not isinstance(result['diagnostics'], dict):
        raise ValueError('Optional estimator diagnostics must be a mapping.')
    if 'method' in result and not isinstance(result['method'], str):
        raise ValueError('The optional method description must be text.')
    if 'account_ratings_used' in result and type(result['account_ratings_used']) is not bool:
        raise ValueError('The optional account-rating indicator must be a boolean.')
    rating_range = result.get('rating_range')
    if not isinstance(rating_range, (list, tuple)) or len(rating_range) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in rating_range) or rating_range[0] >= rating_range[1]:
        raise ValueError('Estimator rating_range must contain two finite increasing bounds.')
    if 'central_interval' not in result:
        raise ValueError('An estimator must explicitly declare central_interval, including None for point-only estimates.')
    central_interval = result['central_interval']
    if central_interval is not None and (isinstance(central_interval, bool) or not isinstance(central_interval, (int, float))
            or not math.isfinite(central_interval) or not 0 < central_interval < 1):
        raise ValueError('Estimator central_interval must be a probability or None for point-only estimates.')
    for side, player in result['players'].items():
        required = {'estimate', 'uncertainty', 'interval', 'moves_used', 'identifiable', 'at_rating_limit'}
        if not isinstance(player, dict) or not required <= player.keys():
            raise ValueError(f'{side} rating summary is missing standard fields.')
        estimate, uncertainty, interval = player['estimate'], player['uncertainty'], player['interval']
        if estimate is not None and (type(estimate) is not int or not rating_range[0] <= estimate <= rating_range[1]):
            raise ValueError('Rating estimates must be integer Elo within the supported range or None.')
        if central_interval is None:
            if interval is not None or uncertainty is not None:
                raise ValueError('A point-only estimator must leave interval and uncertainty as None.')
        elif not isinstance(interval, (list, tuple)) or len(interval) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in interval
        ) or not rating_range[0] <= interval[0] <= interval[1] <= rating_range[1]:
            raise ValueError('Rating intervals must be ordered and inside the supported range.')
        if estimate is not None and interval is not None and not interval[0] <= estimate <= interval[1]:
            raise ValueError('The rating interval must include the point estimate.')
        if estimate is None:
            if uncertainty is not None:
                raise ValueError('An unestimated player cannot have an uncertainty number.')
        elif central_interval is not None and (isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty) or uncertainty < 0):
            raise ValueError('Estimated player uncertainty must be finite and nonnegative.')
        if type(player['moves_used']) is not int or not 0 <= player['moves_used'] <= len(evidence[side]['observations']):
            raise ValueError('Moves used must count available observations.')
        if type(player['identifiable']) is not bool or player['identifiable'] != (estimate is not None) or type(player['at_rating_limit']) is not bool:
            raise ValueError('Player identifiability and rating-limit indicators must be consistent booleans.')
        if player['moves_used'] == 0 and estimate is not None:
            raise ValueError('A player with no observations cannot have an estimated rating.')
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('Estimator output must contain only finite JSON values.') from exc


def fit_evidence(evidence, *, evidence_key=None, args=None, rating_scale='lb'):
    """Fit native Maia coordinates and return ratings on the declared input scale."""
    estimator = get_estimator(args=args)
    evidence = validate_evidence(evidence)
    context = scale.rating_context(override=rating_scale)
    source_ratings = {side: record.get('actual_rating') for side, record in evidence.items()}
    canonical = validate_evidence(scale.normalize_evidence(evidence, context))
    result = estimator.fit(canonical)
    _validate_result(result, canonical)
    result = scale.display_fit(result, context, source_ratings)
    metadata = {**_estimator_signature(estimator), 'evidence_key': evidence_key, 'status': 'applied'}
    metadata.update(context_signature=scale.context_signature(source_ratings, context),
                    rating_scale=context['scale'], conversion_version=elo_convert.MODEL_VERSION)
    return {**result, 'name': estimator.name, 'method_id': estimator.id,
            'rating_fit': metadata}


def evidence_cache_path(cache_directory, key):
    if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{64}', key):
        raise ValueError('Invalid rating evidence cache key.')
    return Path(cache_directory)/'player-rating'/f'rating-{key}.json'


def fit_game(game, records, evaluate, cache_directory, model_signature, *, evaluate_many=None, args=None,
             output_dir=None, ratings=None, rating_scale=None):
    """Collect/cache common evidence and dispatch without an algorithm branch.

    ``cache_directory`` is the common analysis cache root. Estimator identity,
    version and interval do not enter evidence identities, so another method
    algorithm can reuse the same Maia and Stockfish measurements.
    An explicit ``output_dir`` is the figure folder itself (normally the game's
    ``player-rating`` subdirectory). It exports the result and replaces figures on
    every call, including evidence-cache hits. Pure ``fit_evidence`` stays free
    of output-directory side effects.
    """
    get_estimator(args=args)  # Fail before expensive work for invalid method or arguments.
    context = scale.rating_context(game.headers, rating_scale)
    if [row['move'] for row in records] != [move.uci() for move in game.mainline_moves()]:
        raise ValueError('Rating evidence does not match the game.')
    canonical = [{key: row[key] for key in ('move', 'position_score', 'scores')} for row in records]
    for target, row in zip(canonical, records, strict=True):
        if row.get('policies') is not None:
            target['policies'] = {str(rating): policy for rating, policy in row['policies'].items()}
    payload = [SCHEMA_VERSION, model_signature, game.board().fen(), canonical]
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    cache = evidence_cache_path(cache_directory, key)
    try:
        evidence = json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else None
    except json.JSONDecodeError:
        evidence = None  # A full run can rebuild a partial/interrupted cache write.
    if not evidence_matches_game(evidence, game):
        evidence = collect_evidence(game, records, evaluate, evaluate_many=evaluate_many)
        write_json(cache, evidence)
    contextual = attach_context(evidence, account_ratings(game.headers, ratings), game_scores(game, records))
    fit = fit_evidence(contextual, evidence_key=key, args=args, rating_scale=context)
    if output_dir is not None:
        from analysis.player_rating.figures import export_figures
        export_figures(fit, output_dir)
    return fit


def store_elo_fit(analysis, fit):
    def compact(players):
        return {color.lower(): {k: values[k] for k in ('estimate', 'uncertainty', 'interval', 'moves_used', 'at_rating_limit',
                                                      'average_accuracy', 'identifiable', 'interval_touches_limit',
                                                      'unrounded_estimate', 'canonical_estimate', 'canonical_interval') if k in values}
                for color,values in players.items()}
    analysis.update(played_elo=compact(fit['players']), played_elo_central_interval=fit['central_interval'],
                    played_elo_rating_range=list(fit['rating_range']),
                    played_elo_method=fit['method_id'], played_elo_prior=fit['prior'],
                    rating_fit=fit['rating_fit'], played_elo_interval_scope=fit.get('interval_scope'))
    analysis['played_elo_name'] = fit['name']
    analysis['played_elo_description'] = fit.get('method', '')
    analysis['played_elo_diagnostics'] = fit.get('diagnostics', {})
    if 'rating_scale' in fit:
        analysis['played_elo_scale'] = fit['rating_scale']
        analysis['played_elo_canonical_rating_range'] = fit.get('canonical_rating_range', fit['rating_range'])
    if 'account_ratings_used' in fit:
        analysis['played_elo_account_ratings_used'] = fit['account_ratings_used']
    else:
        analysis.pop('played_elo_account_ratings_used', None)
    for obsolete in ('elo_calibration', 'played_elo_curve', 'played_elo_base', 'played_elo_confidence'):
        analysis.pop(obsolete, None)


def refresh_saved_rating(analysis, cache_directory, *, output_dir=None):
    """Refresh saved ratings and optional figures without engine/LLM calls.

    Figures are regenerated even when the saved fit already has the current
    signature, so an updated renderer never leaves old PNGs in the output.
    ``output_dir`` is the figure folder itself, not its parent game directory.
    """
    signature = rating_signature()
    context = scale.rating_context(analysis.get('headers'), analysis.get('rating_scale_override'))
    signature['context_signature'] = scale.context_signature(saved_ratings(analysis), context)
    previous = analysis.get('rating_fit', analysis.get('elo_calibration', {}))
    if (all(previous.get(k) == v for k,v in signature.items())
            and 'played_elo_central_interval' in analysis and 'played_elo_rating_range' in analysis
            and 'played_elo_confidence' not in analysis):
        if output_dir is not None:
            from analysis.player_rating.figures import export_saved_figures
            export_saved_figures(analysis, output_dir)
        return
    key = previous.get('evidence_key', '')
    if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Saved analysis needs rating evidence. Run --analysis-only first; existing engine caches are reused.')
    cache = evidence_cache_path(cache_directory, key)
    if not cache.is_file():
        # Read compatible shared evidence written before the estimator interface.
        cache = Path(cache_directory)/'elo-calibration'/f'calibration-{key}.json'
    if not cache.is_file():
        raise ValueError('Saved rating evidence is missing. Run --analysis-only first; existing engine caches are reused.')
    try:
        evidence = json.loads(cache.read_text(encoding='utf-8'))
        result = fit_evidence(saved_context(evidence, analysis), evidence_key=key, rating_scale=context)
    except (ValueError, TypeError) as exc:
        raise ValueError('Saved rating evidence is incompatible with the selected method. '
                         'Run --analysis-only first; existing engine caches are reused.') from exc
    store_elo_fit(analysis, result)
    if output_dir is not None:
        from analysis.player_rating.figures import export_saved_figures
        export_saved_figures(analysis, output_dir)
