"""One-time, offline conversion of normalized cache documents to readable evidence.

Stage and validate first; --apply swaps only the three cache directories and
keeps their originals in the migration output directory. No engine calls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import chess

from analysis.cache.storage import identity, write_json
from analysis.cache.policy import dominates, request_satisfies
from analysis.cache.positions import PositionCache
from analysis.cache.structure import encode_records, measurement_id
from analysis.cache.validation import validate_measurement
from analysis.cache.artifacts import AnalysisStore
from engine.assets_identity import asset_identity


CATEGORIES = ('positions', 'games', 'game-metadata')
REFERENCE_FIELDS = ('position', 'context', 'namespace', 'request', 'measurement')


def upgrade_identity(request):
    """Replace a path/stamp only when it still identifies the existing asset."""
    request = copy.deepcopy(request)
    engine = request.get('engine')
    if isinstance(engine, dict):
        for name in ('maia', 'stockfish'):
            value = engine.get(name)
            if isinstance(value, list) and len(value) == 2:
                path, stamp = value
                if isinstance(path, str) and isinstance(stamp, int):
                    asset = Path(path)
                    if asset.is_file() and asset.stat().st_mtime_ns == stamp:
                        engine[name] = asset_identity(asset)
    # Only the effective time was actually executed by the bounded search.
    if request.get('kind') == 'evaluation' and request.get('strategy') == 'bounded':
        request['max_budget_seconds'] = request['budget_seconds']
    return request


def convert_document(document):
    """Convert validated old observations and return an old-to-new pin map."""
    histories = {key: {field: value[field] for field in ('start_fen', 'moves')}
                 for key, value in document['contexts'].items()}
    heads = {measurement for context in document['contexts'].values()
             for group in context['evidence'].values() for measurement in group.values()}
    boards = {}
    for context, history in histories.items():
        if identity(history) != context:
            raise ValueError('Invalid source history identity.')
        board = chess.Board(history['start_fen'])
        for move in history['moves']:
            board.push_uci(move)
        if board.fen() != document['fen']:
            raise ValueError('Source history does not reach the position.')
        boards[context] = board
    records, pins = [], {}
    position = hashlib.sha256(document['fen'].encode()).hexdigest()
    for old_id, row in document['measurements'].items():
        if identity(row) != old_id:
            raise ValueError('Invalid source measurement identity.')
        request = document['requests'][row['request']]
        if request.get('engine') is not None:
            engine_key = request['engine']
            engine = document['engines'][engine_key]
            if identity(engine) != engine_key:
                raise ValueError('Invalid source engine identity.')
            request = {**request, 'engine': engine}
        if identity(request) != row['request']:
            raise ValueError('Invalid source request identity.')
        result = document['results'][row['result']]
        if identity(result) != row['result']:
            raise ValueError('Invalid source result identity.')
        namespace, context = row['namespace'], row['context']
        validate_measurement(boards[context], namespace, request, result)
        updated = upgrade_identity(request)
        new_id = measurement_id(context, namespace, updated, result)
        old_pin = (position, context, namespace, row['request'], old_id)
        pins[old_pin] = {'position': position, 'context': context,
            'namespace': namespace, 'request': identity(updated), 'measurement': new_id}
        engine = updated.get('engine') or {}
        identified = all(not isinstance(engine.get(name), list) for name in ('maia', 'stockfish'))
        active = old_id in heads and identified and request_satisfies(updated, result, updated)
        records.append({'context': context, 'namespace': namespace, 'request': updated,
                        'result': result, 'active': active})
    # Retain old pinned observations while selecting only non-dominated heads.
    for index, record in enumerate(records):
        if not record['active']:
            continue
        for other in records[index + 1:]:
            if not other['active'] or other['context'] != record['context'] or other['namespace'] != record['namespace']:
                continue
            if dominates(other['request'], other['result'], record['request'], record['result']):
                record['active'] = False
                break
            if dominates(record['request'], record['result'], other['request'], other['result']):
                other['active'] = False
    converted = encode_records(document['fen'], histories, records)
    PositionCache._decode(converted, strict=True)
    return converted, pins


def rewrite_pins(value, pins):
    if isinstance(value, dict):
        if all(field in value for field in ('position', 'context', 'namespace', 'request', 'measurement')):
            return pins[tuple(value[field] for field in REFERENCE_FIELDS)]
        return {key: rewrite_pins(item, pins) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_pins(item, pins) for item in value]
    return value


def snapshot(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for category in CATEGORIES for path in sorted((directory / category).glob('*.json'))}


def migrate(source, output, artifacts=(), *, apply=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Migration output must be outside the source cache.')
    stage, original = output / 'staged', output / 'original'
    if stage.exists() or original.exists():
        raise ValueError('Use a fresh migration output directory.')
    before = snapshot(source)
    for category in CATEGORIES:
        (stage / category).mkdir(parents=True)
    pins = {}
    for path in sorted((source / 'positions').glob('*.json')):
        document, mapped = convert_document(json.loads(path.read_text(encoding='utf-8')))
        expected = hashlib.sha256(document['fen'].encode()).hexdigest()
        if path.stem != expected:
            raise ValueError(f'Position filename mismatch: {path}')
        write_json(stage / 'positions' / path.name, document)
        pins.update(mapped)
    for path in sorted((source / 'games').glob('*.json')):
        write_json(stage / 'games' / path.name, rewrite_pins(json.loads(path.read_text(encoding='utf-8')), pins))
    migrated_artifacts = []
    for path in sorted(set(Path(path).resolve() for path in artifacts)):
        document = json.loads(path.read_text(encoding='utf-8'))
        # The retired JsonCache used the public document's digest as its key.
        metadata_path = source / 'game-metadata' / f'{identity(identity(document))}.json'
        if not metadata_path.is_file():
            continue
        metadata = rewrite_pins(json.loads(metadata_path.read_text(encoding='utf-8')), pins)
        target = identity(AnalysisStore.artifact_key(path, document))
        write_json(stage / 'game-metadata' / f'{target}.json', metadata)
        migrated_artifacts.append(str(path))
    from tests.analysis.audit_position_cache import audit
    verification = audit(stage)
    if verification['issues']:
        write_json(output / 'failed-audit.json', verification)
        raise ValueError('Staged cache failed validation; source was not changed.')
    if snapshot(source) != before:
        raise RuntimeError('Source cache changed during conversion; source was not changed.')
    report = {'source': str(source), 'applied': False, 'original_files': len(before),
              'converted_pins': len(pins), 'artifacts': migrated_artifacts,
              'audit': verification, 'backup': str(original)}
    write_json(output / 'source-hashes.json', before)
    write_json(output / 'report.json', report)
    if apply:
        original.mkdir()
        backed_up, installed = [], []
        try:
            for category in CATEGORIES:
                target = (source / category).resolve()
                if target.parent != source or target.name not in CATEGORIES:
                    raise ValueError('Unsafe cache migration target.')
                if target.exists():
                    target.rename(original / category)
                    backed_up.append(category)
                (stage / category).rename(target)
                installed.append(category)
        except BaseException:
            for category in reversed(installed):
                (source / category).rename(stage / category)
            for category in reversed(backed_up):
                (original / category).rename(source / category)
            raise
        report['applied'] = True
        write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, nargs='*', default=[])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    artifacts = [file for root in args.artifacts for file in root.rglob('analysis.json')]
    report = migrate(args.cache, args.output, artifacts, apply=args.apply)
    print(json.dumps({key: value for key, value in report.items() if key != 'audit'}, indent=2))
    print(json.dumps(report['audit']['totals'], indent=2))


if __name__ == '__main__':
    main()
