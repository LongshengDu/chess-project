"""Read-only invariant audit of the current structured position cache."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import chess

from analysis.cache.storage import identity, write_json
from analysis.cache.positions import PositionCache
from analysis.cache.structure import entries, measurement_id, request_value, result_value
from analysis.cache.validation import validate_measurement


REFERENCE_FIELDS = ('position', 'context', 'namespace', 'request', 'measurement')


def audit(directory):
    root, totals = Path(directory), Counter()
    cache = PositionCache(root)
    issues, inactive, valid_references = [], set(), set()

    def issue(path, problem, **detail):
        issues.append({'file': str(path), 'problem': problem, **detail})

    for path in sorted((root / 'positions').glob('*.json')):
        document = cache._read(path.stem)
        totals['position_files'] += 1
        totals['disk_bytes'] += path.stat().st_size
        if document is None:
            issue(path, 'invalid current structured document')
            continue
        totals['histories'] += len(document['histories'])
        totals['engines'] += len(document['engines'])
        totals['shared_results'] += len(document['shared_results'])
        boards, used_engines, used_histories, shared_uses = {}, set(), set(), Counter()
        physical_results, measurements = [], set()
        for context in document['histories']:
            try:
                cache._board(document, context, boards)
            except (KeyError, TypeError, ValueError) as error:
                issue(path, str(error), context=context)
        for engine_id, engine in document['engines'].items():
            if identity(engine) != engine_id:
                issue(path, 'invalid engine identity', engine=engine_id)
        for result_id, value in document['shared_results'].items():
            physical_results.append(identity(value))
            if identity(value) != result_id:
                issue(path, 'invalid shared result identity', result=result_id)
        try:
            observations = list(entries(document, strict=True))
        except (KeyError, TypeError, ValueError) as error:
            issue(path, str(error))
            observations = list(entries(document))
        for namespace, context, observation in observations:
            measurement = observation.get('measurement')
            totals['measurements'] += 1
            try:
                request = request_value(document, observation)
                result = result_value(document, observation)
                if measurement != measurement_id(context, namespace, request, result):
                    raise ValueError('invalid measurement identity')
                if measurement in measurements:
                    raise ValueError('duplicate observation')
                measurements.add(measurement)
                if type(observation.get('active')) is not bool:
                    raise ValueError('invalid active state')
                board = cache._board(document, context, boards)
                validate_measurement(board, namespace, request, result)
                if namespace == 'stockfish' and isinstance(request, dict) and request.get('kind') == 'evaluation' and 'is_checkmate' in result:
                    raise ValueError('derived board fact duplicated in result body')
                used_histories.add(context)
                if isinstance(observation['request'], dict) and observation['request'].get('engine') is not None:
                    used_engines.add(observation['request']['engine'])
                if 'result_ref' in observation:
                    shared_uses[observation['result_ref']] += 1
                else:
                    totals['inline_results'] += 1
                    physical_results.append(identity(result))
                totals[f'{namespace}_measurements'] += 1
                if not isinstance(request, dict) or not request.get('engine'):
                    totals[f'{namespace}_unknown_engine'] += 1
                if observation['active']:
                    totals['active_measurements'] += 1
                    if not cache._usable(namespace, request, result, request):
                        raise ValueError('active engine observation has an obsolete or unidentified request')
                else:
                    totals['inactive_measurements'] += 1
                    inactive.add((path.stem, measurement))
                valid_references.add((path.stem, context, namespace, identity(request), measurement))
            except (KeyError, TypeError, ValueError) as error:
                issue(path, str(error), measurement=measurement)
        duplicate_bodies = len(physical_results) - len(set(physical_results))
        totals['duplicate_result_bodies'] += duplicate_bodies
        if duplicate_bodies:
            issue(path, 'duplicate physical result bodies', count=duplicate_bodies)
        for result_id in document['shared_results']:
            if shared_uses[result_id] < 2:
                issue(path, 'shared result must be referenced by multiple observations', result=result_id,
                      references=shared_uses[result_id])
        totals['orphan_engines'] += len(set(document['engines']) - used_engines)
        totals['orphan_histories'] += len(set(document['histories']) - used_histories)
        totals['orphan_shared_results'] += len(set(document['shared_results']) - shared_uses.keys())

    manifest_pins = set()
    for category in ('games', 'game-metadata'):
        for path in sorted((root / category).glob('*.json')):
            try:
                document = json.loads(path.read_text(encoding='utf-8'))
                manifest = document if category == 'games' else document.get('evidence', {})
                positions = manifest.get('positions')
                if positions is None:
                    continue
                if not isinstance(positions, list):
                    raise ValueError('invalid manifest positions')
                totals['manifests'] += 1
                totals['manifest_positions'] += len(positions)
                for index, entry in enumerate(positions):
                    fields = entry['fields']
                    if fields['ply'] != index:
                        raise ValueError('manifest ply does not match position order')
                    position = hashlib.sha256(chess.Board(fields['fen']).fen().encode()).hexdigest()
                    references = [('maia', reference) for reference in entry['maia'].values()]
                    if entry['stockfish'] is None:
                        totals['unmeasured_final_stockfish_positions'] += 1
                        if index != len(positions) - 1:
                            issue(path, 'missing played-position Stockfish', index=index)
                    else:
                        references.append(('stockfish', entry['stockfish']))
                    totals['manifest_references'] += len(references)
                    for namespace, reference in references:
                        if (not cache._valid_reference(reference) or 'measurement' not in reference
                                or reference['namespace'] != namespace or reference['position'] != position):
                            issue(path, 'invalid manifest reference', index=index, reference=reference)
                            continue
                        manifest_pins.add((reference['position'], reference['measurement']))
                        if tuple(reference[field] for field in REFERENCE_FIELDS) not in valid_references:
                            issue(path, 'unresolved manifest reference', reference=reference)
            except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
                issue(path, str(error))
    totals['inactive_measurements_pinned_by_manifests'] = len(inactive & manifest_pins)
    totals['inactive_measurements_not_pinned_by_manifests'] = len(inactive - manifest_pins)
    totals['issues'] = len(issues)
    return {'cache': str(root.resolve()), 'read_only': True, 'totals': dict(totals), 'issues': issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.cache_directory)
    write_json(args.output, result)
    print(json.dumps({'totals': result['totals'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
