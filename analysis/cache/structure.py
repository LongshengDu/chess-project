"""Readable position-document layout, independent of disk IO and reuse policy."""
from __future__ import annotations

from collections import Counter
import copy

from analysis.cache.storage import identity


FORMAT = 'position-evidence-v2'


def empty_document(fen):
    return {'format': FORMAT, 'fen': fen, 'histories': {}, 'engines': {},
            'evidence': {}, 'shared_results': {}}


def evidence_group(namespace, request):
    """Use descriptive groups; exact search parameters remain beside the result."""
    if isinstance(request, dict):
        if namespace == 'maia' and 'own_rating' in request:
            return 'rating-pair', f"{request['own_rating']}-{request.get('opponent_rating')}"
        if namespace == 'stockfish' and request.get('kind') == 'evaluation':
            return 'evaluation', str(request.get('strategy', 'unspecified'))
        if namespace == 'stockfish' and request.get('kind') == 'exploration':
            return 'exploration', 'restricted' if request.get('root_moves') else 'unrestricted'
    return 'measurements', 'default'


def entries(document, namespace=None, *, strict=False):
    """Yield namespace, history id and inline observation without following hashes."""
    def mapping(value):
        if isinstance(value, dict):
            return value
        if strict:
            raise ValueError('Invalid evidence group.')
        return {}

    for name, kinds in document.get('evidence', {}).items():
        if namespace is not None and name != namespace:
            continue
        if not isinstance(name, str) or not name:
            if strict:
                raise ValueError('Invalid evidence namespace.')
            continue
        for profiles in mapping(kinds).values():
            for histories in mapping(profiles).values():
                for history, observations in mapping(histories).items():
                    if not isinstance(observations, list):
                        if strict:
                            raise ValueError('Invalid history observations.')
                        continue
                    for observation in observations:
                        if isinstance(observation, dict):
                            yield name, history, observation
                        elif strict:
                            raise ValueError('Invalid observation.')


def request_value(document, observation):
    request = observation['request']
    if isinstance(request, dict) and request.get('engine') is not None:
        engine_id = request['engine']
        engine = document['engines'][engine_id]
        if identity(engine) != engine_id:
            raise ValueError('Invalid engine identity.')
        request = {**request, 'engine': engine}
    return request


def result_value(document, observation):
    if ('result' in observation) == ('result_ref' in observation):
        raise ValueError('An observation requires one result body or shared reference.')
    if 'result' in observation:
        return observation['result']
    result_id = observation['result_ref']
    result = document['shared_results'][result_id]
    if identity(result) != result_id:
        raise ValueError('Invalid shared result identity.')
    return result


def measurement_id(context, namespace, request, result):
    """Retain the external immutable address used by existing saved games."""
    return identity({'context': context, 'namespace': namespace,
                     'request': identity(request), 'result': identity(result)})


def add_observation(document, context, namespace, request, result, *, active=True):
    """Append an observation, retaining its original request and immutable identity."""
    kind, profile = evidence_group(namespace, request)
    group = document['evidence'].setdefault(namespace, {}).setdefault(kind, {}).setdefault(profile, {})
    observations = group.setdefault(context, [])
    measurement = measurement_id(context, namespace, request, result)
    stored_request = request
    if isinstance(request, dict) and request.get('engine') is not None:
        engine_id = identity(request['engine'])
        document['engines'][engine_id] = request['engine']
        stored_request = {**request, 'engine': engine_id}
    observation = {'measurement': measurement, 'request': stored_request,
                   'active': active, 'result': result}
    for existing in observations:
        if isinstance(existing, dict) and existing.get('measurement') == measurement:
            existing.pop('result_ref', None)
            existing.update(observation)
            return existing
    observations.append(observation)
    return observation


def share_repeated_results(document):
    """Keep unique bodies inline; pool a body only when multiple observations use it."""
    found, invalid_refs = [], set()
    for _, _, observation in entries(document):
        try:
            result = result_value(document, observation)
            found.append((observation, identity(result), result))
        except (KeyError, TypeError, ValueError):
            if isinstance(observation.get('result_ref'), str):
                invalid_refs.add(observation['result_ref'])
    counts = Counter(result_id for _, result_id, _ in found)
    shared = {key: value for key, value in document['shared_results'].items() if key in invalid_refs}
    for observation, result_id, result in found:
        if counts[result_id] > 1:
            shared[result_id] = result
            observation.pop('result', None)
            observation['result_ref'] = result_id
        else:
            observation.pop('result_ref', None)
            observation['result'] = result
    document['shared_results'] = shared


def encode_records(fen, histories, records):
    """Build one current document without IO (also used by explicit migration).

    ``histories`` maps existing history identities to start_fen/moves objects.
    Records contain context, namespace, request, result and optional active.
    Results must already use their canonical cache representation.
    """
    document = empty_document(fen)
    document['histories'] = copy.deepcopy(histories)
    active_observations = {}
    for record in records:
        if record['context'] not in document['histories']:
            raise ValueError('Observation references an absent history.')
        measurement = measurement_id(record['context'], record['namespace'], record['request'], record['result'])
        active = active_observations.get(measurement, False) or record.get('active', True)
        active_observations[measurement] = active
        add_observation(document, record['context'], record['namespace'],
                        copy.deepcopy(record['request']), copy.deepcopy(record['result']),
                        active=active)
    share_repeated_results(document)
    return document
