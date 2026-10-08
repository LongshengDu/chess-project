"""Extensible, history-aware evidence in one atomic JSON file per canonical FEN."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

import chess

from analysis.cache.storage import identity, write_json
from analysis.cache.validation import validate_measurement
from analysis.cache.policy import request_satisfies, history_compatible, quality_key, dominates
from analysis.cache.structure import (FORMAT, empty_document, evidence_group, entries,
    request_value, result_value, measurement_id, add_observation, share_repeated_results)


_STRIPES = tuple(threading.Lock() for _ in range(64))
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _context(board):
    return {"start_fen": board.root().fen(), "moves": [move.uci() for move in board.move_stack]}


def _position(fen):
    return hashlib.sha256(fen.encode("utf-8")).hexdigest()


def _namespace(value):
    if not isinstance(value, str) or not value:
        raise ValueError("An evidence namespace must be a nonempty string.")
    return value


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


@contextmanager
def _locked(directory, position):
    """Serialize merge writes in threads and processes; OS releases crashed writers."""
    stripe = int(position[:8], 16) % len(_STRIPES)
    with _STRIPES[stripe]:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{stripe:02d}.lock").open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                while True:
                    try:
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PositionCache:
    """Readable position evidence with exact immutable pins and policy-based reuse.

    Engine work stays outside merge locks. Refresh views only expose measurements
    written or imported by that instance, including a newly measured result that
    is weaker than the shared cache's preferred observation.
    """

    def __init__(self, directory, *, reuse_existing=True):
        self.directory = Path(directory)
        self.positions_directory = self.directory / "positions"
        self._written = None if reuse_existing else {}
        self._reusable_written = set()
        self._write_sequence = 0
        self._written_lock = threading.Lock()

    def reference(self, board, namespace, request):
        return {"position": _position(board.fen()), "context": identity(_context(board)),
                "namespace": _namespace(namespace), "request": identity(request)}

    @staticmethod
    def _canonical_result(namespace, request, result):
        if namespace == "stockfish" and isinstance(request, dict) and request.get("kind") == "evaluation":
            return {key: value for key, value in result.items() if key != "is_checkmate"}
        return result

    def measurement_reference(self, board, namespace, request, result):
        reference = self.reference(board, namespace, request)
        result = self._canonical_result(namespace, request, result)
        return {**reference, "measurement": measurement_id(reference['context'], namespace, request, result)}

    @staticmethod
    def _valid_reference(reference):
        if not isinstance(reference, dict):
            return False
        if any(not isinstance(reference.get(key), str) or not _HASH.fullmatch(reference[key])
               for key in ("position", "context", "request")):
            return False
        if not isinstance(reference.get('namespace'), str) or not reference['namespace']:
            return False
        return 'measurement' not in reference or (isinstance(reference['measurement'], str)
                                                  and bool(_HASH.fullmatch(reference['measurement'])))

    @staticmethod
    def _board(document, context_id, boards):
        key = (document['fen'], context_id)
        if key not in boards:
            context = document['histories'][context_id]
            history = {name: context[name] for name in ('start_fen', 'moves')}
            if not isinstance(history['moves'], list) or identity(history) != context_id:
                raise ValueError('Invalid history identity.')
            board = chess.Board(history['start_fen'])
            for move in history['moves']:
                board.push_uci(move)
            if board.fen() != document['fen']:
                raise ValueError('Position history does not reach its FEN.')
            boards[key] = board
        return boards[key]

    @classmethod
    def _decode(cls, document, namespace=None, *, strict=False, boards=None):
        if document is None:
            return []
        boards = {} if boards is None else boards
        records = []
        for name, context, observation in entries(document, namespace, strict=strict):
            try:
                request = request_value(document, observation)
                result = result_value(document, observation)
                measurement = measurement_id(context, name, request, result)
                if observation.get('measurement') != measurement or type(observation.get('active')) is not bool:
                    raise ValueError('Invalid observation identity or active state.')
                board = cls._board(document, context, boards)
                validate_measurement(board, name, request, result)
                if name == 'stockfish' and isinstance(request, dict) and request.get('kind') == 'evaluation':
                    result = {**result, 'is_checkmate': board.is_checkmate()}
                records.append({'reference': {'position': _position(document['fen']), 'context': context,
                    'namespace': name, 'request': identity(request), 'measurement': measurement},
                    'request': request, 'result': result, '_board': board, '_entry': observation})
            except (KeyError, TypeError, ValueError):
                if strict:
                    raise
        return records

    @staticmethod
    def _engine_request(namespace, request):
        return isinstance(request, dict) and ((namespace == 'maia' and 'own_rating' in request)
            or (namespace == 'stockfish' and 'kind' in request))

    @classmethod
    def _usable(cls, namespace, request, result, requested):
        if cls._engine_request(namespace, request) or cls._engine_request(namespace, requested):
            return request_satisfies(request, result, requested)
        # Unknown categories have no quality/history model: exact identity only.
        return identity(request) == identity(requested)

    def _visible(self, record, *, pinned=False):
        if self._written is not None:
            with self._written_lock:
                key = identity(record['reference'])
                return key in (self._written if pinned else self._reusable_written)
        return pinned or record['_entry']['active']

    @staticmethod
    def _public(record):
        return None if record is None else copy.deepcopy({key: record[key]
            for key in ('reference', 'request', 'result')})

    def _best(self, records):
        # Equal requested quality uses publication order, including refresh-local
        # results that are inactive because another session has a stronger one.
        with self._written_lock:
            written = dict(self._written or {})
        return max(enumerate(records), key=lambda item: (quality_key(item[1]['request'],
            item[1]['result'])[:-1], written.get(identity(item[1]['reference']), item[0])))[1] if records else None

    def _select(self, board, namespace, request, records):
        context, request_id = identity(_context(board)), identity(request)
        exact, compatible = [], []
        for record in records:
            if not self._visible(record) or not self._usable(namespace, record['request'], record['result'], request):
                continue
            reference = record['reference']
            if reference['context'] == context and reference['request'] == request_id:
                exact.append(record)
            elif (self._engine_request(namespace, request)
                    and history_compatible(board, record['_board'], namespace, record['request'])):
                compatible.append(record)
        return self._best(exact) or self._best(compatible)

    def select_record(self, board, namespace, request):
        return self.select_many(board, namespace, [request])[0]

    def select_many(self, board, namespace, requests):
        """Return original observations for a request batch after one position read."""
        namespace = _namespace(namespace)
        requests = list(requests)
        if not requests:
            return []
        records = self._decode(self._read(_position(board.fen())), namespace)
        return [self._public(self._select(board, namespace, request, records)) for request in requests]

    def get(self, board, namespace, request):
        record = self.select_record(board, namespace, request)
        return None if record is None else record['result']

    def get_many(self, board, namespace, requests):
        return [None if record is None else record['result']
                for record in self.select_many(board, namespace, requests)]

    def resolve_reference(self, board, namespace, request):
        record = self.select_record(board, namespace, request)
        return None if record is None else record['reference']

    def _record(self, records, reference):
        matches = []
        pinned = 'measurement' in reference
        for record in records:
            if any(record['reference'][key] != reference[key] for key in reference
                   if key in ('position', 'context', 'namespace', 'request', 'measurement')):
                continue
            if not self._visible(record, pinned=pinned):
                continue
            if pinned:
                return record
            if self._usable(reference['namespace'], record['request'], record['result'], record['request']):
                matches.append(record)
        return self._best(matches)

    def get_reference(self, reference):
        return self.get_references([reference])[0]

    def get_references(self, references):
        return [None if record is None else record['result']
                for record in self.get_reference_records(references)]

    def get_reference_record(self, reference):
        return self.get_reference_records([reference])[0]

    def get_reference_records(self, references):
        """Resolve immutable pins exactly; do not substitute a preferred observation."""
        decoded, boards, results = {}, {}, []
        for reference in references:
            if not self._valid_reference(reference):
                results.append(None)
                continue
            position = reference['position']
            if position not in decoded:
                decoded[position] = self._decode(self._read(position), boards=boards)
            record = self._record(decoded[position], reference)
            results.append(self._public(record))
        return results

    def reference_matches(self, board, reference):
        """Validate a pin against an exact or engine-compatible game history."""
        return self.references_match(board, [reference])

    def references_match(self, board, references):
        """Check all game-position pins with one document read and history replay."""
        references = list(references)
        position = _position(board.fen())
        if any(not self._valid_reference(reference) or reference['position'] != position
               for reference in references):
            return False
        if not references:
            return True
        records = self._decode(self._read(position))
        for reference in references:
            record = self._record(records, reference)
            if record is None or not history_compatible(board, record['_board'],
                    reference['namespace'], record['request']):
                return False
        return True

    def records(self, board, namespace):
        """List active reusable observations, including compatible stored histories."""
        namespace = _namespace(namespace)
        records = self._decode(self._read(_position(board.fen())), namespace)
        return [self._public(record) for record in records if self._visible(record)
                and self._usable(namespace, record['request'], record['result'], record['request'])
                and history_compatible(board, record['_board'], namespace, record['request'])]

    @staticmethod
    def _prepare_group(document, context, namespace, request, preserve):
        container = document['evidence']
        for key in (namespace, *evidence_group(namespace, request)):
            if not isinstance(container.get(key), dict):
                if key in container:
                    preserve()
                container[key] = {}
            container = container[key]
        if not isinstance(container.get(context), list):
            if context in container:
                preserve()
            container[context] = []

    def _store(self, document, context, namespace, request, result, *, active=True, preserve=lambda: None):
        """Prefer dominating observations while keeping every immutable body for pins."""
        self._prepare_group(document, context, namespace, request, preserve)
        prior_records = self._decode(document, namespace)
        valid_entries = {id(record['_entry']) for record in prior_records}
        measurement = measurement_id(context, namespace, request, result)
        for _, previous_context, observation in entries(document, namespace):
            if (previous_context == context and observation.get('measurement') == measurement
                    and id(observation) not in valid_entries):
                preserve()
        incoming_active = active
        active = active and self._usable(namespace, request, result, request)
        if active:
            for old in prior_records:
                if old['reference']['context'] != context or not old['_entry']['active']:
                    continue
                if not self._usable(namespace, old['request'], old['result'], old['request']):
                    old['_entry']['active'] = False
                elif self._engine_request(namespace, request):
                    if dominates(request, result, old['request'], old['result']):
                        old['_entry']['active'] = False
                    elif dominates(old['request'], old['result'], request, result):
                        active = False
                elif old['reference']['request'] == identity(request):
                    old['_entry']['active'] = False
        elif not incoming_active:
            # Importing an archived copy must not retire a still-active identical
            # observation; subsequent dominating imports can supersede it.
            active = any(old['reference']['measurement'] == measurement and old['_entry']['active']
                         and self._usable(namespace, request, result, request) for old in prior_records)
        return add_observation(document, context, namespace, request, result, active=active)

    def put(self, board, namespace, request, result):
        return self.put_many(board, namespace, [(request, result)])[0]

    def put_many(self, board, namespace, values):
        namespace = _namespace(namespace)
        values = json.loads(json.dumps(list(values), allow_nan=False))
        for request, result in values:
            validate_measurement(board, namespace, request, result)
        references = [self.measurement_reference(board, namespace, request, result) for request, result in values]
        if not references:
            return []
        position, context = references[0]['position'], references[0]['context']
        path = self.positions_directory / f'{position}.json'
        history = _context(board)
        with _locked(self.directory / '.position-locks', position):
            document = self._read(position)
            preserved = False

            def preserve():
                nonlocal preserved
                if path.exists() and not preserved:
                    self._preserve_corrupt(path)
                    preserved = True

            if document is None:
                preserve()
                document = empty_document(board.fen())
            previous = document['histories'].get(context)
            if previous is not None and (not isinstance(previous, dict)
                    or any(previous.get(key) != value for key, value in history.items())):
                preserve()
                previous = None
            document['histories'][context] = {**(previous or {}), **history}
            for request, result in values:
                self._store(document, context, namespace, request,
                            self._canonical_result(namespace, request, result), preserve=preserve)
            share_repeated_results(document)
            write_json(path, document)
            self._remember_writes(references)
        return references

    def _remember_writes(self, references, *, reusable=True):
        if self._written is not None:
            with self._written_lock:
                for reference in references:
                    key = identity(reference)
                    if reusable or key not in self._written:
                        self._write_sequence += 1
                        self._written[key] = self._write_sequence
                    if reusable:
                        self._reusable_written.add(key)

    @staticmethod
    def _metadata_merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                PositionCache._metadata_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    def import_directory(self, source_root):
        """Merge current-format observations without aliasing requests or histories."""
        source = PositionCache(source_root)
        if source.directory.resolve() == self.directory.resolve():
            return 0
        imported = 0
        reserved = {'format', 'fen', 'histories', 'engines', 'evidence', 'shared_results'}
        for path in sorted(source.positions_directory.glob('*.json')):
            if not _HASH.fullmatch(path.stem):
                continue
            incoming = source._read(path.stem)
            if not self._valid_import(incoming):
                continue
            records = self._decode(incoming, strict=True)
            with _locked(self.directory / '.position-locks', path.stem):
                destination = self.positions_directory / path.name
                document = self._read(path.stem)
                preserved = False

                def preserve():
                    nonlocal preserved
                    if destination.exists() and not preserved:
                        self._preserve_corrupt(destination)
                        preserved = True

                if document is None:
                    preserve()
                    document = empty_document(incoming['fen'])
                self._metadata_merge(document, {key: value for key, value in incoming.items() if key not in reserved})
                for context, history in incoming['histories'].items():
                    previous = document['histories'].get(context)
                    if previous is not None and (not isinstance(previous, dict) or
                            any(previous.get(key) != history[key] for key in ('start_fen', 'moves'))):
                        preserve()
                        previous = None
                    document['histories'][context] = previous or {}
                    self._metadata_merge(document['histories'][context], history)
                for record in records:
                    reference = record['reference']
                    self._store(document, reference['context'], reference['namespace'], record['request'],
                        self._canonical_result(reference['namespace'], record['request'], record['result']),
                        active=record['_entry']['active'], preserve=preserve)
                share_repeated_results(document)
                write_json(destination, document)
                # Imported archive bodies remain pin-readable without becoming
                # fresh candidates. Own measurements stay reusable even when a
                # stronger observation keeps them inactive in the shared file.
                self._remember_writes((record['reference'] for record in records
                    if not record['_entry']['active']), reusable=False)
                self._remember_writes(record['reference'] for record in records
                    if record['_entry']['active'])
            imported += 1
        return imported

    @classmethod
    def _valid_import(cls, document):
        if document is None:
            return False
        try:
            boards = {}
            for context in document['histories']:
                cls._board(document, context, boards)
            for key, value in document['engines'].items():
                if identity(value) != key:
                    return False
            for key, value in document['shared_results'].items():
                if identity(value) != key:
                    return False
            records = cls._decode(document, strict=True, boards=boards)
            return len({record['reference']['measurement'] for record in records}) == len(records)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    def _preserve_corrupt(self, path):
        archive = self.directory / '.position-corrupt'
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f'{path.stem}.{uuid.uuid4().hex}.json').write_bytes(path.read_bytes())

    def _read(self, position):
        try:
            value = json.loads((self.positions_directory / f'{position}.json').read_text(encoding='utf-8'),
                               parse_constant=_nonfinite)
            if (isinstance(value, dict) and value.get('format') == FORMAT
                    and isinstance(value.get('fen'), str) and _position(value['fen']) == position
                    and all(isinstance(value.get(key), dict) for key in
                            ('histories', 'engines', 'evidence', 'shared_results'))):
                return value
        except (OSError, ValueError):
            pass
        return None
