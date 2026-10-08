"""Offline cache normalization; backup and verify before explicitly applying changes.

This one-time tool reads the previous measurement layout. Runtime readers support
only the current structured layout. No engine or inference is started here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

import chess

from analysis.cache.storage import identity, write_json
from analysis.cache.positions import PositionCache
from analysis.cache.structure import (FORMAT, empty_document, entries as observations,
                                     request_value, result_value)
from analysis.cache.requests import stockfish_initial_request
from analysis.cache.validation import validate_measurement
from tests.analysis.cache_exploration_conversion import convert_exploration

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "cache-consolidation"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _key(reference):
    return tuple(reference[key] for key in ("position", "context", "namespace", "request", "measurement"))


def _reference(position, context, namespace, request, measurement):
    return {"position": position, "context": context, "namespace": namespace,
            "request": request, "measurement": measurement}


def _board(document, context):
    history = document["histories" if document.get("format") == FORMAT else "contexts"][context]
    if identity({key: history[key] for key in ("start_fen", "moves")}) != context:
        raise ValueError("Invalid history identity in source cache.")
    board = chess.Board(history["start_fen"])
    for move in history["moves"]:
        board.push_uci(move)
    if board.fen() != document["fen"]:
        raise ValueError("Source history does not reach its cached FEN.")
    return board


def _records(document, position):
    """Yield source observations, full requests and whether each is a current head."""
    if document.get('format') == FORMAT:
        for namespace, context, observation in observations(document, strict=True):
            request = request_value(document, observation)
            reference = _reference(position, context, namespace, identity(request), observation['measurement'])
            yield reference, request, result_value(document, observation), observation['active']
        return
    if "measurements" in document:
        for measurement, row in document["measurements"].items():
            request = document['requests'][row['request']]
            if isinstance(request, dict) and request.get('engine') is not None:
                engine_id = request['engine']
                engine = document['engines'][engine_id]
                if identity(engine) != engine_id:
                    raise ValueError('Invalid source engine identity.')
                request = {**request, 'engine': engine}
            if identity(request) != row['request']:
                raise ValueError('Invalid source request identity.')
            result = document["results"][row["result"]]
            if identity(row) != measurement or identity(result) != row["result"]:
                raise ValueError("Invalid normalized source measurement.")
            reference = _reference(position, row["context"], row["namespace"], row["request"], measurement)
            heads = document["contexts"][row["context"]]["evidence"][row["namespace"]]
            yield reference, request, result, heads.get(row["request"]) == measurement
        return
    for context, history in document["contexts"].items():
        for namespace, store in history["evidence"].items():
            if not isinstance(store, dict) or "measurements" not in store:
                continue
            for measurement, row in store["measurements"].items():
                if identity(row) != measurement or set(row) != {"request", "result"}:
                    raise ValueError("Invalid source measurement identity.")
                request = identity(row["request"])
                reference = _reference(position, context, namespace, request, measurement)
                yield reference, row["request"], row["result"], store["requests"].get(request) == measurement


def _payload(document, reference):
    if document.get('format') == FORMAT:
        observation = next(entry for namespace, context, entry in observations(document, strict=True)
            if namespace == reference['namespace'] and context == reference['context']
            and entry['measurement'] == reference['measurement'])
        return result_value(document, observation)
    if "measurements" in document:
        row = document["measurements"][reference["measurement"]]
        return document["results"][row["result"]]
    store = document["contexts"][reference["context"]]["evidence"][reference["namespace"]]
    return store["measurements"][reference["measurement"]]["result"]


def _convert(board, namespace, request, result):
    if namespace == "stockfish" and request.get("kind") in ("exploration", "continuation"):
        converted = convert_exploration(board, request, result)
        if converted is None:
            return None, "incomplete_or_truncated_exploration"
        board, request, result = converted
    elif namespace == "stockfish" and request.get("kind") == "evaluation" and "options" in request:
        canonical = stockfish_initial_request(request["engine"], request["depth"], request["budget_seconds"],
            request["max_budget_seconds"], request["strategy"], request["options"])
        canonical["policy_version"] = request["policy_version"]
        request = canonical
    try:
        validate_measurement(board, namespace, request, result)
    except (KeyError, TypeError, ValueError) as error:
        return None, str(error)
    return (board, request, result), None


def consolidate(cache_directory, *, output_directory=DEFAULT_OUTPUT, apply=False, public_directory=None):
    root, output = Path(cache_directory).resolve(), Path(output_directory).resolve()
    if output == root or root in output.parents:
        raise ValueError("Preservation output must be outside the cache directory.")
    run = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    stage, backup = run / "staged", run / "original-cache.zip"
    run.mkdir(parents=True)
    def public_hashes():
        return {} if public_directory is None else {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(public_directory).rglob("analysis.json")}
    public_before = public_hashes()
    files = sorted(path for category in ("positions", "games", "game-metadata")
                   for path in (root / category).glob("*.json"))
    fingerprints = {}
    with ZipFile(backup, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for path in files:
            data = path.read_bytes()
            name = path.relative_to(root).as_posix()
            fingerprints[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    cache = PositionCache(stage)
    # Current files are imported unchanged in meaning; legacy parsing is confined
    # to this explicit offline utility, never the runtime reader.
    cache.import_directory(root)
    mapping, dropped, counts = {}, {}, Counter()
    extras = []
    for path in files:
        if path.parent.name != "positions":
            continue
        document = _read(path)
        if hashlib.sha256(document["fen"].encode()).hexdigest() != path.stem:
            raise ValueError(f"Position identity mismatch: {path.name}")
        if document.get('format') == FORMAT:
            for reference, request, result, current in _records(document, path.stem):
                if cache.get_reference(reference) is None:
                    raise ValueError('Invalid current source observation.')
                mapping[_key(reference)] = reference
                counts[f"preserved_{reference['namespace']}_measurements"] += 1
            counts['source_position_files'] += 1
            continue
        boards = {context: _board(document, context) for context in document["contexts"]}
        batches = defaultdict(list)
        for reference, request, result, current in _records(document, path.stem):
            converted, reason = _convert(boards[reference["context"]], reference["namespace"], request, result)
            if converted is None:
                mapping[_key(reference)] = None
                dropped[_key(reference)] = {"reference": reference, "reason": reason}
                counts[f"removed_{reference['namespace']}_measurements"] += 1
                continue
            board, request, result = converted
            group = (board.fen(), identity({"start_fen": board.root().fen(), "moves": [m.uci() for m in board.move_stack]}), reference["namespace"])
            batches[group].append((board, reference, request, result, current))
            counts[f"preserved_{reference['namespace']}_measurements"] += 1
        for rows in batches.values():
            rows.sort(key=lambda row: row[4])
            references = cache.put_many(rows[0][0], rows[0][1]["namespace"], [(row[2], row[3]) for row in rows])
            for row, reference in zip(rows, references, strict=True):
                mapping[_key(row[1])] = reference
        top = {key: value for key, value in document.items()
               if key not in ('fen', 'engines', 'requests', 'results', 'measurements', 'contexts')}
        histories = {}
        for context, history in document['contexts'].items():
            metadata = {key: value for key, value in history.items() if key not in ('start_fen', 'moves', 'evidence')}
            opaque = {namespace: store for namespace, store in history['evidence'].items()
                      if 'measurements' not in document
                      and (not isinstance(store, dict) or 'measurements' not in store)}
            if opaque:
                metadata['legacy_evidence'] = opaque
            if metadata:
                histories[context] = {**metadata, 'start_fen': history['start_fen'], 'moves': history['moves']}
        extras.append((path.stem, document['fen'], top, histories))
        counts['source_position_files'] += 1
    for position, fen, top, histories in extras:
        path = stage / 'positions' / f'{position}.json'
        if not path.exists() and not top and not histories:
            continue
        document = _read(path) if path.exists() else empty_document(fen)
        PositionCache._metadata_merge(document, top)
        for context, history in histories.items():
            existing = document['histories'].setdefault(context, {})
            PositionCache._metadata_merge(existing, history)
        write_json(path, document)

    @lru_cache(maxsize=16)
    def source(position):
        return _read(root / "positions" / f"{position}.json")

    for path in files:
        if path.parent.name == "positions":
            continue
        document = _read(path)
        manifest = document if path.parent.name == "games" else document.get("evidence", {})
        entries = manifest.get("positions")
        if entries is not None:
            for index, entry in enumerate(entries):
                expected = {"maia": {}, "stockfish": None, **entry["fields"]}
                revised = {"fields": entry["fields"], "maia": {}, "stockfish": None}
                for label, old in entry["maia"].items():
                    new = mapping[_key(old)]
                    if new is not None:
                        revised["maia"][label] = new
                        expected["maia"][label] = _payload(source(old["position"]), old)
                old = entry["stockfish"]
                if old is not None:
                    revised["stockfish"] = mapping[_key(old)]
                    if revised["stockfish"] is not None:
                        expected["stockfish"] = {**_payload(source(old["position"]), old),
                            "is_checkmate": chess.Board(entry["fields"]["fen"]).is_checkmate()}
                if revised["stockfish"] is None and index < len(entries)-1:
                    raise ValueError(f"Removing incomplete played-position Stockfish would invalidate {path.name}, ply {index}.")
                refs = [*revised["maia"].values(), revised["stockfish"]]
                values = cache.get_references(refs)
                actual = {"maia": dict(zip(revised["maia"], values[:-1], strict=True)),
                          "stockfish": values[-1], **revised["fields"]}
                if actual != expected:
                    raise ValueError(f"Preserved evidence changed in {path.name}, ply {index}.")
                entries[index] = revised
                counts["manifest_positions_verified"] += 1
            counts["manifests_verified"] += 1
        write_json(stage / path.relative_to(root), document)
    staged = {path.relative_to(stage).as_posix(): path for category in ("positions", "games", "game-metadata")
              for path in (stage / category).glob("*.json")}
    changes = [name for name, path in staged.items() if name not in fingerprints or _read(path) != _read(root / name)]
    removed = [name for name in fingerprints if name not in staged]
    for path in (stage / "positions").glob("*.json"):
        document = _read(path)
        counts['structured_position_files'] += 1
        counts['engines'] += len(document['engines'])
        counts['shared_results'] += len(document['shared_results'])
        rows = list(observations(document, strict=True))
        counts['measurements'] += len(rows)
        counts['inline_results'] += sum('result' in entry for _, _, entry in rows)
    counts["source_position_bytes"] = sum(path.stat().st_size for path in files if path.parent.name == "positions")
    counts["structured_position_bytes"] = sum(path.stat().st_size for path in (stage / "positions").glob("*.json"))
    mapping_log = [{"old": dict(zip(("position", "context", "namespace", "request", "measurement"), key, strict=True)),
                    "new": value} for key, value in mapping.items()]
    write_json(run / "reference-mapping.json", mapping_log)
    write_json(run / "removed-measurements.json", list(dropped.values()))
    report = {"cache_directory": str(root), "backup": str(backup), "staged": str(stage),
              "mapping": str(run / "reference-mapping.json"), "counts": dict(counts),
              "changed_files": changes, "removed_files": removed, "applied": False,
              "public_analysis_sha256": public_before, "public_analysis_unchanged": public_hashes() == public_before}
    write_json(run / "report.json", report)
    if apply:
        current = {path.relative_to(root).as_posix() for category in ("positions", "games", "game-metadata")
                   for path in (root / category).glob("*.json")}
        if current != set(fingerprints) or any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
                                              for name, digest in fingerprints.items()):
            raise RuntimeError("Cache changed during staging; no files were published.")
        for name in changes:
            write_json(root / name, _read(staged[name]))
        for name in removed:
            target = (root / name).resolve()
            if target.parent != root / "positions":
                raise ValueError("Only empty position files may be removed.")
            target.unlink()
        report["applied"] = True
        report["public_analysis_unchanged"] = public_hashes() == public_before
        write_json(run / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--public-directory", type=Path)
    args = parser.parse_args()
    report = consolidate(args.cache_directory, output_directory=args.output_directory, apply=args.apply,
                         public_directory=args.public_directory)
    print(json.dumps({**report, "changed_files": len(report["changed_files"]),
                      "removed_files": len(report["removed_files"]),
                      "public_analysis_sha256": len(report["public_analysis_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
