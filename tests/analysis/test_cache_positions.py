"""Position evidence merges histories and namespaces without losing concurrent work."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import chess

from analysis.cache.storage import identity, write_json
from analysis.cache.positions import PositionCache
from analysis.cache.structure import entries, result_value, measurement_id


def _board(order=0):
    board = chess.Board()
    moves = ("g1f3 g8f6 b1c3 b8c6", "b1c3 b8c6 g1f3 g8f6")[order]
    for move in moves.split():
        board.push_uci(move)
    return board


def _process_writer(directory, worker, start):
    start.wait(10)
    cache = PositionCache(directory)
    board = _board(worker % 2)
    for number in range(12):
        cache.put(board, "maia" if worker % 2 else "stockfish",
                  {"worker": worker, "number": number}, {"samples": [worker, number] * 20})
        cache.put(board, "shared", {"request": "same"}, {"worker": worker, "number": number})


def _crash_writer(directory):
    def crash(*_):
        os._exit(17)
    with patch("analysis.cache.positions.write_json", crash):
        PositionCache(directory).put(chess.Board(), "maia", {"elo": 1800}, {"complete": False})


class PositionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cache = PositionCache(self.directory)
        self.board = chess.Board()

    def _path(self, reference):
        return self.directory / "positions" / f"{reference['position']}.json"

    @staticmethod
    def _observation(document, reference):
        return next(observation for _, _, observation in entries(document)
                    if observation['measurement'] == reference['measurement'])

    def test_canonical_fen_identity_and_multiple_history_contexts(self):
        first, second = _board(0), _board(1)
        self.assertEqual(first.fen(), second.fen())
        request = {"elo": [1600, 1600], "model": "maia3"}
        one = self.cache.put(first, "maia", request, {"value": 0.7})
        two = self.cache.put(second, "maia", request, {"value": 0.3})
        fresh = self.cache.put(chess.Board(first.fen()), "maia", request, {"value": 0.5})
        self.assertEqual(one["position"], hashlib.sha256(first.fen().encode()).hexdigest())
        self.assertEqual(len({entry["context"] for entry in (one, two, fresh)}), 3)
        self.assertEqual(len(list(self.cache.positions_directory.glob("*.json"))), 1)
        for board, expected in ((first, 0.7), (second, 0.3), (chess.Board(first.fen()), 0.5)):
            self.assertEqual(self.cache.get(board, "maia", request), {"value": expected})

    def test_namespaces_requests_and_references_share_one_file(self):
        references = self.cache.put_many(self.board, "maia", [
            ({"elo": 1600}, {"value": 0.4}), ({"elo": 1800}, {"value": 0.6})])
        sf = self.cache.put(self.board, "stockfish", {"depth": 18}, {"cp": 20})
        self.assertEqual(self.cache.directory, self.directory)
        self.assertEqual(self.cache.get_reference(references[1]), {"value": 0.6})
        self.assertEqual(self.cache.get_reference(sf), {"cp": 20})
        self.assertIsNone(self.cache.get(self.board, "maia", {"elo": 2000}))
        self.assertIsNone(self.cache.get(self.board, "other", {"elo": 1600}))
        self.assertEqual(len(list(self.cache.positions_directory.glob("*.json"))), 1)

    def test_measurements_are_immutable_and_deduplicated_while_preferred_result_advances(self):
        request = {"depth": 18}
        old = self.cache.put(self.board, "stockfish", request, {"cp": 20})
        duplicate = self.cache.put(self.board, "stockfish", request, {"cp": 20})
        new = self.cache.put(self.board, "stockfish", request, {"cp": 30})
        self.assertEqual(old, duplicate)
        self.assertNotEqual(old["measurement"], new["measurement"])
        self.assertEqual(self.cache.get_reference(old), {"cp": 20})
        self.assertEqual(self.cache.get_reference(new), {"cp": 30})
        self.assertEqual(self.cache.get(self.board, "stockfish", request), {"cp": 30})
        document = json.loads(self._path(old).read_text())
        observations = list(entries(document))
        self.assertEqual(len(observations), 2)
        self.assertFalse(self._observation(document, old)['active'])
        self.assertTrue(self._observation(document, new)['active'])
        for namespace, context, observation in observations:
            self.assertEqual(observation['measurement'], measurement_id(context, namespace,
                observation['request'], result_value(document, observation)))

    def test_repeated_bodies_and_engine_signatures_are_shared_across_histories(self):
        request = {"elo": 1600, "engine": {"model": "same"}}
        first = self.cache.put(_board(), "maia", request, {"value": 0.5})
        second = self.cache.put(_board(1), "maia", request, {"value": 0.5})
        third = self.cache.put(_board(), "maia", {**request, "elo": 1800}, {"value": 0.5})
        document = json.loads(self._path(first).read_text())
        self.assertEqual(len(document["shared_results"]), 1)
        self.assertEqual(len(document["engines"]), 1)
        self.assertEqual(len(document["histories"]), 2)
        self.assertNotIn('requests', document)
        self.assertNotIn('measurements', document)
        self.assertEqual(len(list(entries(document))), 3)
        self.assertTrue(all('result_ref' in entry for _, _, entry in entries(document)))
        self.assertNotEqual(first["measurement"], second["measurement"])
        self.assertEqual(self.cache.get_references([first, second, third]), [{"value": 0.5}] * 3)
        self.assertIsNone(self.cache.get_reference({**first, "context": second["context"]}))
        self.assertEqual(self.cache.measurement_reference(_board(), "maia", request, {"value": 0.5}), first)

    def test_stockfish_board_facts_are_derived_without_duplicate_result_payloads(self):
        legal = [move.uci() for move in self.board.legal_moves]
        result = {"complete": True, "coverage_complete": True, "best_move": legal[0],
                  "engine_moves": [legal[0]], "cp_vec": dict.fromkeys(legal, 20),
                  "mate_vec": {}, "root_move_depth_vec": dict.fromkeys(legal, 4)}
        request = {"kind": "evaluation"}
        first = self.cache.put(self.board, "stockfish", request, result)
        second = self.cache.put(self.board, "stockfish", request, {**result, "is_checkmate": False})
        self.assertEqual(first, second)
        document = json.loads(self._path(first).read_text())
        self.assertEqual(len(list(entries(document))), 1)
        self.assertEqual(document['shared_results'], {})
        self.assertNotIn('is_checkmate', result_value(document, self._observation(document, first)))
        self.assertEqual(self.cache.get_reference(first), {**result, "is_checkmate": False})

    def test_reference_resolution_pins_current_head_and_obeys_refresh_view(self):
        request = {"elo": 1600}
        address = self.cache.reference(self.board, "maia", request)
        self.assertNotIn("measurement", address)
        self.assertIsNone(self.cache.resolve_reference(self.board, "maia", request))
        old = self.cache.put(self.board, "maia", request, {"value": 0.4})
        self.assertEqual(self.cache.resolve_reference(self.board, "maia", request), old)
        refresh = PositionCache(self.directory, reuse_existing=False)
        self.assertIsNone(refresh.resolve_reference(self.board, "maia", request))
        new = refresh.put(self.board, "maia", request, {"value": 0.6})
        self.assertEqual(refresh.resolve_reference(self.board, "maia", request), new)
        self.assertEqual(self.cache.resolve_reference(self.board, "maia", request), new)
        self.assertEqual(self.cache.get_reference(address), {"value": 0.6})
        self.assertEqual(self.cache.get_reference(old), {"value": 0.4})

    def test_batch_references_read_once_per_position_and_keep_misses_in_order(self):
        first = self.cache.put(self.board, "maia", {"elo": 1600}, {"value": 0.4})
        second = self.cache.put(self.board, "stockfish", {}, {"cp": 20})
        third = self.cache.put(_board(), "stockfish", {}, {"cp": 30})
        missing = {**first, "measurement": "f" * 64}
        with patch.object(self.cache, "_read", wraps=self.cache._read) as read:
            self.assertEqual(self.cache.get_references([second, None, third, first, missing]),
                             [{"cp": 20}, None, {"cp": 30}, {"value": 0.4}, None])
            self.assertEqual(read.call_count, 2)

    def test_records_return_only_current_measurements_for_matching_context_and_namespace(self):
        board = _board()
        self.assertEqual(self.cache.records(board, "stockfish"), [])
        self.cache.put(board, "stockfish", {"depth": 18}, {"cp": 10})
        current = self.cache.put(board, "stockfish", {"depth": 18}, {"cp": 20})
        deeper = self.cache.put(board, "stockfish", {"depth": 22}, {"cp": 30})
        self.cache.put(_board(1), "stockfish", {"depth": 18}, {"cp": 40})
        self.cache.put(board, "maia", {"elo": 1600}, {"value": 0.7})
        with patch.object(self.cache, "_read", wraps=self.cache._read) as read:
            records = self.cache.records(board, "stockfish")
            self.assertEqual(read.call_count, 1)
        self.assertEqual(records, [
            {"reference": current, "request": {"depth": 18}, "result": {"cp": 20}},
            {"reference": deeper, "request": {"depth": 22}, "result": {"cp": 30}},
        ])
        records[0]["result"]["cp"] = 999
        self.assertEqual(self.cache.get_reference(current), {"cp": 20})
        refresh = PositionCache(self.directory, reuse_existing=False)
        self.assertEqual(refresh.records(board, "stockfish"), [])
        fresh = refresh.put(board, "stockfish", {"depth": 24}, {"cp": 50})
        self.assertEqual(refresh.records(board, "stockfish"), [
            {"reference": fresh, "request": {"depth": 24}, "result": {"cp": 50}},
        ])

    def test_inactive_measurement_remains_available_only_by_its_pin(self):
        reference = self.cache.put(self.board, "stockfish", {"depth": 18}, {"cp": 20})
        document = json.loads(self._path(reference).read_text())
        self._observation(document, reference)['active'] = False
        write_json(self._path(reference), document)
        self.assertIsNone(self.cache.get(self.board, "stockfish", {"depth": 18}))
        self.assertIsNone(self.cache.resolve_reference(self.board, "stockfish", {"depth": 18}))
        self.assertEqual(self.cache.records(self.board, "stockfish"), [])
        self.assertEqual(self.cache.get_reference(reference), {"cp": 20})

    def test_batch_lookup_reads_position_once_and_preserves_request_order(self):
        self.cache.put_many(self.board, "maia", [({"elo": elo}, {"value": elo})
                                               for elo in (1600, 1800)])
        with patch.object(self.cache, "_read", wraps=self.cache._read) as read:
            results = self.cache.get_many(self.board, "maia", [{"elo": elo}
                                                              for elo in (1800, 2000, 1600)])
            self.assertEqual(results, [{"value": 1800}, None, {"value": 1600}])
            self.assertEqual(read.call_count, 1)
        with patch.object(self.cache, "_read") as read:
            self.assertEqual(self.cache.get_many(self.board, "maia", []), [])
            read.assert_not_called()

    def test_refresh_view_reuses_new_writes_and_preserves_other_evidence(self):
        old = self.cache.put(self.board, "maia", {"elo": 1600}, {"value": 0.1})
        keep = self.cache.put(self.board, "future", {"id": 7}, {"unchanged": True})
        refresh = PositionCache(self.directory, reuse_existing=False)
        self.assertIsNone(refresh.get_reference(old))
        self.assertEqual(refresh.get_many(self.board, "maia", [{"elo": 1600}]), [None])
        new = refresh.put_many(self.board, "maia", [({"elo": 1600}, {"value": 0.7}),
                                                  ({"elo": 1800}, {"value": 0.8})])
        self.assertIsNone(refresh.get_reference(old))
        self.assertEqual(refresh.get_reference(new[0]), {"value": 0.7})
        self.assertEqual(refresh.get_many(self.board, "maia", [{"elo": 1800}, {"elo": 1600}]),
                         [{"value": 0.8}, {"value": 0.7}])
        self.assertEqual(self.cache.get_reference(old), {"value": 0.1})
        self.assertEqual(self.cache.get(self.board, "maia", {"elo": 1600}), {"value": 0.7})
        self.assertEqual(self.cache.get_reference(keep), {"unchanged": True})
        # Starting another explicit refresh requires fresh measurements again.
        self.assertIsNone(PositionCache(self.directory, reuse_existing=False).get_reference(old))

    def test_refresh_view_can_reuse_merged_isolated_results(self):
        refresh = PositionCache(self.directory, reuse_existing=False)
        source = PositionCache(self.directory / "isolated")
        reference = source.put(self.board, "stockfish", {"depth": 18}, {"cp": 20})
        self.assertEqual(refresh.import_directory(source.directory), 1)
        self.assertEqual(refresh.get_reference(reference), {"cp": 20})

    def test_reference_path_cannot_escape_cache(self):
        reference = self.cache.put(self.board, "maia", {}, {"value": 0.5})
        for field in ("position", "context", "request", "measurement"):
            for invalid in ("../outside", "a" * 63, "a" * 65, None, 12, "A" * 64):
                self.assertIsNone(self.cache.get_reference({**reference, field: invalid}))
        for invalid in (None, [], {}, "bad"):
            self.assertIsNone(self.cache.get_reference(invalid))
        for namespace in (None, "", 42):
            self.assertIsNone(self.cache.get_reference({**reference, "namespace": namespace}))
            with self.assertRaises(ValueError):
                self.cache.put(self.board, namespace, {}, {})

    def test_returned_and_supplied_values_are_detached(self):
        request = {"elo": [1600, 1800]}
        result = {"policy": {"e2e4": 0.5}}
        reference = self.cache.put(self.board, "maia", request, result)
        request["elo"][0] = 800
        result["policy"]["e2e4"] = 0.1
        loaded = self.cache.get_reference(reference)
        self.assertEqual(loaded["policy"]["e2e4"], 0.5)
        loaded["policy"]["e2e4"] = 0.9
        self.assertEqual(self.cache.get_reference(reference)["policy"]["e2e4"], 0.5)

    def test_unknown_future_fields_and_categories_survive_merge(self):
        reference = self.cache.put(self.board, "maia", {}, {"value": 0.5})
        future = self.cache.put(self.board, "future-engine", {'version': 1}, {'features': [1, 2]})
        path = self._path(reference)
        document = json.loads(path.read_text())
        document["future"] = {"preserve": [1, 2]}
        document['histories'][reference['context']]['future'] = {'position_features': 'future format'}
        write_json(path, document)
        self.cache.put(self.board, "stockfish", {}, {"cp": 10})
        merged = json.loads(path.read_text())
        self.assertEqual(merged['future'], document['future'])
        self.assertEqual(merged['histories'], document['histories'])
        self.assertEqual(self.cache.get_reference(reference), {"value": 0.5})
        self.assertEqual(self.cache.get_reference(future), {'features': [1, 2]})

    def test_invalid_inputs_do_not_modify_existing_cache(self):
        reference = self.cache.put(self.board, "maia", {}, {"value": 0.5})
        before = self._path(reference).read_bytes()
        for request, result in (({}, {"bad": float("nan")}), ({"bad": float("inf")}, {})):
            with self.assertRaises(ValueError):
                self.cache.put(self.board, "maia", request, result)
        self.assertEqual(self._path(reference).read_bytes(), before)
        self.assertEqual(self.cache.put_many(self.board, "empty", []), [])
        self.assertEqual(self._path(reference).read_bytes(), before)

    def test_corrupt_file_is_miss_and_preserved_before_recovery(self):
        reference = self.cache.reference(self.board, "maia", {})
        path = self._path(reference)
        path.parent.mkdir()
        path.write_bytes(b'{"incomplete":')
        self.assertIsNone(self.cache.get_reference(reference))
        self.cache.put(self.board, "maia", {}, {"value": 0.5})
        archives = list((self.directory / ".position-corrupt").glob("*.json"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_bytes(), b'{"incomplete":')
        self.assertEqual(self.cache.get_reference(reference), {"value": 0.5})

    def test_invalid_record_hashes_and_nonfinite_json_are_not_hits(self):
        reference = self.cache.put(self.board, "maia", {}, {"value": 0.5})
        path = self._path(reference)
        original = json.loads(path.read_text())
        context = original['histories'][reference['context']]
        observation = self._observation(original, reference)
        observation['request'] = {'tampered': True}
        write_json(path, original)
        self.assertIsNone(self.cache.get_reference(reference))
        observation['request'] = {}
        observation['result'] = {'value': 0.6}
        write_json(path, original)
        self.assertIsNone(self.cache.get_reference(reference))
        observation['result'] = {'value': 0.5}
        context['moves'] = ['e2e4']
        write_json(path, original)
        self.assertIsNone(self.cache.get_reference(reference))
        context['moves'] = []
        observation['result'] = float('nan')
        path.write_text(json.dumps(original))
        self.assertIsNone(self.cache.get_reference(reference))

    def test_malformed_namespace_recovers_without_discarding_other_evidence(self):
        reference = self.cache.put(self.board, "stockfish", {}, {"cp": 12})
        path = self._path(reference)
        document = json.loads(path.read_text())
        document["evidence"]["maia"] = "invalid"
        write_json(path, document)
        self.cache.put(self.board, "maia", {}, {"value": 0.5})
        self.assertEqual(self.cache.get_reference(reference), {"cp": 12})
        self.assertEqual(len(list((self.directory / ".position-corrupt").glob("*.json"))), 1)

    def test_failed_publication_preserves_previous_and_releases_lock(self):
        reference = self.cache.put(self.board, "maia", {}, {"value": 0.5})
        with patch("analysis.cache.positions.write_json", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                self.cache.put(self.board, "stockfish", {}, {"cp": 20})
        self.assertEqual(self.cache.get_reference(reference), {"value": 0.5})
        self.cache.put(self.board, "stockfish", {}, {"cp": 20})
        self.assertEqual(self.cache.get(self.board, "stockfish", {}), {"cp": 20})

    def test_threads_with_independent_instances_do_not_lose_updates(self):
        start = threading.Barrier(12)

        def write(number):
            cache = PositionCache(self.directory)
            start.wait(10)
            return cache.put(_board(number % 2), "maia" if number % 3 else "stockfish",
                             {"number": number}, {"policy": [number] * 100})

        with ThreadPoolExecutor(12) as executor:
            references = list(executor.map(write, range(12)))
        for number, reference in enumerate(references):
            self.assertEqual(self.cache.get_reference(reference), {"policy": [number] * 100})
        self.assertEqual(len(list(self.cache.positions_directory.glob("*.json"))), 1)

    def test_processes_merge_namespaces_contexts_and_same_request(self):
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        processes = [context.Process(target=_process_writer, args=(str(self.directory), worker, start))
                     for worker in range(4)]
        try:
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            for worker in range(4):
                board = _board(worker % 2)
                for number in range(12):
                    self.assertEqual(self.cache.get(board, "maia" if worker % 2 else "stockfish",
                        {"worker": worker, "number": number}), {"samples": [worker, number] * 20})
                self.assertIsInstance(self.cache.get(board, "shared", {"request": "same"}), dict)
            self.assertEqual(len(list(self.cache.positions_directory.glob("*.json"))), 1)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(10)

    def test_crashed_process_releases_lock_and_preserves_previous_publication(self):
        reference = self.cache.put(self.board, "stockfish", {}, {"cp": 20})
        process = multiprocessing.get_context("spawn").Process(target=_crash_writer,
                                                                args=(str(self.directory),))
        try:
            process.start()
            process.join(20)
            self.assertEqual(process.exitcode, 17)
            self.assertEqual(self.cache.get_reference(reference), {"cp": 20})
            self.cache.put(self.board, "maia", {"elo": 1800}, {"complete": True})
            self.assertEqual(self.cache.get(self.board, "maia", {"elo": 1800}), {"complete": True})
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)

    def test_directory_import_merges_complete_requests_contexts_and_future_metadata(self):
        source = PositionCache(self.directory / "isolated")
        first, second = _board(0), _board(1)
        old = self.cache.put(first, "stockfish", {}, {"cp": 20, "obsolete": True})
        keep = self.cache.put(first, "maia", {"elo": 1600}, {"value": 0.5})
        newest = source.put(first, "stockfish", {}, {"cp": 30})
        other = source.put(second, "maia", {"elo": 2000}, {"value": 0.7})
        source.put(first, "future-engine", {"version": 1}, {"future-format": [1, 2]})
        for cache, reference, label in ((self.cache, old, "old"), (source, newest, "new")):
            path = cache.positions_directory / f"{reference['position']}.json"
            document = json.loads(path.read_text())
            document["future"] = {label: True}
            document["histories"][reference["context"]]["future"] = {label: True}
            write_json(path, document)
        self.assertEqual(self.cache.import_directory(source.directory), 1)
        self.assertEqual(self.cache.get_reference(old), {"cp": 20, "obsolete": True})
        self.assertEqual(self.cache.get_reference(newest), {"cp": 30})
        self.assertEqual(self.cache.get(first, "stockfish", {}), {"cp": 30})
        self.assertEqual(self.cache.get_reference(keep), {"value": 0.5})
        self.assertEqual(self.cache.get_reference(other), {"value": 0.7})
        self.assertEqual(self.cache.get(first, "future-engine", {"version": 1}),
                         {"future-format": [1, 2]})
        merged = json.loads(self._path(old).read_text())
        self.assertEqual(merged["future"], {"old": True, "new": True})
        self.assertEqual(merged["histories"][old["context"]]["future"], {"old": True, "new": True})
        self.assertEqual(self.cache.import_directory(self.directory), 0)

    def test_directory_import_rejects_corrupt_or_mismatched_sources(self):
        source = PositionCache(self.directory / "isolated")
        reference = source.put(self.board, "maia", {}, {"value": 0.5})
        path = source.positions_directory / f"{reference['position']}.json"
        original = json.loads(path.read_text())
        for corrupt in ("{", "null", json.dumps({**original, "fen": _board().fen()})):
            path.write_text(corrupt)
            self.assertEqual(self.cache.import_directory(source.directory), 0)
        self._observation(original, reference)["request"] = []
        write_json(path, original)
        self.assertEqual(self.cache.import_directory(source.directory), 0)
        self.assertFalse(self.cache.positions_directory.exists())

    def test_directory_import_and_concurrent_writers_keep_all_entries(self):
        source = PositionCache(self.directory / "isolated")
        for number in range(12):
            source.put(_board(number % 2), "imported", {"number": number}, {"value": number})
        start = threading.Barrier(2)

        def merge():
            start.wait(10)
            return self.cache.import_directory(source.directory)

        def write():
            start.wait(10)
            return [self.cache.put(_board(number % 2), "live", {"number": number}, {"value": number})
                    for number in range(12)]

        with ThreadPoolExecutor(2) as executor:
            imported, written = executor.submit(merge), executor.submit(write)
            self.assertEqual(imported.result(15), 1)
            references = written.result(15)
        for number, reference in enumerate(references):
            self.assertEqual(self.cache.get_reference(reference), {"value": number})
            self.assertEqual(self.cache.get(_board(number % 2), "imported", {"number": number}),
                             {"value": number})


if __name__ == "__main__":
    unittest.main()
