"""Offline normalization removes incomplete observations and preserves valid evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import chess

from analysis.cache.storage import identity, write_json
from analysis.cache.positions import PositionCache
from analysis.cache.structure import FORMAT, entries as observations
from tests.analysis.audit_position_cache import audit
from tests.analysis.consolidate_position_cache import consolidate


def _maia(board):
    legal = [] if board.is_game_over() else [move.uci() for move in board.legal_moves]
    return {"policy": {move: 1 / len(legal) for move in legal}, "value": 0.5}


def _stockfish(board):
    legal = [] if board.is_game_over() else [move.uci() for move in board.legal_moves]
    return {"complete": True, "coverage_complete": True, "best_move": legal[0] if legal else None,
            "engine_moves": legal[:1], "cp_vec": dict.fromkeys(legal, 20), "mate_vec": {},
            "root_move_depth_vec": dict.fromkeys(legal, 4)}


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache, self.output = self.directory / "cache", self.directory / "output"
        self.sf_request = {"kind": "evaluation", "policy_version": 2,
            "engine": {"stockfish": ["engine.exe", 100], "threads": 4, "hash_mb": 256},
            "depth": 18, "budget_seconds": 3., "max_budget_seconds": 8., "strategy": "bounded",
            "candidate_moves": ["e2e4", "d2d4"]}

    def _old_put(self, board, namespace, request, result):
        position = hashlib.sha256(board.fen().encode()).hexdigest()
        path = self.cache / "positions" / f"{position}.json"
        document = json.loads(path.read_text()) if path.exists() else {"fen": board.fen(), "contexts": {}}
        history = {"start_fen": board.root().fen(), "moves": [move.uci() for move in board.move_stack]}
        context = identity(history)
        evidence = document["contexts"].setdefault(context, {**history, "evidence": {}})["evidence"]
        store = evidence.setdefault(namespace, {"requests": {}, "measurements": {}})
        record = {"request": request, "result": result}
        request_hash, measurement = identity(request), identity(record)
        store["requests"][request_hash] = measurement
        store["measurements"][measurement] = record
        write_json(path, document)
        return {"position": position, "context": context, "namespace": namespace,
                "request": request_hash, "measurement": measurement}

    def _fixtures(self):
        board = chess.Board()
        maia = {}
        for rating in (1600, 1800):
            maia[f"maia_kdd_{rating}"] = self._old_put(board, "maia", {
                "version": 1, "engine": {"maia": "model", "history_window": 8},
                "own_rating": rating, "opponent_rating": rating}, _maia(board))
        sf = self._old_put(board, "stockfish", self.sf_request, _stockfish(board))
        duplicate = self._old_put(board, "stockfish", {**self.sf_request, "engine": None},
                                  {**_stockfish(board), "is_checkmate": False})
        entries = [{"fields": {"fen": board.fen(), "ply": 0}, "maia": maia, "stockfish": duplicate}]
        board.push_uci("e2e4")
        maia = {"maia_kdd_1600": self._old_put(board, "maia", {"version": 1, "engine": None,
                    "own_rating": 1600, "opponent_rating": 1600}, {"policy": {}, "value": None, "available": False})}
        missing = self._old_put(board, "stockfish", {**self.sf_request, "engine": None},
                               {"complete": False, "available": False})
        entries.append({"fields": {"fen": board.fen(en_passant="fen"), "ply": 1}, "maia": maia, "stockfish": missing})
        write_json(self.cache / "games" / "game.json", {"positions": entries})
        write_json(self.cache / "game-metadata" / "metadata.json", {"analysis": {"schema_version": 8}, "evidence": {"positions": entries}})
        self.public = self.directory / "analysis.json"
        self.public.write_text('{"prepared":"unchanged"}')
        return entries, sf

    def test_apply_pools_duplicate_payloads_and_signatures_and_preserves_backups(self):
        entries, sf = self._fixtures()
        before = {path.relative_to(self.cache).as_posix(): path.read_bytes() for path in self.cache.rglob("*.json")}
        report = consolidate(self.cache, output_directory=self.output, apply=True)
        self.assertTrue(report["applied"])
        self.assertEqual(report["counts"]["manifest_positions_verified"], 4)
        self.assertEqual(report["counts"]["removed_maia_measurements"], 1)
        self.assertEqual(report["counts"]["removed_stockfish_measurements"], 1)
        with ZipFile(report["backup"]) as archive:
            self.assertEqual(set(archive.namelist()), set(before))
            for name, data in before.items():
                self.assertEqual(archive.read(name), data)
        revised = json.loads((self.cache / "games" / "game.json").read_text())["positions"]
        cache = PositionCache(self.cache)
        self.assertEqual(cache.get_reference(revised[0]["stockfish"]), {**_stockfish(chess.Board()), "is_checkmate": False})
        self.assertEqual(revised[1], {"fields": entries[1]["fields"], "maia": {}, "stockfish": None})
        document = json.loads((self.cache / "positions" / (sf["position"] + ".json")).read_text())
        self.assertEqual(document['format'], FORMAT)
        self.assertEqual(len(document['shared_results']), 2)
        self.assertEqual(len(document["engines"]), 2)
        self.assertEqual(len(list(observations(document))), 4)
        self.assertNotIn('requests', document)
        self.assertNotIn('measurements', document)
        self.assertEqual(len(report["removed_files"]), 1)
        self.assertEqual(self.public.read_text(), '{"prepared":"unchanged"}')
        checked = audit(self.cache)
        self.assertEqual(checked["issues"], [])
        for name in ("orphan_engines", "orphan_shared_results", "duplicate_result_bodies"):
            self.assertEqual(checked["totals"][name], 0)

    def test_forced_missing_values_removed_but_complete_terminal_maia_retained(self):
        forced = chess.Board("r4B2/3k1p1R/6p1/3p2P1/8/1P6/5P2/1Kq5 w - - 0 39")
        self.assertEqual(forced.legal_moves.count(), 1)
        self._old_put(forced, "maia", {"own_rating": 1600, "opponent_rating": 1600, "engine": None},
            {"policy": {next(iter(forced.legal_moves)).uci(): 1.}, "value": None, "policy_source": "only_legal_move"})
        terminal = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        for rating in (1600, 1800):
            self._old_put(terminal, "maia", {"own_rating": rating, "opponent_rating": rating, "engine": None}, _maia(terminal))
        report = consolidate(self.cache, output_directory=self.output, apply=True)
        self.assertEqual(report["counts"]["removed_maia_measurements"], 1)
        self.assertEqual(report["counts"]["preserved_maia_measurements"], 2)
        self.assertEqual(report["counts"]["shared_results"], 1)
        self.assertEqual(report["counts"]["structured_position_files"], 1)

    def test_dry_run_never_changes_originals_and_conversion_is_idempotent(self):
        self._fixtures()
        before = {path: path.read_bytes() for path in self.cache.rglob("*.json")}
        plan = consolidate(self.cache, output_directory=self.output)
        self.assertFalse(plan["applied"])
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        consolidate(self.cache, output_directory=self.output, apply=True)
        converted = {path: path.read_bytes() for path in self.cache.rglob("*.json")}
        repeated = consolidate(self.cache, output_directory=self.output, apply=True)
        self.assertEqual(repeated["changed_files"], [])
        self.assertEqual(repeated["removed_files"], [])
        self.assertEqual({path: path.read_bytes() for path in converted}, converted)

    def test_previous_normalized_format_converts_without_runtime_legacy_helpers(self):
        board = chess.Board()
        history = {'start_fen': board.fen(), 'moves': []}
        context = identity(history)
        request = {'source': 'fixture', 'engine': {'model': 'local'}}
        value = {'features': [1, 2]}
        engine, request_id, result_id = identity(request['engine']), identity(request), identity(value)
        observation = {'context': context, 'namespace': 'future', 'request': request_id, 'result': result_id}
        measurement = identity(observation)
        position = hashlib.sha256(board.fen().encode()).hexdigest()
        source = {'fen': board.fen(), 'engines': {engine: request['engine']},
                  'requests': {request_id: {**request, 'engine': engine}},
                  'results': {result_id: value}, 'measurements': {measurement: observation},
                  'contexts': {context: {**history, 'note': 'preserve',
                      'evidence': {'future': {request_id: measurement}}}}, 'custom': {'keep': True}}
        path = self.cache / 'positions' / f'{position}.json'
        write_json(path, source)
        self.assertIsNone(PositionCache(self.cache).get(board, 'future', request))
        consolidate(self.cache, output_directory=self.output, apply=True)
        reference = {'position': position, 'context': context, 'namespace': 'future',
                     'request': request_id, 'measurement': measurement}
        self.assertEqual(PositionCache(self.cache).get_reference(reference), value)
        current = json.loads(path.read_text())
        self.assertEqual(current['format'], FORMAT)
        self.assertEqual(current['histories'][context]['note'], 'preserve')
        self.assertEqual(current['custom'], {'keep': True})
        self.assertEqual(audit(self.cache)['issues'], [])

    def test_unresolvable_manifest_aborts_before_publication(self):
        self._fixtures()
        path = self.cache / "games" / "game.json"
        document = json.loads(path.read_text())
        document["positions"][0]["stockfish"]["measurement"] = "f" * 64
        write_json(path, document)
        before = {path: path.read_bytes() for path in self.cache.rglob("*.json")}
        with self.assertRaises((KeyError, ValueError)):
            consolidate(self.cache, output_directory=self.output, apply=True)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual(len(list(self.output.rglob("original-cache.zip"))), 1)


if __name__ == "__main__":
    unittest.main()
