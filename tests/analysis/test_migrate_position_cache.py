"""Offline conversion preserves measurements, references and engine identity."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import chess

from analysis.cache.storage import identity, write_json
from analysis.cache.positions import PositionCache
from analysis.cache.requests import stockfish_initial_request
from analysis.cache.structure import entries, request_value, result_value
from analysis.cache.artifacts import AnalysisStore
from tests.analysis.migrate_position_cache import convert_document, rewrite_pins, upgrade_identity, migrate
from tests.analysis.test_stockfish_search import evaluation_frame


def normalized_document(board, request, result):
    result = {key: value for key, value in result.items() if key != 'is_checkmate'}
    history = {'start_fen': board.root().fen(), 'moves': [move.uci() for move in board.move_stack]}
    context, engine_key, request_key, result_key = map(identity, (history, request['engine'], request, result))
    row = {'context': context, 'namespace': 'stockfish', 'request': request_key, 'result': result_key}
    measure = identity(row)
    document = {'fen': board.fen(), 'engines': {engine_key: request['engine']},
        'requests': {request_key: {**request, 'engine': engine_key}}, 'results': {result_key: result},
        'measurements': {measure: row}, 'contexts': {context: {**history, 'evidence': {'stockfish': {request_key: measure}}}}}
    return document, {'position': hashlib.sha256(board.fen().encode()).hexdigest(),
        'context': context, 'namespace': 'stockfish', 'request': request_key, 'measurement': measure}


class MigrationTests(unittest.TestCase):
    def test_stage_and_publish_preserve_originals_and_artifact_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            request = stockfish_initial_request({'stockfish': 'fixture'}, 18, 2, 8, 'bounded', {})
            board = chess.Board()
            document, reference = normalized_document(board, request, evaluation_frame(board))
            source = cache / 'positions' / (reference['position'] + '.json')
            write_json(source, document)
            public = {'start_fen': board.fen(), 'moves': []}
            artifact = root / 'analysis.json'
            write_json(artifact, public)
            manifest = {'positions': [{'fields': {'fen': board.fen(), 'ply': 0}, 'maia': {}, 'stockfish': reference}]}
            write_json(cache / 'games' / (identity([board.fen(), []]) + '.json'), manifest)
            write_json(cache / 'game-metadata' / (identity(identity(public)) + '.json'), {'evidence': manifest})
            originals = {path.relative_to(cache): path.read_bytes() for path in cache.rglob('*.json')}
            preview = migrate(cache, root / 'preview', [artifact])
            self.assertFalse(preview['applied'])
            self.assertEqual({path.relative_to(cache): path.read_bytes() for path in cache.rglob('*.json')}, originals)
            report = migrate(cache, root / 'apply', [artifact], apply=True)
            self.assertTrue(report['applied'])
            self.assertEqual(report['artifacts'], [str(artifact.resolve())])
            self.assertEqual(json.loads(artifact.read_text()), public)
            for path, content in originals.items():
                self.assertEqual((Path(report['backup']) / path).read_bytes(), content)
            restored = AnalysisStore(cache).load_positions(artifact)
            self.assertEqual(restored[0]['stockfish']['cp_vec'], evaluation_frame(board)['cp_vec'])

    def test_valid_asset_fingerprint_and_pin_are_upgraded_without_changing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stockfish'
            path.write_bytes(b'fixture')
            request = stockfish_initial_request({'stockfish': [str(path), path.stat().st_mtime_ns]}, 18, 2, 8, 'bounded', {})
            request['max_budget_seconds'] = 8.0
            board = chess.Board()
            result = evaluation_frame(board, depth=6)
            old, reference = normalized_document(board, request, result)
            document, pins = convert_document(old)
            _, _, observation = next(entries(document))
            new_request = request_value(document, observation)
            self.assertTrue(new_request['engine']['stockfish'].startswith('sha256:'))
            self.assertEqual(new_request['depth'], 18)
            self.assertEqual(new_request['max_budget_seconds'], 2)
            self.assertTrue(observation['active'])
            self.assertEqual(result_value(document, observation),
                             {key: value for key, value in result.items() if key != 'is_checkmate'})
            converted = rewrite_pins({'positions': [reference]}, pins)['positions'][0]
            self.assertEqual(converted['measurement'], observation['measurement'])
            self.assertEqual(converted['request'], identity(new_request))
            self.assertEqual(len(PositionCache._decode(document, strict=True)), 1)

    def test_unverifiable_engine_is_preserved_only_for_exact_pins(self):
        request = stockfish_initial_request({'stockfish': ['missing-engine', 1]}, 18, 2, 8, 'bounded', {})
        document, _ = normalized_document(chess.Board(), request, evaluation_frame(chess.Board()))
        converted, _ = convert_document(document)
        observation = next(entries(converted))[2]
        self.assertFalse(observation['active'])
        self.assertEqual(request_value(converted, observation), request)

    def test_mismatched_asset_timestamp_never_gets_current_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'engine'
            path.write_bytes(b'fixture')
            request = {'engine': {'stockfish': [str(path), 1]}}
            self.assertEqual(upgrade_identity(request), request)

    def test_corrupt_measurement_aborts_conversion(self):
        request = stockfish_initial_request({'stockfish': 'fixture'}, 18, 2, 8, 'bounded', {})
        document, _ = normalized_document(chess.Board(), request, evaluation_frame(chess.Board()))
        document['results'][next(iter(document['results']))]['complete'] = False
        with self.assertRaisesRegex(ValueError, 'result identity'):
            convert_document(document)


if __name__ == '__main__':
    unittest.main()
