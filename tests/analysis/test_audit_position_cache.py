"""The offline audit checks inactive pins and physical payload sharing without writes."""
import json
from pathlib import Path
import tempfile
import unittest

import chess

from analysis.cache.storage import write_json
from analysis.cache.positions import PositionCache
from analysis.cache.structure import entries
from tests.analysis.audit_position_cache import audit
from tests.analysis.test_cache_structure import request, result


class PositionAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = PositionCache(self.root)
        self.board = chess.Board()

    def path(self, reference):
        return self.root / 'positions' / f"{reference['position']}.json"

    def test_inactive_manifest_pin_is_valid_and_audit_does_not_write(self):
        old = self.cache.put(self.board, 'stockfish', request(18, 5), result(self.board, 10))
        self.cache.put(self.board, 'stockfish', request(24, 10), result(self.board, 20))
        write_json(self.root / 'games' / 'game.json', {'positions': [
            {'fields': {'fen': self.board.fen(), 'ply': 0}, 'maia': {}, 'stockfish': old}]})
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        report = audit(self.root)
        self.assertTrue(report['read_only'])
        self.assertEqual(report['issues'], [])
        self.assertEqual(report['totals']['inactive_measurements_pinned_by_manifests'], 1)
        self.assertEqual(report['totals']['manifest_references'], 1)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()})

    def test_duplicate_physical_body_and_single_use_pool_are_reported(self):
        first = self.cache.put(self.board, 'future', {'id': 1}, {'value': 10})
        self.cache.put(self.board, 'future', {'id': 2}, {'value': 10})
        self.assertEqual(audit(self.root)['issues'], [])
        document = json.loads(self.path(first).read_text())
        observation = next(observation for _, _, observation in entries(document))
        observation['result'] = document['shared_results'][observation.pop('result_ref')]
        write_json(self.path(first), document)
        report = audit(self.root)
        self.assertEqual(report['totals']['duplicate_result_bodies'], 1)
        self.assertEqual({issue['problem'] for issue in report['issues']},
            {'duplicate physical result bodies', 'shared result must be referenced by multiple observations'})

    def test_corrupt_inactive_observation_does_not_hide_unresolved_manifest_pin(self):
        old = self.cache.put(self.board, 'stockfish', request(18, 5), result(self.board, 10))
        self.cache.put(self.board, 'stockfish', request(24, 10), result(self.board, 20))
        write_json(self.root / 'game-metadata' / 'artifact.json', {'evidence': {'positions': [
            {'fields': {'fen': self.board.fen(), 'ply': 0}, 'maia': {}, 'stockfish': old}]}})
        document = json.loads(self.path(old).read_text())
        observation = next(entry for _, _, entry in entries(document) if entry['measurement'] == old['measurement'])
        observation['result']['cp_vec']['e2e4'] = 999
        write_json(self.path(old), document)
        report = audit(self.root)
        self.assertIn('invalid measurement identity', {issue['problem'] for issue in report['issues']})
        self.assertIn('unresolved manifest reference', {issue['problem'] for issue in report['issues']})

    def test_malformed_manifest_and_noncurrent_position_are_reported(self):
        reference = self.cache.reference(self.board, 'future', {})
        write_json(self.path(reference), {'fen': self.board.fen(), 'contexts': {}})
        manifest = self.root / 'games' / 'bad.json'
        manifest.parent.mkdir()
        manifest.write_text('{')
        report = audit(self.root)
        self.assertEqual(len(report['issues']), 2)
        self.assertEqual(report['issues'][0]['problem'], 'invalid current structured document')


if __name__ == '__main__':
    unittest.main()
