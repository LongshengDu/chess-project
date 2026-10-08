"""Readable storage keeps source identities during compatible evidence selection."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import chess

from analysis.cache.storage import identity, write_json
from analysis.cache.positions import PositionCache
from analysis.cache.structure import FORMAT, encode_records, entries, measurement_id
from analysis.stockfish_search import BOUNDED_POLICY_VERSION


def request(depth=18, seconds=5, *, candidates=None):
    return {'kind': 'evaluation', 'policy_version': BOUNDED_POLICY_VERSION,
            'engine': {'stockfish': 'fixture', 'threads': 1, 'hash_mb': 16},
            'strategy': 'bounded', 'depth': depth, 'budget_seconds': float(seconds),
            'max_budget_seconds': float(seconds), 'candidate_moves': candidates or []}


def result(board, score=10):
    legal = sorted(move.uci() for move in board.legal_moves)
    return {'strategy': 'bounded', 'complete': True, 'coverage_complete': True,
            'cp_vec': dict.fromkeys(legal, score), 'mate_vec': {},
            'root_move_depth_vec': dict.fromkeys(legal, 12), 'best_move': legal[0],
            'engine_moves': legal[:4], 'target_reached': False}


def history(order=0, *, irreversible=False):
    board = chess.Board()
    for move in ('g1f3 g8f6 b1c3 b8c6', 'b1c3 b8c6 g1f3 g8f6')[order].split():
        board.push_uci(move)
    if irreversible:
        board.push_uci('e2e4')
    return board


class StructuredPositionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = PositionCache(self.directory)
        self.board = chess.Board()

    def path(self, reference):
        return self.directory / 'positions' / f"{reference['position']}.json"

    def test_readable_groups_keep_unique_result_inline_and_history_once(self):
        reference = self.cache.put(self.board, 'stockfish', request(), result(self.board))
        document = json.loads(self.path(reference).read_text())
        self.assertEqual(document['format'], FORMAT)
        observations = document['evidence']['stockfish']['evaluation']['bounded'][reference['context']]
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]['request']['depth'], 18)
        self.assertEqual(observations[0]['result']['cp_vec']['e2e4'], 10)
        self.assertEqual(document['shared_results'], {})
        self.assertEqual(document['histories'], {reference['context']:
            {'start_fen': self.board.fen(), 'moves': []}})
        for obsolete in ('contexts', 'requests', 'results', 'measurements'):
            self.assertNotIn(obsolete, document)

    def test_pure_encoder_preserves_existing_external_measurement_address(self):
        record = {'context': identity({'start_fen': self.board.fen(), 'moves': []}),
                  'namespace': 'stockfish', 'request': request(), 'result': result(self.board)}
        document = encode_records(self.board.fen(), {record['context']:
            {'start_fen': self.board.fen(), 'moves': []}}, [record, {**record, 'active': False}])
        reference = self.cache.measurement_reference(self.board, 'stockfish', record['request'], record['result'])
        self.assertEqual(reference['measurement'], measurement_id(record['context'], 'stockfish',
            record['request'], record['result']))
        write_json(self.path(reference), document)
        self.assertEqual(len(list(entries(document))), 1)
        self.assertTrue(next(entries(document))[2]['active'])
        self.assertEqual(self.cache.get_reference(reference)['cp_vec'], record['result']['cp_vec'])
        self.assertFalse((self.directory / '.position-locks').exists())

    def test_sufficient_request_selection_keeps_source_pin_without_alias_writes(self):
        stored = request(24, 10, candidates=['e2e4', 'd2d4'])
        reference = self.cache.put(self.board, 'stockfish', stored, result(self.board))
        before = self.path(reference).read_bytes()
        requested = request(18, 5, candidates=['e2e4'])
        with patch.object(self.cache, '_read', wraps=self.cache._read) as read:
            selected = self.cache.select_many(self.board, 'stockfish', [requested, request(28, 5)])
        self.assertEqual(read.call_count, 1)
        self.assertEqual(selected[0]['reference'], reference)
        self.assertEqual(selected[0]['request'], stored)
        self.assertNotEqual(selected[0]['reference']['request'], identity(requested))
        self.assertIsNone(selected[1])
        self.assertEqual(self.cache.resolve_reference(self.board, 'stockfish', requested), reference)
        with patch.object(self.cache, '_read', wraps=self.cache._read) as read:
            pinned = self.cache.get_reference_records([reference, reference, None])
        self.assertEqual(read.call_count, 1)
        self.assertEqual(pinned[:2], [selected[0], selected[0]])
        self.assertIsNone(pinned[2])
        self.assertEqual(before, self.path(reference).read_bytes())

    def test_requested_budget_dominance_preserves_pins_and_rejects_weaker_replacement(self):
        old = self.cache.put(self.board, 'stockfish', request(18, 5), result(self.board, 10))
        stronger = self.cache.put(self.board, 'stockfish', request(24, 10), result(self.board, 20))
        weaker = self.cache.put(self.board, 'stockfish', request(16, 3), result(self.board, 30))
        self.assertEqual([record['reference'] for record in self.cache.records(self.board, 'stockfish')], [stronger])
        self.assertEqual(self.cache.select_record(self.board, 'stockfish', request(18, 5))['reference'], stronger)
        self.assertEqual([self.cache.get_reference(ref)['cp_vec']['e2e4']
                          for ref in (old, stronger, weaker)], [10, 20, 30])

    def test_incomparable_budgets_remain_until_a_combined_request_supersedes_both(self):
        first = self.cache.put(self.board, 'stockfish', request(24, 5), result(self.board, 10))
        second = self.cache.put(self.board, 'stockfish', request(18, 10), result(self.board, 20))
        self.assertEqual({record['reference']['measurement'] for record in self.cache.records(self.board, 'stockfish')},
                         {first['measurement'], second['measurement']})
        self.assertIsNone(self.cache.get(self.board, 'stockfish', request(24, 10)))
        combined = self.cache.put(self.board, 'stockfish', request(24, 10), result(self.board, 30))
        self.assertEqual([record['reference'] for record in self.cache.records(self.board, 'stockfish')], [combined])
        self.assertIsNotNone(self.cache.get_reference(first))
        self.assertIsNotNone(self.cache.get_reference(second))

    def test_refresh_can_reuse_own_weaker_observation_without_replacing_global_preference(self):
        strongest = self.cache.put(self.board, 'stockfish', request(24, 10), result(self.board, 10))
        refresh = PositionCache(self.directory, reuse_existing=False)
        self.assertIsNone(refresh.get(self.board, 'stockfish', request(18, 5)))
        fresh = refresh.put(self.board, 'stockfish', request(18, 5), result(self.board, 20))
        self.assertEqual(refresh.select_record(self.board, 'stockfish', request(18, 5))['reference'], fresh)
        self.assertIsNone(refresh.get_reference(strongest))
        self.assertEqual(self.cache.select_record(self.board, 'stockfish', request(18, 5))['reference'], strongest)
        latest = refresh.put(self.board, 'stockfish', request(18, 5), result(self.board, 30))
        self.assertEqual(refresh.select_record(self.board, 'stockfish', request(18, 5))['reference'], latest)
        repeated = refresh.put(self.board, 'stockfish', request(18, 5), result(self.board, 20))
        self.assertEqual(repeated, fresh)
        self.assertEqual(refresh.select_record(self.board, 'stockfish', request(18, 5))['reference'], fresh)

    def test_compatible_history_reuses_original_observation_without_adding_alias_history(self):
        first, second = history(0, irreversible=True), history(1, irreversible=True)
        self.assertEqual(first.fen(), second.fen())
        reference = self.cache.put(first, 'stockfish', request(), result(first))
        before = self.path(reference).read_bytes()
        self.assertEqual(self.cache.select_record(second, 'stockfish', request())['reference'], reference)
        self.assertNotEqual(reference['context'], self.cache.reference(second, 'stockfish', request())['context'])
        with patch.object(self.cache, '_read', wraps=self.cache._read) as read:
            self.assertTrue(self.cache.references_match(second, [reference] * 3))
        self.assertEqual(read.call_count, 1)
        self.assertTrue(self.cache.reference_matches(second, reference))
        self.assertEqual(before, self.path(reference).read_bytes())

    def test_refresh_import_keeps_archived_pins_without_selecting_them(self):
        source = PositionCache(self.directory / 'source')
        current = source.put(self.board, 'stockfish', request(), result(self.board, 10))
        archived = source.put(self.board, 'stockfish', request(), result(self.board, 20))
        source.put(self.board, 'stockfish', request(), result(self.board, 10))
        refresh = PositionCache(self.directory, reuse_existing=False)
        self.assertEqual(refresh.import_directory(source.directory), 1)
        self.assertEqual(refresh.select_record(self.board, 'stockfish', request())['reference'], current)
        self.assertEqual([record['reference'] for record in refresh.records(self.board, 'stockfish')], [current])
        self.assertEqual(refresh.get_reference(archived)['cp_vec']['e2e4'], 20)
        fresh = refresh.put(self.board, 'stockfish', request(), result(self.board, 30))
        refresh.import_directory(source.directory)
        self.assertIsNotNone(refresh.get_reference(fresh))
        self.assertEqual(refresh.select_record(self.board, 'stockfish', request())['reference'], current)

    def test_refresh_selection_preserves_concurrent_publication_order(self):
        refresh = PositionCache(self.directory, reuse_existing=False)
        first_published, release_first, second_done = (threading.Event() for _ in range(3))
        remember = refresh._remember_writes
        errors = []

        def remember_after_pause(references, **kwargs):
            if threading.current_thread().name == 'first-publisher':
                first_published.set()
                if not release_first.wait(3):
                    raise TimeoutError('First publication was not released.')
            remember(references, **kwargs)

        def publish(score):
            try:
                refresh.put(self.board, 'stockfish', request(), result(self.board, score))
            except Exception as error:
                errors.append(error)
            finally:
                if score == 20:
                    second_done.set()

        with patch.object(refresh, '_remember_writes', side_effect=remember_after_pause):
            first = threading.Thread(name='first-publisher', target=publish, args=(10,))
            second = threading.Thread(target=publish, args=(20,))
            first.start()
            try:
                self.assertTrue(first_published.wait(3))
                second.start()
                # A second commit must not get ahead of the first publication's
                # refresh bookkeeping; older code finishes it during this wait.
                second_done.wait(0.25)
            finally:
                release_first.set()
                first.join(3)
                if second.ident is not None:
                    second.join(3)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.cache.get(self.board, 'stockfish', request())['cp_vec']['e2e4'], 20)
        self.assertEqual(refresh.get(self.board, 'stockfish', request())['cp_vec']['e2e4'], 20)

    def test_live_repetition_context_is_not_an_alias(self):
        first, second = history(), history(1)
        reference = self.cache.put(first, 'stockfish', request(), result(first))
        self.assertIsNone(self.cache.get(second, 'stockfish', request()))
        self.assertFalse(self.cache.reference_matches(second, reference))

    def test_unknown_engine_and_old_policy_stay_pinned_without_active_reuse(self):
        for changes in ({'engine': {}}, {'policy_version': BOUNDED_POLICY_VERSION - 1}):
            old_request = {**request(), **changes}
            reference = self.cache.put(self.board, 'stockfish', old_request, result(self.board))
            self.assertIsNotNone(self.cache.get_reference(reference))
            self.assertIsNone(self.cache.get(self.board, 'stockfish', old_request))
        self.assertEqual(self.cache.records(self.board, 'stockfish'), [])
        current = self.cache.put(self.board, 'stockfish', request(), result(self.board))
        self.assertEqual(self.cache.select_record(self.board, 'stockfish', request())['reference'], current)

    def test_current_reader_does_not_accept_a_legacy_document(self):
        reference = self.cache.reference(self.board, 'stockfish', {})
        legacy = {'fen': self.board.fen(), 'engines': {}, 'requests': {}, 'results': {},
                  'measurements': {}, 'contexts': {}}
        write_json(self.path(reference), legacy)
        self.assertIsNone(self.cache.get(self.board, 'stockfish', {}))
        self.assertIsNone(self.cache._read(reference['position']))

    def test_import_rebuilds_groups_without_aliasing_compatible_histories(self):
        first, second = history(0, irreversible=True), history(1, irreversible=True)
        source = PositionCache(self.directory / 'isolated')
        old = self.cache.put(first, 'stockfish', request(18, 5), result(first, 10))
        fresh = source.put(second, 'stockfish', request(24, 10), result(second, 20))
        self.assertEqual(self.cache.import_directory(source.directory), 1)
        document = json.loads(self.path(old).read_text())
        self.assertEqual(set(document['histories']), {old['context'], fresh['context']})
        self.assertEqual(len(list(entries(document))), 2)
        self.assertEqual(set(self.cache.get_reference(old)['cp_vec'].values()), {10})
        self.assertEqual(set(self.cache.get_reference(fresh)['cp_vec'].values()), {20})
        self.assertTrue(self.cache.reference_matches(first, fresh))

    def test_corrupt_observation_repair_keeps_unrelated_entries(self):
        first = self.cache.put(self.board, 'future', {'id': 1}, {'value': 10})
        second = self.cache.put(self.board, 'future', {'id': 2}, {'value': 20})
        document = json.loads(self.path(first).read_text())
        observation = next(entry for _, _, entry in entries(document) if entry['measurement'] == first['measurement'])
        observation['result'] = {'tampered': True}
        write_json(self.path(first), document)
        self.assertIsNone(self.cache.get_reference(first))
        self.cache.put(self.board, 'future', {'id': 1}, {'value': 10})
        self.assertEqual(self.cache.get_reference(first), {'value': 10})
        self.assertEqual(self.cache.get_reference(second), {'value': 20})
        self.assertEqual(len(list((self.directory / '.position-corrupt').glob('*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
