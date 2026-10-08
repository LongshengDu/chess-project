"""Engine-free recalculation and immutable game evidence across configuration changes."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import chess.pgn

from analysis.cache.positions import PositionCache
from analysis.cache.storage import identity
from analysis.cache.artifacts import AnalysisStore
from analysis.game.pipeline import analyze_game
from analysis.cache.session import CachedAnalysisSession
from coach.coach import main
from tests.coach.fixtures import FakeAnalysisSession


class CachedSessionTests(unittest.TestCase):
    def setUp(self):
        self.game = chess.pgn.read_game(io.StringIO(
            '[WhiteElo "1400"]\n[BlackElo "1600"]\n[Site "https://lichess.org"]\n'
            '[TimeControl "300+0"]\n\n1. e4 e5 *'))
        self.session = FakeAnalysisSession(self.game.board().fen())
        self.addCleanup(self.session._temp.cleanup)
        self.directory = self.session.cache.directory
        self.store = AnalysisStore(self.directory)
        self.analysis = analyze_game(self.game, self.session, progress=lambda _: None)
        self.path = self.directory / 'analysis.json'
        self.store.save(self.path, self.analysis)

    def test_rebuild_uses_pinned_measurements_without_engines_or_search_settings(self):
        before = {p: p.read_bytes() for p in (self.directory/'positions').glob('*.json')}
        with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')):
            with CachedAnalysisSession(self.game, self.directory) as session:
                result = analyze_game(self.game, session, progress=lambda _: None)
        for field in ('moves', 'positions', 'accuracy_curve', 'performance'):
            self.assertEqual(result[field], self.analysis[field], field)
        self.assertEqual(session.stats['stockfish_calls'], 0)
        self.assertEqual(session.stats['maia_batches'], 0)
        self.store.save(self.path, result)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        document = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertNotIn('positions', document)
        self.assertNotIn('position_references', document)

    def test_cache_only_keeps_original_measurement_after_request_is_refreshed(self):
        board = self.game.board()
        record = self.session.cache.records(board, 'stockfish')[0]
        changed = dict(record['result'], elapsed_seconds=987)
        self.session.cache.put(board, 'stockfish', record['request'], changed)
        with CachedAnalysisSession(self.game, self.directory) as session:
            result = analyze_game(self.game, session, progress=lambda _: None)
        self.assertEqual(result['positions'], self.analysis['positions'])

    def test_missing_evidence_fails_before_any_engine_can_be_constructed(self):
        reference = self.analysis['position_references'][1]['stockfish']
        (self.directory/'positions'/f'{reference["position"]}.json').unlink()
        with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')):
            with self.assertRaisesRegex(ValueError, 'missing position evidence'):
                CachedAnalysisSession(self.game, self.directory)

    def test_another_game_cannot_silently_search_missing_positions(self):
        game = chess.pgn.read_game(io.StringIO('1. d4 d5 *'))
        with self.assertRaisesRegex(ValueError, 'Missing complete saved'):
            CachedAnalysisSession(game, self.directory, engine_signature={'test': True})

    def test_unplayed_final_board_can_be_unscored_without_fabricating_engine_evidence(self):
        self.analysis['position_references'][-1]['stockfish'] = None
        self.analysis['positions'][-1]['stockfish'] = None
        self.analysis['position_references'][-1]['maia'] = {}
        self.analysis['positions'][-1]['maia'] = {}
        self.store.save(self.path, self.analysis)
        before = {p: p.read_bytes() for p in (self.directory/'positions').glob('*.json')}
        callbacks = []
        with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')):
            with CachedAnalysisSession(self.game, self.directory) as session:
                rebuilt = analyze_game(self.game, session, progress=lambda _: None,
                    on_position=lambda index, position: callbacks.append((index, position)))
        self.assertEqual(rebuilt['moves'], self.analysis['moves'])
        self.assertEqual(rebuilt['performance'], self.analysis['performance'])
        self.assertEqual(rebuilt['accuracy_curve'], self.analysis['accuracy_curve'])
        self.assertIsNone(rebuilt['positions'][-1]['stockfish'])
        self.assertEqual(rebuilt['positions'][-1]['maia'], {})
        self.assertIsNone(rebuilt['position_references'][-1]['stockfish'])
        self.assertIsNone(callbacks[-1][1]['stockfish'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(session.stats['stockfish_calls'], 0)
        self.assertEqual(session.stats['stockfish_cache_hits'], len(rebuilt['moves']))
        self.assertEqual(session.stats['maia_cache_hits'], 21*len(rebuilt['moves']))

    def test_forced_policy_is_derived_from_rules_without_inventing_maia_measurements(self):
        game = chess.pgn.read_game(io.StringIO(
            '[FEN "8/8/8/8/r7/2k5/K7/8 w - - 0 42"]\n\n42. Kb1 *'))
        original = FakeAnalysisSession(game.board().fen())
        self.addCleanup(original._temp.cleanup)
        analysis = analyze_game(game, original, progress=lambda _: None)
        analysis['positions'][0]['maia'] = {}
        analysis['position_references'][0]['maia'] = {}
        store = AnalysisStore(original.cache.directory)
        store.save(original.cache.directory/'analysis.json', analysis)
        with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')), \
             patch.object(PositionCache, 'put', side_effect=AssertionError('cache write')), \
             patch.object(PositionCache, 'put_many', side_effect=AssertionError('cache write')):
            with CachedAnalysisSession(game, original.cache.directory) as session:
                rebuilt = analyze_game(game, session, progress=lambda _: None)
        self.assertEqual(rebuilt['positions'], analysis['positions'])
        self.assertEqual(rebuilt['moves'], analysis['moves'])
        self.assertEqual(rebuilt['accuracy_curve'], analysis['accuracy_curve'])
        self.assertEqual(rebuilt['positions'][0]['maia'], {})
        self.assertEqual(rebuilt['position_references'][0]['maia'], {})
        for record in rebuilt['moves'][0]['maia'].values():
            self.assertEqual(record['moves'][0]['p'], 1.)
            self.assertEqual(record['absolute_deviation'], 0.)

    def test_missing_policy_at_a_played_nonforced_position_cannot_be_inferred(self):
        del self.analysis['position_references'][0]['maia']['maia_kdd_600']
        del self.analysis['positions'][0]['maia']['maia_kdd_600']
        self.store.save(self.path, self.analysis)
        with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')):
            with self.assertRaisesRegex(ValueError, 'Missing saved Maia rating pair 600/600'):
                CachedAnalysisSession(self.game, self.directory)

    def test_missing_non_null_final_measurement_is_an_error(self):
        reference = self.analysis['position_references'][-1]['stockfish']
        (self.directory/'positions'/f'{reference["position"]}.json').unlink()
        with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')):
            with self.assertRaisesRegex(ValueError, 'missing position evidence'):
                CachedAnalysisSession(self.game, self.directory)

    def test_cli_uses_engine_free_session_and_rejects_conflicting_modes(self):
        pgn = self.directory/'game.pgn'
        pgn.write_text(str(self.game), encoding='utf-8')
        args = [str(pgn), '--analysis-only', '--cache-only', '--cache-dir', str(self.directory),
                '--output-dir', str(self.directory/'output')]
        with patch('coach.coach.AnalysisSession', side_effect=AssertionError('engine startup')), \
                patch('analysis.accuracy.figures.export_saved_figures'):
            self.assertEqual(main(args), 0)
            self.assertEqual(main([*args, '--refresh-cache']), 1)
        output = json.loads((self.directory/'output'/'analysis.json').read_text(encoding='utf-8'))
        self.assertEqual(output['moves'], self.analysis['moves'])

    def remove_manifest(self):
        key = self.store._game_key(self.analysis)
        (self.store.game_cache.directory / f'{identity(key)}.json').unlink()

    def test_manifestless_complete_cache_is_discovered_without_engines_or_writes(self):
        self.remove_manifest()
        before = {path: path.read_bytes() for path in self.directory.rglob('*') if path.is_file()}
        with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')):
            session = CachedAnalysisSession(self.game, self.directory, engine_signature={'test': True})
            rebuilt = analyze_game(self.game, session, progress=lambda _: None)
        for field in ('moves', 'positions', 'accuracy_curve', 'performance'):
            self.assertEqual(rebuilt[field], self.analysis[field], field)
        self.assertEqual(before, {path: path.read_bytes() for path in self.directory.rglob('*') if path.is_file()})

    def add_profile(self, signature, *, indices=None):
        board = self.game.board()
        for index in range(len(self.analysis['moves']) + 1):
            if indices is None or index in indices:
                for namespace in ('maia', 'stockfish'):
                    records = self.session.cache.records(board, namespace)
                    for record in records:
                        if record['request'].get('engine') == {'test': True}:
                            request = {**record['request'], 'engine': signature}
                            self.session.cache.put(board, namespace, request, record['result'])
            if index < len(self.analysis['moves']):
                board.push_uci(self.analysis['moves'][index]['played']['move'])

    def test_discovery_prefers_complete_current_profile_and_rejects_ambiguity(self):
        self.remove_manifest()
        alternate = {'test': 'alternate'}
        self.add_profile(alternate)
        current = CachedAnalysisSession(self.game, self.directory, engine_signature={'test': True})
        self.assertEqual(current._references, self.analysis['position_references'])
        with self.assertRaisesRegex(ValueError, 'Ambiguous saved'):
            CachedAnalysisSession(self.game, self.directory, engine_signature={'test': 'unavailable'})

    def test_missing_local_assets_allow_one_coherent_recorded_profile(self):
        self.remove_manifest()
        with patch('engine.runtime.configured_signature', side_effect=FileNotFoundError('removed asset')), \
                patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')):
            cached = CachedAnalysisSession(self.game, self.directory)
        self.assertEqual(cached._references, self.analysis['position_references'])

    def test_discovery_rejects_unknown_engine_records(self):
        self.remove_manifest()
        with patch.object(PositionCache, 'records', autospec=True) as records:
            board = self.game.board()
            # Unknown signatures cannot be promoted to the currently configured engine.
            legal = {move.uci(): 1 / board.legal_moves.count() for move in board.legal_moves}
            records.return_value = [{'request': {'own_rating': 600, 'opponent_rating': 600},
                                     'result': {'policy': legal, 'value': .5}, 'reference': {}}]
            with self.assertRaisesRegex(ValueError, 'Missing complete saved'):
                CachedAnalysisSession(self.game, self.directory, engine_signature={'test': True})

    def test_discovery_never_replaces_an_existing_broken_manifest(self):
        reference = self.analysis['position_references'][1]['stockfish']
        path = self.directory / 'positions' / f'{reference["position"]}.json'
        path.unlink()
        with patch.object(PositionCache, 'records', side_effect=AssertionError('no discovery fallback')):
            with self.assertRaisesRegex(ValueError, 'missing position evidence'):
                CachedAnalysisSession(self.game, self.directory, engine_signature={'test': True})

    def test_malformed_or_null_manifest_cannot_trigger_discovery(self):
        key = self.store._game_key(self.analysis)
        path = self.store.game_cache.directory / f'{identity(key)}.json'
        with patch.object(PositionCache, 'records', side_effect=AssertionError('no discovery fallback')):
            for value in ('{invalid', 'null', '{}', '{"positions":null}', '{"positions":{}}'):
                with self.subTest(value=value):
                    path.write_text(value, encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'Invalid saved evidence manifest'):
                        CachedAnalysisSession(self.game, self.directory, engine_signature={'test': True})

    def test_manifestless_forced_move_and_absent_final_board_need_no_maia(self):
        game = chess.pgn.read_game(io.StringIO('[FEN "8/8/8/8/r7/2k5/K7/8 w - - 0 42"]\n\n42. Kb1 *'))
        original = FakeAnalysisSession(game.board().fen())
        self.addCleanup(original._temp.cleanup)
        analyze_game(game, original, progress=lambda _: None)
        isolated = PositionCache(self.directory / 'forced-only')
        record = original.cache.records(game.board(), 'stockfish')[0]
        isolated.put(game.board(), 'stockfish', record['request'], record['result'])
        with patch('engine.runtime.EngineRuntime.__init__', side_effect=AssertionError('engine startup')):
            cached = CachedAnalysisSession(game, isolated.directory, engine_signature={'test': True})
            rebuilt = analyze_game(game, cached, progress=lambda _: None)
        self.assertEqual(rebuilt['positions'][0]['maia'], {})
        self.assertIsNone(rebuilt['positions'][-1]['stockfish'])
        self.assertEqual(rebuilt['positions'][-1]['maia'], {})
        self.assertEqual(rebuilt['moves'][0]['maia']['600']['moves'][0]['p'], 1.)

    def test_manifestless_evidence_cannot_mix_engine_profiles_across_positions(self):
        isolated = PositionCache(self.directory / 'mixed-engines')
        board = self.game.board()
        for index, move in enumerate(self.game.mainline_moves()):
            for namespace in ('maia', 'stockfish'):
                for record in self.session.cache.records(board, namespace):
                    request = {**record['request'], 'engine': {'test': index}}
                    isolated.put(board, namespace, request, record['result'])
            board.push(move)
        with self.assertRaisesRegex(ValueError, 'No coherent saved maia profile'):
            CachedAnalysisSession(self.game, isolated.directory, engine_signature={'test': True})

    def test_manifestless_evidence_cannot_mix_search_strategies_across_positions(self):
        isolated = PositionCache(self.directory / 'mixed-strategies')
        board = self.game.board()
        for index, move in enumerate(self.game.mainline_moves()):
            for namespace in ('maia', 'stockfish'):
                for record in self.session.cache.records(board, namespace):
                    request, result = record['request'], record['result']
                    if namespace == 'stockfish' and index:
                        request = {**request, 'strategy': 'staged', 'budget_seconds': request['max_budget_seconds']}
                        result = {**result, 'strategy': 'staged'}
                    isolated.put(board, namespace, request, result)
            board.push(move)
        with self.assertRaisesRegex(ValueError, 'No coherent saved stockfish profile'):
            CachedAnalysisSession(self.game, isolated.directory, engine_signature={'test': True})

    def test_two_complete_strategies_for_current_engine_require_explicit_selection(self):
        self.remove_manifest()
        board = self.game.board()
        for move in self.game.mainline_moves():
            for record in self.session.cache.records(board, 'stockfish'):
                request = {**record['request'], 'strategy': 'staged',
                           'budget_seconds': record['request']['max_budget_seconds']}
                self.session.cache.put(board, 'stockfish', request, {**record['result'], 'strategy': 'staged'})
            board.push(move)
        with self.assertRaisesRegex(ValueError, 'Ambiguous saved stockfish profiles'):
            CachedAnalysisSession(self.game, self.directory, engine_signature={'test': True})
