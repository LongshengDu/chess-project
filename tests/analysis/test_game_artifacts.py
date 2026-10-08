"""Public analysis keeps prepared chess data while raw evidence stays cached."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess

from analysis.cache.storage import identity, write_json
from analysis.accuracy.evidence import RATINGS
from analysis.cache.artifacts import AnalysisStore
from analysis.game.history import history_at
from analysis.game.pipeline import ANALYSIS_VERSION
from analysis.maia_context import native_actual_ratings


def sample_analysis(cache):
    candidates = [
        {'move': 'd2d4', 'san': 'd4', 'eval': .3, 'loss': 0., 'maia_p': {'1000': .4, '2600': .5}},
        {'move': 'e2e4', 'san': 'e4', 'eval': .25, 'loss': .05, 'maia_p': {'1000': .4, '2600': .3}},
        {'move': 'a2a3', 'san': 'a3', 'eval': -.2, 'loss': .5, 'maia_p': {'1000': .001, '2600': .0001}},
    ]
    for candidate in candidates:
        probabilities = candidate['maia_p']
        candidate['maia_p'] = {str(rating): probabilities['2600' if rating == 2600 else '1000']
                               for rating in RATINGS}
    rankings = {str(rating): candidates[:2] if rating == 2600 else [candidates[1], candidates[0]]
                for rating in RATINGS}
    maia = {rating: {'moves': [{**{key: c[key] for key in ('move', 'san', 'eval', 'loss')},
                               'p': c['maia_p'][rating]} for c in choices],
                     'expected_accuracy': 80.+.01*(int(rating)-600), 'absolute_deviation': 5.}
            for rating, choices in rankings.items()}
    start = chess.STARTING_FEN
    final = chess.Board(start)
    final.push_uci('a2a3')
    positions, references = [], []
    for ply, board in enumerate((chess.Board(start), final)):
        legal = sorted(move.uci() for move in board.legal_moves)
        raw_maia = {f'maia_kdd_{rating}': {'policy': dict.fromkeys(legal, 1 / len(legal)),
                                    'value': rating / 4000} for rating in RATINGS}
        stockfish = {'cp_vec': dict.fromkeys(legal, 15), 'mate_vec': dict.fromkeys(legal, None),
                     'root_move_depth_vec': dict.fromkeys(legal, 18),
                     'best_move': legal[0], 'engine_moves': legal[:4], 'complete': True, 'coverage_complete': True,
                     'is_checkmate': board.is_checkmate()}
        fields = {'ply': ply, 'fen': board.fen()}
        positions.append({**fields, 'maia': raw_maia, 'stockfish': stockfish})
        references.append({'fields': fields, 'maia': {
            label: cache.put(board, 'maia', {'own_rating': int(label.rsplit('_', 1)[1]),
                                           'opponent_rating': int(label.rsplit('_', 1)[1])}, result)
            for label, result in raw_maia.items()},
            'stockfish': cache.put(board, 'stockfish', {'kind': 'evaluation', 'depth': 18}, stockfish)})
    return {'schema_version': ANALYSIS_VERSION, 'game_id': identity([start, ['a2a3']]),
        'headers': {'White': 'Alice', 'Black': 'Bob', 'WhiteElo': '1400', 'BlackElo': '1500',
                    'Site': 'Chess.com', 'TimeControl': '600+0', 'Result': '*', 'Date': '2026.10.08',
                    'Opening': 'Test opening', 'ECO': 'A00', 'Variation': 'Test variation',
                    'Termination': 'Unfinished', 'WhiteUrl': 'original game context'},
        'start_fen': start,
        'game': {'white': {'name': 'Alice', 'elo': 1450},
                 'black': {'name': 'Bob', 'elo': 1500}, 'rating_scale': 'chess_com_rapid',
                 'result': None, 'date': '2026.10.08',
                 'opening': "Anderssen's Opening", 'eco': 'A00'},
        'coaching': {},
        'moves': [{'ply': 1, 'label': '1. a3', 'side': 'white', 'stage': 'opening', 'fen': start,
                   'position_eval': .3, 'played': {key: candidates[2][key] for key in ('move', 'san', 'eval', 'loss')},
                   'maia': maia, 'candidate_moves': candidates, 'flags': ['inaccuracy'],
                   'accuracy': 88.5, 'centipawn_loss': 50}],
        'positions': positions, 'position_references': references,
        'analysis_execution': {'workers': 4},
        'performance': {'version': 1, 'rating_context_signature': 'internal signature',
                        'method': 'lichess', 'evaluation_source': 'next_position_or_final_played',
                        'initial_cp': 15, 'division': {'middle': None, 'end': None},
                        'players': {'white': {'accuracy': 88.5}}},
        'accuracy_curve': {'schema_version': 1, 'ratings': [600, 2600], 'expected_accuracy': [70., 95.]},
        'agent_run': {'usage': {'total_tokens': 100}}}


class GameArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.store = AnalysisStore(self.directory / 'cache')
        self.path = self.directory / 'output' / 'analysis.json'

    def sample(self):
        return sample_analysis(self.store.positions_cache)

    def test_prepared_moves_are_public_and_engine_measurements_are_referenced_without_copying(self):
        source = self.sample()
        before = copy.deepcopy(source)
        measurements = {path.name: path.read_bytes() for path in self.store.positions_cache.positions_directory.glob('*.json')}
        public = self.store.save(self.path, source)
        self.assertEqual(json.loads(self.path.read_text()), public)
        self.assertEqual(public['moves'], source['moves'])
        self.assertNotIn('positions', public)
        self.assertNotIn('position_references', public)
        self.assertEqual(public['moves'][0]['maia']['1000']['moves'][0]['move'], 'e2e4')
        self.assertIsInstance(public['moves'][0]['maia']['1000']['moves'][0], dict)
        self.assertEqual(self.store.load(self.path), {key: value for key, value in source.items() if key not in ('positions', 'position_references')})
        self.assertEqual(self.store.load_positions(self.path), source['positions'])
        self.assertEqual(self.store.positions_cache.positions_directory.name, 'positions')
        self.assertEqual(len(list(self.store.positions_cache.positions_directory.glob('*.json'))), 2)
        self.assertEqual({path.name: path.read_bytes() for path in self.store.positions_cache.positions_directory.glob('*.json')}, measurements)
        for raw in measurements.values():
            self.assertEqual(set(json.loads(raw)['evidence']), {'maia', 'stockfish'})
        self.assertEqual(self.store.game_cache.get(self.store._game_key(public)),
                         {'positions': source['position_references']})
        self.assertEqual(history_at(self.store.load(self.path), 2), ['a2a3'])
        self.assertEqual(source, before)

    def test_internal_fields_are_removed_and_metadata_cache_has_no_evidence(self):
        source = self.sample()
        public = self.store.save(self.path, source)
        for name in ('schema_version', 'game_id', 'analysis_execution',
                     'agent_run', 'player', 'selected_player', 'rating_scale',
                     'rating_account_overrides', 'rating_scale_override', 'positions', 'position_references'):
            self.assertNotIn(name, public)
        self.assertEqual(public['game'], source['game'])
        self.assertEqual(public['coaching'], source['coaching'])
        self.assertEqual(public['headers'], source['headers'])
        for name in ('version', 'rating_context_signature'):
            self.assertNotIn(name, public['performance'])
        for name in ('method', 'evaluation_source', 'initial_cp', 'division', 'players'):
            self.assertEqual(public['performance'][name], source['performance'][name])
        self.assertEqual(public['accuracy_curve'], {key: value for key, value in source['accuracy_curve'].items()
                                                  if key != 'schema_version'})
        self.assertEqual(self.store.metadata_cache.directory.name, 'game-metadata')
        cached = self.store.metadata_cache.get(self.store.artifact_key(self.path, public))
        self.assertEqual(cached['analysis']['analysis_execution'], source['analysis_execution'])
        self.assertNotIn('moves', cached['analysis'])
        self.assertNotIn('positions', cached['analysis'])
        self.assertNotIn('position_references', cached['analysis'])
        self.assertEqual(cached['evidence']['positions'], source['position_references'])
        self.assertNotIn('players', cached['performance'])
        self.assertNotIn('rating_headers', cached)
        self.assertNotIn('rating_account_overrides', cached['analysis'])

    def test_cache_is_optional_for_reading_prepared_chess_data(self):
        source = self.sample()
        self.store.save(self.path, source)
        unavailable = AnalysisStore(self.directory / 'missing-cache')
        with patch('analysis.game.metadata.opening_name', side_effect=AssertionError('Use saved opening')):
            loaded = unavailable.load(self.path)
        self.assertEqual(loaded['moves'], source['moves'])
        self.assertNotIn('positions', loaded)
        self.assertNotIn('position_references', loaded)
        self.assertIsNone(unavailable.load_positions(self.path))
        self.assertEqual(loaded['schema_version'], ANALYSIS_VERSION)
        self.assertEqual(loaded['game_id'], source['game_id'])
        self.assertEqual(loaded['headers'], source['headers'])
        self.assertEqual(loaded['game'], source['game'])
        self.assertEqual(loaded['coaching'], source['coaching'])
        self.assertEqual(native_actual_ratings(loaded), native_actual_ratings(source))
        self.assertFalse(unavailable.metadata_cache.directory.exists())
        self.assertFalse(unavailable.positions_cache.directory.exists())

    def test_normal_load_does_not_read_position_measurements(self):
        self.store.save(self.path, self.sample())
        with patch.object(self.store.positions_cache, '_read', side_effect=AssertionError('No hydration')):
            loaded = self.store.load(self.path)
        self.assertNotIn('positions', loaded)
        self.assertNotIn('position_references', loaded)

    def test_prepared_only_save_needs_no_raw_evidence_or_references(self):
        source = self.sample()
        source.pop('positions')
        source.pop('position_references')
        other = AnalysisStore(self.directory/'empty-cache')
        public = other.save(self.path, source)
        self.assertEqual(other.load(self.path), source)
        self.assertIsNone(other.load_positions(self.path))
        self.assertFalse(other.positions_cache.positions_directory.exists())
        self.assertNotIn('positions', public)

    def test_raw_positions_without_references_are_not_duplicated_into_cache(self):
        source = self.sample()
        source.pop('position_references')
        other = AnalysisStore(self.directory/'empty-cache')
        public = other.save(self.path, source)
        self.assertNotIn('positions', public)
        self.assertIsNone(other.load_positions(self.path))
        self.assertFalse(other.positions_cache.positions_directory.exists())

    def test_explicit_empty_maia_measurement_map_round_trips_without_fabricating_policies(self):
        source = self.sample()
        source['positions'][-1]['maia'] = {}
        source['position_references'][-1]['maia'] = {}
        self.store.save(self.path, source)
        self.assertEqual(self.store.load_positions(self.path), source['positions'])

    def test_unplayed_final_position_can_explicitly_have_no_stockfish_measurement(self):
        source = self.sample()
        source['positions'][-1]['stockfish'] = None
        source['position_references'][-1]['stockfish'] = None
        self.store.save(self.path, source)
        self.assertEqual(self.store.load_positions(self.path), source['positions'])

    def test_played_position_cannot_have_a_null_stockfish_reference(self):
        source = self.sample()
        source['positions'][0]['stockfish'] = None
        source['position_references'][0]['stockfish'] = None
        with self.assertRaisesRegex(ValueError, 'Invalid game evidence'):
            self.store.save(self.path, source)
        self.assertFalse(self.path.exists())

    def test_pinned_game_measurements_survive_refresh_of_the_same_requests(self):
        source = self.sample()
        self.store.save(self.path, source)
        board = chess.Board(source['start_fen'])
        newer = copy.deepcopy(source['positions'][0]['stockfish'])
        newer['cp_vec']['a2a3'] = -250
        newer['best_move'] = 'e2e4'
        fresh = self.store.positions_cache.put(board, 'stockfish', {'kind': 'evaluation', 'depth': 18}, newer)
        self.assertNotEqual(fresh, source['position_references'][0]['stockfish'])
        self.assertEqual(self.store.load_positions(self.path), source['positions'])
        self.assertEqual(self.store.positions_cache.get_reference(fresh), newer)

    def test_editing_prepared_metadata_keeps_its_evidence_after_another_game_run(self):
        source = self.sample()
        self.store.save(self.path, source)
        newer = copy.deepcopy(source)
        board = chess.Board(source['start_fen'])
        newer['positions'][0]['stockfish']['cp_vec']['a2a3'] = 300
        newer['position_references'][0]['stockfish'] = self.store.positions_cache.put(
            board, 'stockfish', {'kind': 'evaluation', 'depth': 18}, newer['positions'][0]['stockfish'])
        newer['headers']['Event'] = 'Another run'
        self.store.save(self.directory/'newer.json', newer)
        prepared = self.store.load(self.path)
        prepared['headers']['Event'] = 'Edited title'
        self.store.save(self.path, prepared)
        self.assertEqual(self.store.load_positions(self.path), source['positions'])

    def test_references_require_the_exact_history_and_immutable_measurement(self):
        source = self.sample()
        last = source['positions'][1]
        # The FEN is identical, but this cache entry has no originating a3 history.
        unrelated = self.store.positions_cache.put(chess.Board(last['fen']), 'stockfish',
            {'kind': 'evaluation', 'depth': 18}, last['stockfish'])
        for change in ('history', 'unpinned', 'namespace', 'fields', 'missing', 'measurement', 'payload'):
            with self.subTest(change=change):
                invalid = copy.deepcopy(source)
                entry = invalid['position_references'][1]
                if change == 'history':
                    entry['stockfish'] = unrelated
                elif change == 'unpinned':
                    del entry['stockfish']['measurement']
                elif change == 'namespace':
                    entry['stockfish'] = next(iter(entry['maia'].values()))
                elif change == 'fields':
                    entry['fields']['ply'] = 0
                elif change == 'missing':
                    entry['stockfish']['measurement'] = '0'*64
                elif change == 'payload':
                    entry['stockfish']['result'] = last['stockfish']
                else:
                    invalid['positions'][1]['stockfish']['best_move'] = 'e7e5'
                with self.assertRaisesRegex(ValueError, 'reference|measurement'):
                    self.store.save(self.path, invalid)
                self.assertFalse(self.path.exists())
                self.assertIsNone(self.store.game_cache.get(self.store._game_key(source)))

    def test_identical_public_artifacts_keep_separate_measurement_pins(self):
        source = self.sample()
        public = self.store.save(self.path, source)
        newer = copy.deepcopy(source)
        newer['positions'][0]['stockfish']['elapsed_seconds'] = 1234
        newer['position_references'][0]['stockfish'] = self.store.positions_cache.put(
            chess.Board(source['start_fen']), 'stockfish', {'kind': 'evaluation', 'depth': 18},
            newer['positions'][0]['stockfish'])
        other = self.directory / 'newer' / 'analysis.json'
        self.assertEqual(self.store.save(other, newer), public)
        self.assertEqual(self.store.load_positions(self.path), source['positions'])
        self.assertEqual(self.store.load_positions(other), newer['positions'])
        prepared = self.store.load(self.path)
        prepared['headers']['Event'] = 'Edited original'
        self.store.save(self.path, prepared)
        self.assertEqual(self.store.load_positions(self.path), source['positions'])

    def test_maia_labels_require_the_pinned_own_and_opponent_rating_pair(self):
        source = self.sample()
        for mismatch in ('other_rating', 'other_opponent'):
            with self.subTest(mismatch=mismatch):
                invalid = copy.deepcopy(source)
                invalid.pop('positions')
                entry = invalid['position_references'][0]['maia']
                if mismatch == 'other_rating':
                    entry['maia_kdd_600'] = entry['maia_kdd_2600']
                else:
                    entry['maia_kdd_600'] = self.store.positions_cache.put(
                        chess.Board(source['start_fen']), 'maia',
                        {'own_rating': 600, 'opponent_rating': 2600},
                        source['positions'][0]['maia']['maia_kdd_600'])
                with self.assertRaisesRegex(ValueError, 'Invalid game evidence'):
                    self.store.save(self.path, invalid)
        self.assertFalse(self.path.exists())

    def test_relocated_artifact_does_not_silently_adopt_game_manifest(self):
        source = self.sample()
        public = self.store.save(self.path, source)
        relocated = self.directory / 'relocated.json'
        write_json(relocated, public)
        self.assertIsNone(self.store.load_positions(relocated))
        self.store.save(relocated, self.store.load(relocated))
        self.assertIsNone(self.store.load_positions(relocated))
        self.assertEqual(self.store.load_positions(self.path), source['positions'])

    def test_unavailable_pinned_measurement_does_not_substitute_current_evidence(self):
        source = self.sample()
        public = self.store.save(self.path, source)
        reference = source['position_references'][0]['stockfish']
        path = self.store.positions_cache.positions_directory/(reference['position']+'.json')
        document = json.loads(path.read_text())
        from analysis.cache.structure import entries
        for _, _, observation in entries(document):
            if observation['measurement'] == reference['measurement']:
                observation['measurement'] = '0' * 64
        write_json(path, document)
        self.assertIsNone(self.store.load_positions(self.path))
        self.assertEqual(self.store.load(self.path)['moves'], public['moves'])

    def test_missing_non_null_final_reference_is_not_treated_as_an_unscored_position(self):
        source = self.sample()
        self.store.save(self.path, source)
        reference = source['position_references'][-1]['stockfish']
        path = self.store.positions_cache.positions_directory/(reference['position']+'.json')
        path.unlink()
        self.assertIsNone(self.store.load_positions(self.path))

    def test_two_sided_actual_overrides_survive_cache_loss_without_rounding(self):
        source = self.sample()
        source['game']['black']['elo'] = 1590.123456789
        public = self.store.save(self.path, source)
        self.assertEqual(public['headers'], source['headers'])
        self.assertEqual(public['headers']['WhiteElo'], '1400')
        self.assertEqual(public['headers']['BlackElo'], '1500')
        self.assertEqual(public['game']['white']['elo'], 1450)
        self.assertEqual(public['game']['black']['elo'], 1590.123456789)
        self.assertEqual(self.store.load(self.path)['headers'], source['headers'])
        self.assertEqual(self.store.load(self.path)['game'], source['game'])
        loaded = AnalysisStore(self.directory / 'absent').load(self.path)
        self.assertEqual(loaded['headers'], source['headers'])
        self.assertEqual(loaded['game'], source['game'])
        self.assertEqual(loaded['coaching'], {})
        self.assertEqual(native_actual_ratings(loaded), native_actual_ratings(source))

    def test_public_game_metadata_is_authoritative_even_when_headers_and_cache_disagree(self):
        source = self.sample()
        source['game'].update({'result': '1-0', 'date': '2025.01.02',
                               'opening': 'Scandinavian Defense', 'eco': 'B01',
                               'rating_scale': 'lichess_blitz'})
        source['game']['white']['name'] = 'Effective White'
        source['game']['black']['name'] = None
        public = self.store.save(self.path, source)
        metadata = self.store.metadata_cache.get(self.store.artifact_key(self.path, public))
        metadata['analysis'].update({'game': {'white': {'name': 'Wrong', 'elo': 2600}},
                                      'headers': {'Result': '0-1'}})
        self.store.metadata_cache.put(self.store.artifact_key(self.path, public), metadata)
        for store in (self.store, AnalysisStore(self.directory / 'missing')):
            with self.subTest(cache=store.metadata_cache.directory):
                loaded = store.load(self.path)
                self.assertEqual(loaded['game'], source['game'])
                self.assertEqual(loaded['headers'], source['headers'])
                self.assertEqual(native_actual_ratings(loaded), {'White': 1450., 'Black': 1500.})

    def test_unrated_web_analysis_has_explicit_game_context_and_no_coaching_target(self):
        source = self.sample()
        source['game'].update({
            'white': {'name': None, 'elo': None},
            'black': {'name': None, 'elo': None}, 'rating_scale': 'lichess_blitz'})
        for field in ('result', 'date', 'opening', 'eco'):
            source['game'][field] = None
        source['coaching'] = {}
        self.store.save(self.path, source)
        loaded = AnalysisStore(self.directory/'unavailable').load(self.path)
        self.assertEqual(loaded['game'], source['game'])
        self.assertEqual(loaded['coaching'], {})

    def test_current_structure_is_validated_even_when_metadata_exists(self):
        public = self.store.save(self.path, self.sample())
        cases = []
        for key in ('game', 'coaching'):
            document = copy.deepcopy(public)
            del document[key]
            cases.append(document)
        for key, value in (('player', {'white': {'elo': 1450}, 'black': {'elo': 1500}, 'rating_scale': 'cr'}),
                           ('selected_player', {'side': 'white', 'actual_elo': 1450}),
                           ('rating_scale', 'cr'), ('rating_scale_override', 'cr'),
                           ('rating_account_overrides', {'White': 1450})):
            document = copy.deepcopy(public)
            document[key] = value
            cases.append(document)
        invalid = copy.deepcopy(public)
        del invalid['moves'][0]['maia']['1600']['expected_accuracy']
        cases.append(invalid)
        for index, document in enumerate(cases):
            with self.subTest(index=index):
                self.store.metadata_cache.put(self.store.artifact_key(self.path, document), {'analysis': {
                    'schema_version': ANALYSIS_VERSION, 'game_id': 'metadata-is-not-validation'}})
                write_json(self.path, document)
                with self.assertRaisesRegex(ValueError, '--analysis-only'):
                    self.store.load(self.path)

    def test_invalid_context_is_rejected_without_guessing_from_headers(self):
        public = self.store.save(self.path, self.sample())
        for elo in (True, '1500', float('nan'), float('inf'), -1, 5000):
            with self.subTest(elo=elo):
                invalid = copy.deepcopy(public)
                invalid['game']['white']['elo'] = elo
                self.path.write_text(json.dumps(invalid), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, '--analysis-only'):
                    self.store.load(self.path)
        for value in (None, {'side': None}, {'side': 'white'}, {'side': 'black'}):
            invalid = copy.deepcopy(public)
            invalid['coaching'] = value
            write_json(self.path, invalid)
            with self.assertRaisesRegex(ValueError, '--analysis-only'):
                self.store.load(self.path)
        invalid = copy.deepcopy(public)
        invalid['game']['rating_scale'] = 'unknown'
        write_json(self.path, invalid)
        with self.assertRaisesRegex(ValueError, '--analysis-only'):
            self.store.load(self.path)
        invalid = copy.deepcopy(public)
        invalid['game']['side'] = 'white'
        write_json(self.path, invalid)
        with self.assertRaisesRegex(ValueError, '--analysis-only'):
            self.store.load(self.path)

    def test_invalid_or_missing_game_metadata_is_not_reconstructed_from_headers(self):
        public = self.store.save(self.path, self.sample())
        for field in ('result', 'date', 'opening', 'eco'):
            for missing in (False, True):
                with self.subTest(field=field, missing=missing):
                    invalid = copy.deepcopy(public)
                    if missing:
                        del invalid['game'][field]
                    else:
                        invalid['game'][field] = ['not a string']
                    write_json(self.path, invalid)
                    with self.assertRaisesRegex(ValueError, '--analysis-only'):
                        self.store.load(self.path)
        for side in ('white', 'black'):
            for missing in (False, True):
                with self.subTest(side=side, missing=missing):
                    invalid = copy.deepcopy(public)
                    if missing:
                        del invalid['game'][side]['name']
                    else:
                        invalid['game'][side]['name'] = 123
                    write_json(self.path, invalid)
                    with self.assertRaisesRegex(ValueError, '--analysis-only'):
                        self.store.load(self.path)

    def test_unassociated_external_edit_does_not_adopt_latest_game_evidence(self):
        source = self.sample()
        public = self.store.save(self.path, source)
        public['headers']['White'] = 'Edited name'
        public['moves'][0]['maia']['1600']['expected_accuracy'] += 1.
        write_json(self.path, public)
        self.assertIsNone(self.store.load_positions(self.path))
        self.assertNotIn('positions', self.store.load(self.path))

    def test_repeated_save_of_prepared_data_preserves_metadata_and_raw_cache(self):
        source = self.sample()
        public = self.store.save(self.path, source)
        metadata = self.store.metadata_cache.get(self.store.artifact_key(self.path, public))
        for value in (self.store.load(self.path), public, self.store.load(self.path)):
            self.assertEqual(self.store.save(self.path, value), public)
            self.assertEqual(self.store.metadata_cache.get(self.store.artifact_key(self.path, public)), metadata)
            self.assertEqual(self.store.load_positions(self.path), source['positions'])
        self.assertEqual(self.store.load(self.path), {key: value for key, value in source.items() if key not in ('positions', 'position_references')})

    def test_incomplete_or_illegal_documents_do_not_gain_current_schema(self):
        public = self.store.save(self.path, self.sample())
        incomplete = copy.deepcopy(public)
        incomplete.pop('moves')
        write_json(self.path, incomplete)
        with self.assertRaisesRegex(ValueError, 'missing complete'):
            self.store.load(self.path)
        illegal = copy.deepcopy(public)
        illegal['moves'][0]['played']['move'] = 'a2a5'
        write_json(self.path, illegal)
        with self.assertRaisesRegex(ValueError, 'invalid coaching'):
            self.store.load(self.path)

    def test_missing_or_invalid_prepared_statistics_do_not_gain_current_schema(self):
        public = self.store.save(self.path, self.sample())
        for field in ('moves', 'expected_accuracy', 'absolute_deviation'):
            with self.subTest(field=field):
                incomplete = copy.deepcopy(public)
                del incomplete['moves'][0]['maia']['1600'][field]
                write_json(self.path, incomplete)
                with self.assertRaisesRegex(ValueError, 'invalid coaching'):
                    self.store.load(self.path)
        for field in ('expected_accuracy', 'absolute_deviation'):
            for value in (None, True, '99', float('nan'), float('inf'), -1., 101.):
                with self.subTest(field=field, value=value):
                    invalid = copy.deepcopy(public)
                    invalid['moves'][0]['maia']['1600'][field] = value
                    self.path.write_text(json.dumps(invalid), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        self.store.load(self.path)


if __name__ == '__main__':
    unittest.main()

