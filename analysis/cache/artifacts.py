"""Publish complete coaching evidence without internal persistence metadata."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import chess

from analysis.cache.storage import JsonCache, identity, write_json
from analysis.cache.positions import PositionCache
from analysis.settings import CONFIG

INTERNAL_FIELDS = ('schema_version', 'game_id', 'analysis_execution', 'agent_run')
PERFORMANCE_INTERNAL = ('version', 'rating_context_signature')


class AnalysisStore:
    """Publish prepared analysis and retain references to canonical engine measurements."""

    def __init__(self, cache_directory=None):
        directory = CONFIG['ANALYSIS']['CACHE_DIR'] if cache_directory is None else cache_directory
        self.metadata_cache = JsonCache(Path(directory) / 'game-metadata')
        self.positions_cache = PositionCache(directory)
        self.game_cache = JsonCache(Path(directory) / 'games')

    def save(self, path, analysis):
        """Preserve prepared move evidence while separating raw positions and metadata."""
        document = copy.deepcopy(analysis)
        positions = document.pop('positions', None)
        references = document.pop('position_references', None)
        metadata = {'analysis': {name: document.pop(name) for name in INTERNAL_FIELDS if name in document}}
        performance = document.get('performance', {})
        metadata['performance'] = {name: performance.pop(name) for name in PERFORMANCE_INTERNAL if name in performance}
        curve = document.get('accuracy_curve', {})
        if 'schema_version' in curve:
            metadata['accuracy_curve'] = {'schema_version': curve.pop('schema_version')}

        key = self.artifact_key(path, document)
        if references is not None:
            measured = self._read_positions(document, references)
            if measured is None:
                raise ValueError('Game evidence references a missing cached measurement.')
            if positions is not None and measured != positions:
                raise ValueError('Game evidence references differ from the supplied position measurements.')
            metadata['evidence'] = {'positions': references}
            self.game_cache.put(self._game_key(document), metadata['evidence'])
        else:
            # Keep only the evidence associated with this exact artifact. Two
            # paths can contain identical prepared JSON but pin different runs.
            evidence = (self.metadata_cache.get(key) or {}).get('evidence')
            if not evidence and Path(path).is_file():
                previous = json.loads(Path(path).read_text(encoding='utf-8'))
                if self._game_key(previous) == self._game_key(document):
                    evidence = (self.metadata_cache.get(self.artifact_key(path, previous)) or {}).get('evidence')
            metadata['evidence'] = evidence or {}
        existing = self.metadata_cache.get(key)
        if isinstance(existing, dict):
            for section, fields in (('analysis', INTERNAL_FIELDS),
                                    ('performance', PERFORMANCE_INTERNAL),
                                    ('accuracy_curve', ('schema_version',))):
                stored = {name: value for name, value in existing.get(section, {}).items()
                          if name in fields}
                metadata[section] = {**stored, **metadata.get(section, {})}
        if any(metadata.values()):
            self.metadata_cache.put(key, metadata)
        write_json(path, document)
        return document

    def load(self, path):
        """Load prepared analysis and optional bookkeeping without hydrating raw positions."""
        document = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(document, dict):
            raise ValueError('Saved game analysis must be an object; regenerate it with --analysis-only.')
        self._validate(document)
        metadata = self.metadata_cache.get(self.artifact_key(path, document))
        analysis = dict(document)
        analysis.pop('positions', None)
        analysis.pop('position_references', None)
        if isinstance(metadata, dict):
            analysis.update({name: value for name, value in metadata.get('analysis', {}).items()
                             if name in INTERNAL_FIELDS})
            for section, fields in (('performance', PERFORMANCE_INTERNAL),
                                    ('accuracy_curve', ('schema_version',))):
                if section in analysis:
                    analysis[section] = {**analysis[section], **{
                        name: value for name, value in metadata.get(section, {}).items() if name in fields}}
        from analysis.game.pipeline import ANALYSIS_VERSION
        analysis.setdefault('schema_version', ANALYSIS_VERSION)
        analysis['game_id'] = identity(self._game_key(analysis))
        return analysis

    def load_positions(self, path):
        """Explicit raw-evidence access for audits/recalculation, separate from coaching."""
        document = json.loads(Path(path).read_text(encoding='utf-8'))
        key = self.artifact_key(path, document)
        metadata = self.metadata_cache.get(key) or {}
        manifest = metadata.get('evidence') or {}
        refs = manifest.get('positions')
        if refs is not None:
            return self._read_positions(document, refs)
        return None

    def game_evidence(self, game, *, discover=False, engine_signature=None):
        """Resolve an existing game's pinned observations without engine startup."""
        document = {'start_fen': game.board().fen(), 'moves': [
            {'played': {'move': move.uci()}} for move in game.mainline_moves()]}
        key = self._game_key(document)
        manifest = self.game_cache.get(key)
        manifest_path = self.game_cache.directory / f'{identity(key)}.json'
        if manifest is None and not manifest_path.exists():
            if discover:
                from analysis.cache.session import discover_game_evidence
                return discover_game_evidence(game, self.positions_cache, engine_signature=engine_signature)
            raise ValueError('No saved evidence manifest for this game; rebuilding from cache cannot search for missing evidence.')
        if not isinstance(manifest, dict) or not isinstance(manifest.get('positions'), list):
            raise ValueError('Invalid saved evidence manifest; rebuilding from cache cannot replace its original observations.')
        references = manifest['positions']
        positions = self._read_positions(document, references)
        if positions is None:
            raise ValueError('Saved game references missing position evidence; rebuilding from cache cannot run engines.')
        return references, positions

    @staticmethod
    def artifact_key(path, document):
        """Private artifact association; copying public JSON does not copy its pins."""
        return {'artifact': str(Path(path).resolve()), 'content': identity(document)}

    @staticmethod
    def _game_key(document):
        return [document['start_fen'], [row['played']['move'] for row in document['moves']]]

    def _read_positions(self, document, references):
        """Resolve pinned measurements only when every address matches its game history."""
        moves = document['moves']
        if not isinstance(references, list) or len(references) != len(moves)+1:
            raise ValueError('Game evidence references must cover every played board and the final board.')
        board, positions = chess.Board(document['start_fen']), []
        for index, entry in enumerate(references):
            try:
                fields, maia = entry['fields'], entry['maia']
                if (set(entry) != {'fields', 'maia', 'stockfish'}
                        or set(fields) != {'fen', 'ply'} or type(fields['ply']) is not int
                        or fields['ply'] != index or chess.Board(fields['fen']).fen() != board.fen()
                        or not isinstance(maia, dict)
                        or any(not isinstance(label, str) or not label for label in maia)):
                    raise ValueError('Invalid position fields or Maia references.')
                stockfish = entry['stockfish']
                if stockfish is None and index < len(moves):
                    raise ValueError('Every played position requires a Stockfish measurement.')
                refs = [*maia.values(), *([stockfish] if stockfish is not None else [])]
                namespaces = ['maia'] * len(maia) + (['stockfish'] if stockfish is not None else [])
                for namespace, reference in zip(namespaces, refs, strict=True):
                    if (not isinstance(reference, dict)
                            or set(reference) != {'position', 'context', 'namespace', 'request', 'measurement'}
                            or not reference.get('measurement')
                            or reference.get('namespace') != namespace):
                        raise ValueError('Reference does not match the complete position history.')
                records = self.positions_cache.get_reference_records(refs)
                if any(record is None for record in records):
                    return None
                for label, record in zip(maia, records[:len(maia)], strict=True):
                    request = record['request']
                    rating = request.get('own_rating') if isinstance(request, dict) else None
                    if (type(rating) is not int or request.get('opponent_rating') != rating
                            or label != f'maia_kdd_{rating}'):
                        raise ValueError('Maia label does not match its pinned rating pair.')
                results = [record['result'] for record in records]
                if not self.positions_cache.references_match(board, refs):
                    raise ValueError('Reference does not match the complete position history.')
                measured = results[-1] if stockfish is not None else None
                if measured is not None:
                    legal = set() if board.is_game_over(claim_draw=False) else {m.uci() for m in board.legal_moves}
                    if (not isinstance(measured, dict) or not measured.get('complete')
                            or not measured.get('coverage_complete') or set(measured.get('cp_vec', {})) != legal):
                        raise ValueError('Cached Stockfish evaluations must cover every legal move.')
                positions.append({**fields, 'maia': dict(zip(maia, results[:len(maia)], strict=True)),
                                  'stockfish': measured})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Invalid game evidence reference or position history.') from exc
            if index < len(moves):
                board.push_uci(moves[index]['played']['move'])
        return positions

    @staticmethod
    def _validate(analysis):
        """Require complete current prepared evidence, independently of optional metadata."""
        from analysis.game.history import replay
        from analysis.game.metadata import validate_game_metadata
        rows = analysis.get('moves')
        if (not isinstance(rows, list)
                or not isinstance(analysis.get('performance'), dict)
                or not isinstance(analysis.get('accuracy_curve'), dict)):
            raise ValueError('Saved game analysis is missing complete coaching evidence; regenerate it with --analysis-only.')
        try:
            if any(name in analysis for name in ('player', 'selected_player', 'rating_scale',
                                                 'rating_scale_override', 'rating_account_overrides')):
                raise ValueError('Unsupported game context.')
            if not isinstance(analysis['headers'], dict):
                raise ValueError('Invalid PGN headers.')
            validate_game_metadata(analysis['game'])
            coaching = analysis['coaching']
            if not isinstance(coaching, dict) or 'side' in coaching:
                raise ValueError('Saved coaching data must be independent of the report target.')
            moves = [row['played']['move'] for row in rows]
            start = analysis['start_fen']
            replay(start, moves)
            from analysis.accuracy.evidence import RATINGS
            for row in rows:
                if not isinstance(row['maia'], dict) or not isinstance(row['candidate_moves'], list):
                    raise ValueError('Invalid candidate evidence.')
                for rating in RATINGS:
                    record = row['maia'][str(rating)]
                    if not isinstance(record, dict) or not isinstance(record['moves'], list):
                        raise ValueError('Invalid Maia moves.')
                    for field in ('expected_accuracy', 'absolute_deviation'):
                        value = record[field]
                        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                            raise ValueError('Invalid Maia accuracy.')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Saved game analysis contains invalid coaching evidence; regenerate it with --analysis-only.') from exc
