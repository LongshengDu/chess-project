"""Recalculate a saved game using its original measurements, without any engines."""
from __future__ import annotations

import copy

from analysis.accuracy.evidence import RATINGS
from analysis.cache.storage import identity
from analysis.cache.requests import maia_signature, stockfish_signature, validate_maia_prediction
from analysis.cache.artifacts import AnalysisStore
from analysis.game.cancellation import AnalysisCancelled
from analysis.game.history import replay
from analysis.position_results import stockfish_scan


def discover_game_evidence(game, cache, *, engine_signature=None):
    """Select coherent existing observations when no game manifest has been saved.

    This is original-evidence reconstruction, not a request for today's search
    limits. Every played decision needs complete evidence; only the final board
    and forced Maia policies may be absent. Selection never writes the cache.
    """
    from analysis.cache.policy import quality_key, request_satisfies
    if engine_signature is None:
        from engine.runtime import configured_signature
        try:
            engine_signature = configured_signature()
        except FileNotFoundError:
            # Reconstruction can use one unambiguous recorded profile even
            # after local engine assets have been removed. Never invent identity.
            engine_signature = {}
    board = game.board()
    boards = [board.copy(stack=True)]
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy(stack=True))
    candidates = {'maia': [], 'stockfish': []}
    profile_engines = {namespace: {} for namespace in candidates}
    for board in boards:
        for namespace in candidates:
            profiles = {}
            for record in cache.records(board, namespace):
                request, result = record['request'], record['result']
                engine = request.get('engine')
                if (not isinstance(engine, dict) or not engine
                        or not request_satisfies(request, result, request)):
                    continue
                if namespace == 'maia':
                    rating = request.get('own_rating')
                    if rating not in RATINGS or request.get('opponent_rating') != rating:
                        continue
                    key = rating
                else:
                    # Exploration is usable only after conversion to a complete
                    # evaluation record; PV-only results never score a game.
                    legal = set() if board.is_game_over(claim_draw=False) else {move.uci() for move in board.legal_moves}
                    if (request.get('kind') != 'evaluation' or not result.get('complete')
                            or not result.get('coverage_complete') or set(result.get('cp_vec', {})) != legal):
                        continue
                    key = None
                variable = ({'own_rating', 'opponent_rating'} if namespace == 'maia' else
                            {'depth', 'budget_seconds', 'max_budget_seconds', 'candidate_moves', 'movetime_ms'})
                profile = identity({name: value for name, value in request.items() if name not in variable})
                profile_engines[namespace][profile] = identity(engine)
                profiles.setdefault(profile, {}).setdefault(key, []).append(record)
            candidates[namespace].append(profiles)

    selected = {}
    for namespace, by_position in candidates.items():
        required = [index for index, board in enumerate(boards[:-1])
                    if namespace == 'stockfish' or board.legal_moves.count() != 1]
        keys = set(RATINGS) if namespace == 'maia' else {None}
        common = None
        for index in required:
            complete = {profile for profile, values in by_position[index].items() if keys <= values.keys()}
            if not complete:
                raise ValueError(f'Missing complete saved {namespace} evidence at ply {index + 1}; rebuilding from cache cannot search.')
            common = complete if common is None else common & complete
        expected = identity(maia_signature(engine_signature) if namespace == 'maia' else stockfish_signature(engine_signature))
        if not required:
            available = set().union(*(set(values) for values in by_position))
            matching = {profile for profile in available if profile_engines[namespace][profile] == expected}
            selected[namespace] = (next(iter(matching)) if len(matching) == 1 else
                                   next(iter(available)) if len(available) == 1 else None)
        else:
            matching = {profile for profile in common if profile_engines[namespace][profile] == expected}
            if len(matching) == 1:
                selected[namespace] = next(iter(matching))
            elif len(common) == 1:
                selected[namespace] = next(iter(common))
            elif not common:
                raise ValueError(f'No coherent saved {namespace} profile covers the game; rebuilding from cache cannot mix engine/search profiles.')
            else:
                raise ValueError(f'Ambiguous saved {namespace} profiles; no unique complete profile identifies which observations to use.')

    references, positions = [], []
    for index, board in enumerate(boards):
        policies, policy_refs = {}, {}
        for rating, records in candidates['maia'][index].get(selected['maia'], {}).items():
            record = max(records, key=lambda item: identity(item['reference']))
            label = f'maia_kdd_{rating}'
            policies[label], policy_refs[label] = record['result'], record['reference']
        scores = candidates['stockfish'][index].get(selected['stockfish'], {}).get(None, [])
        score = max(scores, key=lambda item: (quality_key(item['request'], item['result']),
                                              identity(item['reference']))) if scores else None
        fields = {'ply': index, 'fen': board.fen(en_passant='fen')}
        positions.append({**fields, 'maia': policies, 'stockfish': score['result'] if score else None})
        references.append({'fields': fields, 'maia': policy_refs, 'stockfish': score['reference'] if score else None})
    return references, positions


class CachedAnalysisSession:
    """The game-pipeline interface backed exclusively by pinned cached evidence.

    Search settings are deliberately absent: this mode reuses the observations
    actually made for the saved game rather than requesting a different search.
    Missing played-position data is an error, never permission to load an engine
    or infer moves. The final unplayed board may explicitly have no evaluation.
    """

    def __init__(self, game, cache_directory, *, engine_signature=None):
        store = AnalysisStore(cache_directory)
        self.cache = store.positions_cache
        self.start_fen = game.board().fen()
        self._moves = [move.uci() for move in game.mainline_moves()]
        self._references, self._positions = store.game_evidence(game, discover=True, engine_signature=engine_signature)
        self._preflight()
        self.record = None
        self.stats = {'stockfish_calls': 0, 'stockfish_cache_hits': 0,
                      'maia_batches': 0, 'maia_cache_hits': 0}
        self.last_analysis_execution = {'mode': 'cache_only', 'positions': len(self._positions)}

    def _preflight(self):
        """Check the complete game before callers create or replace any outputs."""
        board = self.board([])
        for index, move in enumerate(self._moves):
            position = self._positions[index]
            if board.legal_moves.count() != 1:
                for rating in RATINGS:
                    prediction = position['maia'].get(f'maia_kdd_{rating}')
                    if prediction is None:
                        raise ValueError(f'Missing saved Maia rating pair {rating}/{rating} at ply {index + 1}; rebuilding from cache cannot infer it.')
                    validate_maia_prediction(board, prediction)
            result = position['stockfish']
            if result is None:
                raise ValueError('Rebuilding from cache requires Stockfish evidence for every played position.')
            legal = {candidate.uci() for candidate in board.legal_moves}
            if not result.get('complete') or not result.get('coverage_complete') or set(result.get('cp_vec', {})) != legal:
                raise ValueError(f'Saved Stockfish evidence at ply {index + 1} is incomplete; rebuilding from cache cannot search missing moves.')
            board.push_uci(move)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def board(self, history):
        return replay(self.start_fen, history)

    def _index(self, board):
        history = [move.uci() for move in board.move_stack]
        index = len(history)
        if (board.root().fen() != self.start_fen or index > len(self._moves)
                or history != self._moves[:index]):
            raise ValueError('Rebuilding from cache requires the saved game\'s exact position history.')
        return index

    def human_pair_batches(self, requests):
        batches = []
        for board, ratings, opponents in requests:
            index = self._index(board)
            position = self._positions[index]
            values = []
            for own, other in zip(ratings, opponents, strict=True):
                key = f'maia_kdd_{own}'
                if own != other:
                    raise ValueError(f'Missing saved Maia rating pair {own}/{other}; rebuilding from cache cannot infer it.')
                value = position['maia'].get(key)
                if value is None and index < len(self._moves) and board.legal_moves.count() != 1:
                    raise ValueError(f'Missing saved Maia rating pair {own}/{other}; rebuilding from cache cannot infer it.')
                # A forced move needs no prediction for its deterministic policy.
                # The final board also contributes no played decision. Keep both
                # absences explicit instead of manufacturing engine observations.
                values.append(copy.deepcopy(value))
            batches.append(values)
            self.stats['maia_cache_hits'] += sum(value is not None for value in values)
        return batches

    def initial_analysis(self, history, played, human_candidates):
        board = self.board(history)
        index = self._index(board)
        result = self._positions[index]['stockfish']
        if result is None:
            if index < len(self._moves):
                raise ValueError('Rebuilding from cache requires Stockfish evidence for every played position.')
            return None
        legal = set() if board.is_game_over(claim_draw=False) else {move.uci() for move in board.legal_moves}
        if (not result.get('complete') or not result.get('coverage_complete')
                or set(result.get('cp_vec', {})) != legal):
            raise ValueError(f'Saved Stockfish evidence at ply {index + 1} is incomplete; '
                             'rebuilding from cache cannot search missing moves.')
        self.stats['stockfish_cache_hits'] += 1
        return stockfish_scan(board, copy.deepcopy(result))

    def analyze_positions(self, jobs, *, cancel=None):
        for index, job in enumerate(jobs):
            if cancel is not None and cancel.is_set():
                raise AnalysisCancelled('Game analysis cancelled.')
            yield index, self.initial_analysis(*job)

    def position_references(self, boards, ratings):
        if [self._index(board) for board in boards] != list(range(len(self._positions))):
            raise ValueError('Rebuilding from cache must retain the complete saved game.')
        return copy.deepcopy(self._references)
