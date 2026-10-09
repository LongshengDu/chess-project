"""Offline session and scripted model shared by coach tests."""
import json
import re
import tempfile
import threading
from types import SimpleNamespace
import chess
from analysis.session import AnalysisSession, Limits
from analysis.cache.positions import PositionCache
from analysis.cache.requests import maia_request, stockfish_initial_request
from analysis.position_results import stockfish_result
from analysis.game.history import replay
from analysis.game.cancellation import AnalysisCancelled


class FakeAnalysisSession(AnalysisSession):
    def __init__(self, start_fen=chess.STARTING_FEN, *, cache_directory=None):
        self.start_fen, self.limits, self.stats = start_fen, Limits(), {}
        self.engines = SimpleNamespace(signature={'test': True}, maia_model='fixture',
            stockfish_path='unused', analysis_pool=None, threads_per_worker=1, hash_mb=16)
        self.analysis_workers = 1
        self.human_calls, self.sf_calls, self.pair_calls = [], [], []
        self._temp = tempfile.TemporaryDirectory()
        self.cache = PositionCache(self._temp.name if cache_directory is None else cache_directory)
        self.lock = threading.RLock()
        self._evidence = {}

    def __del__(self):
        self._temp.cleanup()

    def board(self, history):
        return replay(self.start_fen, history)

    def human(self, history, ratings, opponent):
        self.human_calls.append((history.copy(), list(ratings), opponent))
        board = self.board(history)
        legal = sorted(move.uci() for move in board.legal_moves)
        outcome = board.outcome(claim_draw=False)
        values = [({'policy': {}, 'value': .5 if outcome.winner is None else float(outcome.winner)}
                   if outcome else {'policy': {move: 1/len(legal) for move in legal}, 'value': .5}) for r in ratings]
        self._cache_maia(board, ratings, [opponent]*len(ratings), values)
        return dict(zip(map(str, ratings), values, strict=True))

    def human_pairs(self, history, ratings, opponents):
        board = self.board(history)
        outcome = board.outcome(claim_draw=False)
        if outcome:
            values = [{'policy': {}, 'value': .5 if outcome.winner is None else float(outcome.winner)} for _ in ratings]
        else:
            self.pair_calls.append((history.copy(), list(ratings), list(opponents)))
            values = [{'policy': {m.uci(): 1/board.legal_moves.count() for m in board.legal_moves}, 'value': .5} for _ in ratings]
        self._cache_maia(board, ratings, opponents, values)
        return values

    def human_pair_batches(self, requests):
        batches = []
        for board, ratings, opponents in requests:
            values = self.human_pairs([move.uci() for move in board.move_stack], ratings, opponents)
            # Subclasses can supply distinct distributions; pin the values they returned.
            self._cache_maia(board, ratings, opponents, values)
            batches.append(values)
        return batches

    def _cache_maia(self, board, ratings, opponents, values):
        entries = [(maia_request(self.engines.signature, rating, opponent), value)
                   for rating, opponent, value in zip(ratings, opponents, values, strict=True)]
        pending = []
        for request, value in entries:
            reference = self.cache.measurement_reference(board, 'maia', request, value)
            previous = self._evidence.get(reference['context'], {}).get('maia', {}).get(
                (request['own_rating'], request['opponent_rating']))
            if previous != reference:
                pending.append((request, value))
        if pending:
            self.cache.put_many(board, 'maia', pending)
        for request, value in entries:
            self._remember(board, 'maia', request, value)

    def _cache_initial(self, history, played, human_candidates, payload):
        board = self.board(history)
        options = {'forcedCandidateMoves': [played] if played else [],
                   'maiaCandidateMoves': list(human_candidates)[:4]}
        budget = self.limits.verify_ms if self.limits.analysis_strategy == 'bounded' else self.limits.max_ms
        request = stockfish_initial_request(self.engines.signature, self.limits.depth, budget/1000,
            self.limits.max_ms/1000, self.limits.analysis_strategy, options)
        result = stockfish_result(board, payload)
        reference = self.cache.measurement_reference(board, 'stockfish', request, result)
        previous = self._evidence.get(reference['context'], {}).get('stockfish')
        if previous != reference:
            self.cache.put(board, 'stockfish', request, result)
        self._remember(board, 'stockfish', request, result)
        return payload

    def analyze_positions(self, jobs, *, cancel=None):
        for index, job in enumerate(jobs):
            if cancel is not None and cancel.is_set():
                raise AnalysisCancelled('Game analysis cancelled.')
            yield index, self._cache_initial(*job, self.initial_analysis(*job))

    def initial_analysis(self, history, played, human_candidates):
        board = self.board(history)
        if board.is_game_over(claim_draw=False):
            result = {'lines': [], 'best_move': None, 'engine_moves': [], 'search': {
                'complete': True, 'coverage_complete': True, 'depth': 0,
                'terminal_cp': (-10000 if board.turn else 10000) if board.is_checkmate() else 0,
                'terminal_mate': 0 if board.is_checkmate() else None, 'strategy': 'bounded'}}
            return self._cache_initial(history, played, human_candidates, result)
        moves = sorted(move.uci() for move in board.legal_moves)
        result = self.sf(history, 6000, root_moves=moves)
        result.update(best_move=moves[0], engine_moves=moves[:4], search={'strategy': 'bounded'})
        return self._cache_initial(history, played, human_candidates, result)

    def sf(self, history, movetime_ms, multipv=5, root_moves=None, pv_plies=10):
        self.sf_calls.append((history.copy(), movetime_ms, multipv, root_moves))
        board = self.board(history)
        roots = root_moves or sorted(move.uci() for move in board.legal_moves)[:multipv]
        lines = []
        for uci in roots:
            child, pv, sans = board.copy(stack=True), [], []
            for step in range(min(pv_plies, 6)):
                if child.is_game_over(): break
                move = chess.Move.from_uci(uci) if step == 0 else sorted(child.legal_moves, key=lambda m:m.uci())[0]
                sans.append(child.san(move)); pv.append(move.uci()); child.push(move)
            lines.append({'uci': uci, 'san': board.san(chess.Move.from_uci(uci)), 'cp': 20, 'mate': None,
                          'depth': 18, 'pv_uci': pv, 'pv_san': sans, 'fen_after_pv': child.fen()})
        return {'lines': lines, 'evaluation': {'cp':20, 'mate':None}, 'terminal':False}


class ScriptedCodex:
    """Offline Codex-run fixture; never a production model/provider option."""
    def __init__(self):
        self.index = 0

    def report(self, library, brief=False):
        image = next(iter(library.diagrams.paths))
        if brief:
            return f'# Small report\n\n![Position]({image})\n\nActual Elo 1400. The local evidence compares the played move and a legal alternative through their strongest defenses. This fixture cannot establish a strategic advantage. Exercise: compare the resulting piece placement.'
        return f'# Game review\n\n## Performance snapshot\nActual Elo 1400. The short game provides limited evidence for recurring strengths and weaknesses.\n\n## Opening\n![Position]({image})\nInvestigated the played move and an alternative, their best defenses and the stronger human continuations.\n\n## Best decisions\nDeveloping a piece improves its available squares.\n\n## Worst decisions\nNo clear avoidable error is established by this short fixture.\n\n## Main pattern from this game\nThere is insufficient evidence for a recurring weakness.\n\n## Prioritized improvement plan\nReplay the inspected positions.\n\n## Exercises\nCompare the legal development choices in the position above.'

    def __call__(self, library, gate, budget, instructions, task, **kwargs):
        from coach.agent_codex import CodexChessSession
        session = CodexChessSession(library, budget, allow_tools=kwargs['allow_tools'])
        steps = [('get_game_analysis', {}), ('get_position', {'ply': 3}),
                 ('maia_compare', {'ply':3,'ratings':[1400,1600,1800,2000]}),
                 ('compare_played_vs_candidate', {'ply':3,'candidate':'b1c3'}),
                 ('compare_played_vs_candidate', {'ply':1,'candidate':'d2d4'})]
        if kwargs['allow_tools']:
            for name, args in steps[self.index:]:
                self.index += 1
                budget.requests += 1
                budget.record_usage(100, 30, source='codex_fixture')
                result = session.handle_request('item/tool/call', {'tool':name, 'arguments':args})
                if not result['success']:
                    raise ValueError(result['contentItems'][0]['text'])
        for _ in range(gate.max_attempts):
            self.index += 1
            budget.requests += 1
            budget.record_usage(100, 30, source='codex_fixture')
            answer = self.report(library, brief=not kwargs['allow_tools'])
            try:
                gate.validate(answer)
                return answer
            except ValueError:
                gate.check_retry()

    def run(self, *args, **kwargs):
        from unittest.mock import patch
        from coach.agent_runner import run_coach
        with patch('coach.agent_codex.run_codex', side_effect=self):
            return run_coach(*args, **kwargs)
