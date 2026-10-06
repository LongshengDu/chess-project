"""Offline engines and scripted model shared by coach tests."""
import json
import re
import tempfile
import chess
from analysis.engine_session import Limits
from analysis.cache import JsonCache
from analysis.game.history import replay


class FakeEngines:
    def __init__(self, start_fen=chess.STARTING_FEN):
        self.start_fen, self.limits, self.signature, self.stats = start_fen, Limits(), {'test': True}, {}
        self.human_calls, self.sf_calls, self.pair_calls = [], [], []
        self._temp = tempfile.TemporaryDirectory()
        self.cache = JsonCache(self._temp.name)

    def __del__(self):
        self._temp.cleanup()

    def board(self, history):
        return replay(self.start_fen, history)

    def human(self, history, ratings, opponent):
        self.human_calls.append((history.copy(), list(ratings), opponent))
        board = self.board(history)
        legal = sorted(move.uci() for move in board.legal_moves)
        return {str(r): {'policy': {move: 1/len(legal) for move in legal}, 'value': .5} for r in ratings}

    def human_pairs(self, history, ratings, opponents):
        self.pair_calls.append((history.copy(), list(ratings), list(opponents)))
        board = self.board(history)
        return [{'policy': {m.uci(): 1/board.legal_moves.count() for m in board.legal_moves}, 'value': .5} for _ in ratings]

    def initial_analysis(self, history, played, human_candidates):
        board = self.board(history)
        if board.is_game_over(claim_draw=False):
            return {'lines': [], 'best_move': None, 'engine_moves': [], 'search': {
                'complete': True, 'coverage_complete': True, 'depth': 0,
                'terminal_cp': (-10000 if board.turn else 10000) if board.is_checkmate() else 0,
                'terminal_mate': 0 if board.is_checkmate() else None, 'strategy': 'bounded'}}
        moves = sorted(move.uci() for move in board.legal_moves)
        result = self.sf(history, 6000, root_moves=moves)
        result.update(best_move=moves[0], engine_moves=moves[:4], search={'strategy': 'bounded'})
        return result

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
        return f'# Game review\n\n## Performance snapshot\nActual Elo 1400. The short game gives low confidence in fitted performance.\n\n## Opening\n![Position]({image})\nInvestigated the played move and an alternative, their best defenses and the stronger human continuations.\n\n## Best decisions\nDeveloping a piece improves its available squares.\n\n## Worst decisions\nNo clear avoidable error is established by this short fixture.\n\n## Main pattern from this game\nThere is insufficient evidence for a recurring weakness.\n\n## Prioritized improvement plan\nReplay the inspected positions.\n\n## Exercises\nCompare the legal development choices in the position above.'

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
