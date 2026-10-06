"""Stateful chess investigations, reusable evidence and bounded tool execution."""
from __future__ import annotations

import json
import time
from pathlib import Path

import chess

from analysis.game.context import leadup_context
from analysis.position_evaluation import PIECE_VALUES, clip_elo, eval_value
from analysis.player_rating.scale import native_player_rating
from analysis.game.summary import compact_summary
from analysis.game.history import history_at
from analysis.cache import identity
from .agent_budget import CoachingLimitError
from .report_diagrams import BoardDiagrams
from .tools_evidence import ToolEvidence, compact_evidence
from .agent_progress import CoachProgress, tool_activity, tool_summary
from .tools_schema import chess_tools, validate_tool_arguments


def checking_lines(board):
    """Exact checker-to-king paths, so explanations can cite verified squares."""
    king = board.king(board.turn)
    if king is None:
        return []
    return [{'piece': chess.piece_name(board.piece_type_at(square)),
             'squares': [chess.square_name(sq) for sq in [square, *sorted(
                 chess.SquareSet(chess.between(square, king)), key=lambda sq: chess.square_distance(square, sq)), king]]}
            for square in board.checkers()]


def _maia_moves(board, policy, topk):
    moves = sorted(policy, key=lambda uci: (-policy[uci], uci))[:topk]
    return [{'move': uci, 'san': board.san(chess.Move.from_uci(uci)), 'p': policy[uci]} for uci in moves]


class ChessTools:
    def __init__(self, analysis, engines, output_dir, *, max_calls=None, progress=None):
        self.analysis, self.engines, self.directory = analysis, engines, Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace = self.directory / 'agent_trace.jsonl'
        self.trace.write_text('', encoding='utf-8')
        self.rows = {tuple(history_at(analysis, row['ply'])): row for row in analysis['moves']}
        self.results, self.invocations, self.investigated = {}, [], set()
        self.diagrams = BoardDiagrams(analysis, engines.board, self.directory)
        self.tools = chess_tools(self)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.request_cache = {}
        self.sent_baselines = set()
        self.sent_human_replies = set()
        self.max_calls = max_calls
        self.check_budget = lambda: None
        self.phase = 'followup'
        self.progress = progress if progress is not None else CoachProgress()

    def get_leadup(self, ply, lookback_plies=8):
        return leadup_context(self.analysis, [ply], lookback_plies)

    def investigate_batch(self, requests):
        if not isinstance(requests, list) or not 1 <= len(requests) <= 6:
            raise ValueError('Supply 1–6 independent investigations.')
        available = {name: tool for name, tool in self.tools_by_name.items() if name != 'investigate_batch'}
        for item in requests:
            if not isinstance(item, dict) or set(item) != {'tool', 'arguments'} \
                    or item['tool'] not in available or not isinstance(item['arguments'], dict):
                raise ValueError('Each investigation needs a valid tool and arguments; nested batches are not allowed.')
            validate_tool_arguments(available[item['tool']], item['arguments'])
        if self.max_calls is not None and len(self.invocations)+len(requests) > self.max_calls:
            raise CoachingLimitError('Batch exceeds remaining local tool-call budget.')
        results = []
        self.progress.emit(f'Checking {len(requests)} independent chess questions.')
        # One LLM round trip; engines execute serially to preserve UCI state.
        for index, item in enumerate(requests):
            try:
                result = self.call(item['tool'], item['arguments'])
                results.append({'index': index, 'tool': item['tool'], 'result': result})
            except CoachingLimitError:
                raise
            except (ValueError, TypeError) as exc:
                results.append({'index': index, 'tool': item['tool'], 'error': str(exc)[:300]})
        return ToolEvidence(results=results)

    def call(self, name, arguments):
        self.check_budget()
        if name not in self.tools_by_name:
            raise ValueError('Unknown local chess tool.')
        if name == 'investigate_batch':
            return self.investigate_batch(**arguments)
        if self.max_calls is not None and len(self.invocations) >= self.max_calls:
            raise CoachingLimitError('Local tool-call budget reached. Finish the report using the evidence already returned.')
        tick, result_id, status = time.perf_counter(), None, 'ok'
        response_chars = 0
        self.progress.stage(tool_activity(name, arguments, self.analysis))
        try:
            request_id = identity([name, arguments])
            result_id = self.request_cache.get(request_id)
            cached = result_id is not None
            if result_id is None:
                result = getattr(self, name)(**arguments)
                result_id = identity([name, arguments, result])[:20]
                self.results[result_id] = result
                self.request_cache[request_id] = result_id
            else:
                result = self.results[result_id]
            selected_side = self.analysis['selected_player']['side']
            level = (native_player_rating(self.analysis, selected_side)
                     or native_player_rating(self.analysis, selected_side, fitted=True) or 1500)
            evidence = ToolEvidence(**{'result_id': result_id, 'maia_rating_scale': 'Lichess Blitz',
                                      **compact_evidence(result, level)})
            for branch in (evidence, evidence.get('played', {}), evidence.get('candidate', {})):
                human = branch.get('human_replies')
                if human:
                    reply_id = identity(human)[:16]
                    if reply_id in self.sent_human_replies:
                        branch['human_replies'] = {'reply_ref': reply_id}
                    else:
                        branch['human_replies'] = {'reply_id': reply_id, **human}
                        self.sent_human_replies.add(reply_id)
            baseline = evidence.get('baseline')
            if baseline:
                if self.phase == 'initial':
                    # Curves already contain rating-specific evidence; avoid
                    # duplicating evals in another table in the initial package.
                    baseline.pop('maia', None)
                if baseline['ply'] in self.sent_baselines:
                    evidence['baseline_ref'] = baseline['ply']
                    del evidence['baseline']
                else:
                    self.sent_baselines.add(baseline['ply'])
            response_chars = len(str(evidence))
            self.progress.emit(('Reused saved evidence. ' if cached else '') + tool_summary(name, result)
                               + f' ({time.perf_counter()-tick:.1f}s)')
            return evidence
        except Exception:
            status = 'error'
            self.progress.emit('Investigation could not complete; the coach will receive the tool error.')
            raise
        finally:
            self.progress.idle()
            event = {'tool': name, 'arguments': arguments, 'result_id': result_id,
                     'phase': self.phase,
                     'elapsed_seconds': round(time.perf_counter()-tick, 4), 'status': status, 'response_chars': response_chars}
            self.invocations.append(event)
            with self.trace.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def get_game_analysis(self):
        return compact_summary(self.analysis)

    def get_position(self, ply, line=None):
        history = history_at(self.analysis, ply, line)
        board = self.engines.board(history)
        baseline = self.rows.get(tuple(history))
        return {'position': {'ply': ply, 'line': line or []}, 'fen': board.fen(),
            'turn': 'white' if board.turn else 'black',
            'legal_moves': {m.uci(): board.san(m) for m in board.legal_moves},
            'material': {side: {chess.piece_name(p): len(board.pieces(p, color)) for p in PIECE_VALUES if p != chess.KING}
                         for side, color in (('white', chess.WHITE), ('black', chess.BLACK))},
            'in_check': board.is_check(), 'checkmate': board.is_checkmate(), 'checking_lines': checking_lines(board),
            'terminal': board.is_game_over(claim_draw=False),
            'baseline': baseline, 'diagram': self.diagrams.render(history, [baseline['played']['move']] if baseline else [], played=True)}

    def opponent_elo(self, history):
        other = 'black' if self.engines.board(history).turn else 'white'
        return clip_elo(native_player_rating(self.analysis, other, fitted=True) or 1500)

    def maia_analyze(self, ply, player_elo, opponent_elo, line=None, topk=3):
        history = history_at(self.analysis, ply, line)
        board = self.engines.board(history)
        topk = 3 if topk is None else topk
        if type(topk) is not int or not 1 <= topk <= 5:
            raise ValueError('topk must be 1–5.')
        policy = self.engines.human(history, [player_elo], opponent_elo)[str(player_elo)]['policy']
        return {'position': {'ply': ply, 'line': line or []}, 'opponent_elo': opponent_elo,
                'maia': {str(player_elo): _maia_moves(board, policy, topk)}}

    def maia_compare(self, ply, ratings, line=None, topk=3):
        history = history_at(self.analysis, ply, line)
        topk = 3 if topk is None else topk
        if not 1 <= len(ratings) <= 6 or type(topk) is not int or not 1 <= topk <= 5:
            raise ValueError('Use 1–6 ratings and topk 1–5.')
        baseline = self.rows.get(tuple(history))
        if baseline and all(str(r) in baseline['maia'] for r in ratings):
            return {'position': {'ply': ply, 'line': line or []},
                    'conditioning': 'equal_rating',
                    'maia': {str(r): baseline['maia'][str(r)][:topk] for r in ratings}}
        opponent = self.opponent_elo(history)
        policies = self.engines.human(history, ratings, opponent)
        board = self.engines.board(history)
        return {'position': {'ply': ply, 'line': line or []}, 'conditioning': 'fixed_opponent_elo', 'opponent_elo': opponent,
                'maia': {str(r): _maia_moves(board, policies[str(r)]['policy'], topk) for r in ratings}}

    def scored_lines(self, history, result):
        board = self.engines.board(history)
        if result.get('terminal'):
            return {'terminal': True, 'result': result['result'], 'eval': eval_value(result['evaluation']), 'lines': []}
        return {'lines': [{'move': item['uci'], 'san': item['san'], 'eval': eval_value(item), 'depth': item['depth'],
                'line': item['pv_uci'], 'san_line': board.variation_san([chess.Move.from_uci(m) for m in item['pv_uci']])}
                for item in result['lines']]}

    def stockfish_analyze(self, ply, movetime_ms, line=None, multipv=2, root_moves=None, pv_plies=8):
        history = history_at(self.analysis, ply, line)
        if not 1 <= (multipv or 2) <= 5 or not 1 <= (pv_plies or 8) <= 16:
            raise ValueError('Use MultiPV 1–5 and PV length 1–16.')
        result = self.engines.sf(history, movetime_ms, multipv or 2, root_moves, pv_plies or 8)
        return {'position': {'ply': ply, 'line': line or []}, **self.scored_lines(history, result)}

    def human_replies(self, ply, line, engine_reply, movetime_ms):
        """Bounded immediate reply evidence; no recursive tree or LLM calls.

        Use equal-rating conditioning, matching analysis.json's Maia curves,
        for both actual-game and hypothetical positions. Preserve all sampled
        ratings' first choices, then fill to five distinct replies. Omitted
        probability remains explicit; these are not calibrated win chances.
        """
        history = history_at(self.analysis, ply, line)
        board = self.engines.board(history)
        if board.is_game_over(claim_draw=False):
            return None
        side = 'white' if board.turn else 'black'
        fitted = native_player_rating(self.analysis, side, fitted=True)
        actual = native_player_rating(self.analysis, side)
        reference = max(1000, min(2600, round((actual or fitted or 1500)/100)*100))
        ratings = sorted({reference, max(1000, min(2600, round((fitted or reference)/100)*100)), 2000, 2200, 2600})
        baseline = self.rows.get(tuple(history))
        if baseline:
            curves = {c['move']: c['maia_p'] for c in baseline['candidate_moves']}
            ranked = {r: [c['move'] for c in baseline['maia'][str(r)]] for r in ratings}
        else:
            policies = self.engines.human_pairs(history, ratings, ratings)
            curves = {m.uci(): {str(r): policy['policy'][m.uci()] for r, policy in zip(ratings, policies, strict=True)}
                      for m in board.legal_moves}
            ranked = {r: sorted(curves, key=lambda m: (-curves[m][str(r)], m)) for r in ratings}
        choices = {ranked[r][0] for r in ratings}
        for move in [*ranked[reference][:3], *ranked[2600][:3]]:
            if len(choices) >= 5:
                break
            choices.add(move)
        choices = sorted(choices, key=lambda m: (-curves[m][str(reference)], m))
        evaluations = {c['move']: c['eval'] for c in baseline['candidate_moves']} if baseline else {}
        missing = [m for m in choices if m not in evaluations]
        if missing:
            self.check_budget()
            checked = self.engines.sf(history, movetime_ms, multipv=len(missing), root_moves=missing, pv_plies=2)
            evaluations.update({c['uci']: eval_value(c) for c in checked['lines']})
        replies = []
        for uci in choices:
            move = chess.Move.from_uci(uci)
            child = board.copy(stack=False); child.push(move)
            mates = []
            for response in child.legal_moves:
                child.push(response)
                mate = child.is_checkmate()
                child.pop()
                if mate:
                    mates.append({'move': response.uci(), 'san': child.san(response)})
            value = evaluations.get(uci)  # Missing short-search scores stay unknown.
            if child.is_checkmate():
                value = '#-0' if child.turn else '#0'
            elif child.is_game_over(claim_draw=False):
                value = 0.0
            elif mates:
                value = '#1' if child.turn else '#-1'
            replies.append({'move': uci, 'san': board.san(move),
                'maia_p': {str(r): curves[uci][str(r)] for r in ratings}, 'eval': value,
                **({'allows_mate_in_one': mates} if mates else {})})
        return {'position': {'ply': ply, 'line': line}, 'side': side, 'reference_elo': reference,
            'conditioning': 'equal_rating', 'moves': replies,
            'covered_probability': {str(r): min(1., sum(curves[m][str(r)] for m in choices)) for r in ratings},
            'engine_reply_p': {str(r): curves.get(engine_reply, {}).get(str(r)) for r in ratings}}

    def explore_candidate(self, ply, candidate, line=None, stockfish_ms=None, maia_elos=None, max_plies=8):
        history = history_at(self.analysis, ply, line)
        max_plies = 8 if max_plies is None else max_plies
        if type(max_plies) is not int or not 2 <= max_plies <= 16:
            raise ValueError('Branch length must be 2–16 plies.')
        ms = self.engines.limits.verify_ms if stockfish_ms is None else stockfish_ms
        selected_side = self.analysis['selected_player']['side']
        level = (native_player_rating(self.analysis, selected_side)
                 or native_player_rating(self.analysis, selected_side, fitted=True) or 1500)
        ratings = sorted(set(clip_elo(level+step) for step in (0,200,400,600))) if maia_elos is None else maia_elos
        root = self.engines.sf(history, ms, multipv=1, root_moves=[candidate], pv_plies=max_plies)
        continuation = root['lines'][0]['pv_uci']
        after_candidate = history + [candidate]
        if len(continuation) < 2 and not self.engines.board(after_candidate).is_game_over():
            defense = self.engines.sf(after_candidate, ms, multipv=1, pv_plies=max_plies-1)['lines'][0]
            continuation = [candidate] + defense['pv_uci']
        branch = (line or []) + continuation[:2]
        after_history = history_at(self.analysis, ply, branch)
        after = self.engines.board(after_history)
        human = self.maia_compare(ply, ratings, branch) if not after.is_game_over() else None
        baseline = self.rows.get(tuple(history))
        if baseline:
            self.investigated.add(baseline['ply'])
        return {'position': {'ply': ply, 'line': line or []},
                'human_replies': self.human_replies(ply, (line or [])+[candidate],
                    continuation[1] if len(continuation) >= 2 else None, ms),
                'stockfish': self.scored_lines(history, root)['lines'][0],
                'best_defense': continuation[1] if len(continuation) >= 2 else None,
                'after_defense': {'ply': ply, 'line': branch}, 'fen_after_defense': after.fen(),
                'after_defense_diagram': self.diagrams.render(after_history),
                'terminal': after.is_game_over(claim_draw=False), 'checkmate': after.is_checkmate(),
                'checking_lines': checking_lines(after), 'maia_after_defense': human['maia'] if human else None,
                'maia_after_defense_context': {k: human[k] for k in ('conditioning', 'opponent_elo') if k in human} if human else None}

    def compare_played_vs_candidate(self, ply, candidate):
        history = history_at(self.analysis, ply)
        if ply > len(self.analysis['moves']):
            raise ValueError('Comparison requires a played game move.')
        row = self.analysis['moves'][ply-1]
        played = row['played']['move']
        if candidate == played:
            raise ValueError('Choose an alternative to the played move.')
        return {'ply': ply, 'label': row['label'], 'stage': row['stage'], 'fen': row['fen'],
                'baseline': row, 'played': self.explore_candidate(ply, played),
                'candidate': self.explore_candidate(ply, candidate),
                'actual_continuation': [r['label'] for r in self.analysis['moves'][ply-1:ply+5]],
                'comparison_diagram': self.diagrams.render(history, [candidate, played])}
