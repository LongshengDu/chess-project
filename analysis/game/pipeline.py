"""Whole-game orchestration over common engine and rating services."""
from __future__ import annotations

import time

import chess
import chess.pgn

from analysis.cache import identity
from analysis.game.cancellation import AnalysisCancelled
from analysis.position_evaluation import RATINGS, header_elo, eval_value, eval_loss, centipawns
from analysis.move_hints import add_flags
from analysis.game.performance import refresh_performance
from analysis.position_results import stockfish_result
from analysis.settings import CONFIG

from analysis.player_rating.policies import prepare_rating_policies
from analysis.player_rating.parameters import RATINGS as FIT_RATINGS
from analysis.player_rating.service import fit_game, get_estimator, store_elo_fit
from analysis.player_rating.scale import rating_context, native_actual_ratings

ANALYSIS_VERSION = 4


def prepare_game_policies(game, records, engines, *, predictions=None):
    """Prepare common Maia evidence and the policies used by move displays."""
    fit_records = [{'move': row['played']['move'], 'side': row['side'].title()} for row in records]
    def evaluate(board, own, opponents):
        if predictions is not None:
            return predictions[len(board.move_stack)]
        return engines.human_pairs([move.uci() for move in board.move_stack], own, opponents)
    evaluate_many = getattr(engines, 'human_pair_batches', None)
    if predictions is not None:
        evaluate_many = lambda requests: [evaluate(*request) for request in requests]
    prepare_rating_policies(game, fit_records, evaluate,
        engines.cache.directory / 'rating-policies', identity(engines.signature),
        evaluate_many=evaluate_many)
    for row, record in zip(records, fit_records, strict=True):
        row['_policies'] = {str(r): record['policies'][r] for r in RATINGS}
        row['_rating_policies'] = record['policies']


class GameAnalyzer:
    """One game's analysis job, with explicit preparation, scoring and fitting phases."""

    def __init__(self, game, engines, side=None, actual_elo=None, *, progress=print,
                 on_position=None, cancel=None, rating_output_dir=None, rating_scale=None):
        if side not in (None, 'white', 'black'):
            raise ValueError('Player side must be white, black, or unspecified.')
        if actual_elo is not None and (type(actual_elo) is not int or not 100 <= actual_elo <= 4000):
            raise ValueError('Actual Elo must be an integer from 100 to 4000.')
        self.game = game
        self.engines = engines
        self.side = side
        self.actual_elo = actual_elo
        self.rating_account_overrides = {side.title(): actual_elo} if side and actual_elo is not None else {}
        if side and actual_elo is None:
            self.actual_elo = header_elo(game.headers, side) or 1500
        self.progress = progress
        self.on_position = on_position
        self.cancel = cancel
        self.rating_output_dir = rating_output_dir
        self.headers = dict(game.headers)
        self.rating_scale_override = rating_scale
        self.rating_context = rating_context(self.headers, rating_scale)
        declared = {'headers': self.headers, 'rating_scale_override': self.rating_context}
        self.native_header_ratings = native_actual_ratings(declared)
        native_actual_ratings({**declared, 'rating_account_overrides': self.rating_account_overrides})

    def run(self):
        started = time.perf_counter()
        estimator = get_estimator()  # Reject invalid selection before engine work.
        self.rows = []
        self.history = []
        self.prepared = []
        self.rating_policies = []
        self.all_scores = {}
        self.boards = []

        self._check_cancelled()
        self._positions()
        self.progress(f'Maia rating evidence: {len(self.boards)} positions, equal-rating profiles from 600–2600 Lichess Blitz (cached).')
        self._prepare_maia()
        prepare_game_policies(self.game, self.rows, self.engines, predictions=self.predictions)
        jobs = self._jobs()
        scans = (self.engines.analyze_positions(jobs, cancel=self.cancel) if hasattr(self.engines, 'analyze_positions')
                 else ((i, self.engines.initial_analysis(*job)) for i, job in enumerate(jobs)))
        try:
            for completed, (index, scan) in enumerate(scans, 1):
                self._check_cancelled()
                self.positions[index]['stockfish'] = stockfish_result(self.boards[index], scan)
                if index < len(self.rows):
                    self._apply_scan(index, scan)
                if self.on_position:
                    self.on_position(index, self.positions[index])
                label = self.rows[index]['label'] if index < len(self.rows) else 'final position'
                self.progress(f'Analyzed {completed}/{len(self.boards)}: {label}')
        finally:
            if hasattr(scans, 'close'):
                scans.close()
        self._check_cancelled()
        self.progress(f'Estimating played level with {estimator.name}…')
        fitting = time.perf_counter()
        fit = self._fit()
        record = getattr(self.engines, 'record', None)
        if record:
            record('player_rating', wall_ms=(time.perf_counter()-fitting)*1000)
        analysis = {
            'schema_version': ANALYSIS_VERSION,
            'game_id': identity([self.game.board().fen(), self.history]),
            'headers': self.headers,
            'start_fen': self.game.board().fen(),
            'selected_player': {'side': self.side, 'actual_elo': self.actual_elo},
            'rating_account_overrides': self.rating_account_overrides,
            'analysis_execution': getattr(self.engines, 'last_analysis_execution', None),
            'positions': self.positions,
            'moves': self.rows,
        }
        if self.rating_scale_override is not None:
            analysis['rating_scale_override'] = self.rating_context['scale']
        store_elo_fit(analysis, fit)
        result = refresh_performance(add_flags(analysis, self.all_scores))
        self._check_cancelled()
        if self.rating_output_dir is not None:
            from analysis.player_rating.figures import export_saved_figures
            export_saved_figures(result, self.rating_output_dir)
        if record:
            record('game_analysis_completed', wall_ms=(time.perf_counter()-started)*1000)
        return result

    def _check_cancelled(self):
        if self.cancel is not None and self.cancel.is_set():
            raise AnalysisCancelled('Game analysis cancelled.')

    def _positions(self):
        board = self.game.board()
        for ply, move in enumerate(self.game.mainline_moves(), 1):
            self.boards.append(board.copy(stack=True))
            san = board.san(move)
            self.rows.append({
                'ply': ply, 'label': f'{board.fullmove_number}{"." if board.turn else "..."} {san}',
                'side': 'white' if board.turn else 'black',
                'fen': board.fen(), 'played': {'move': move.uci(), 'san': san},
            })
            board.push(move)
        self.boards.append(board.copy(stack=True))

    def _prepare_maia(self):
        """Infer each position/rating pair once, retaining policy and value for every UI."""
        self.predictions = [None] * len(self.boards)
        pending = []
        for index, board in enumerate(self.boards):
            started = time.perf_counter()
            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                value = .5 if outcome.winner is None else float(outcome.winner)
                self.predictions[index] = [{'policy': {}, 'value': value} for _ in FIT_RATINGS]
                if getattr(self.engines, 'record', None):
                    self.engines.record('maia_terminal', fen=board.fen(), ply=index,
                                        wall_ms=(time.perf_counter()-started)*1000)
            else:
                pending.append(index)
        size = max(1, CONFIG['MAIA']['BATCH_SIZE'] // len(FIT_RATINGS))
        for offset in range(0, len(pending), size):
            self._check_cancelled()
            indices = pending[offset:offset+size]
            requests = [(self.boards[i], list(FIT_RATINGS), list(FIT_RATINGS)) for i in indices]
            batches = (self.engines.human_pair_batches(requests) if hasattr(self.engines, 'human_pair_batches') else
                       [self.engines.human_pairs([move.uci() for move in board.move_stack], own, other)
                        for board, own, other in requests])
            for index, entries in zip(indices, batches, strict=True):
                self.predictions[index] = entries
        self._check_cancelled()
        self.positions = [{'ply': index, 'fen': board.fen(en_passant='fen'),
                           'maia': {f'maia_kdd_{rating}': entry for rating, entry in zip(FIT_RATINGS, entries, strict=True)}}
                          for index, (board, entries) in enumerate(zip(self.boards, self.predictions, strict=True))]

    def _jobs(self):
        jobs = []
        for row in self.rows:
            policies = row.pop('_policies')
            self.rating_policies.append(row.pop('_rating_policies', None))
            top = {rating: sorted(policy, key=lambda m: (-policy[m], m))[:5]
                   for rating, policy in policies.items()}
            # Engine evidence must not depend on which client requests a report.
            level = self.native_header_ratings[row['side'].title()]
            level = 1500 if level is None else level
            near = str(min(RATINGS, key=lambda rating: abs(rating-level)))
            jobs.append((self.history.copy(), row['played']['move'], top[near][:4]))
            self.prepared.append((policies, top))
            self.history.append(row['played']['move'])
        final = self.positions[-1]['maia']['maia_kdd_1500']['policy']
        jobs.append((self.history.copy(), None, sorted(final, key=lambda move: (-final[move], move))[:4]))
        return jobs

    def _apply_scan(self, index, scan):
        row = self.rows[index]
        board = self.boards[index]
        policies, top = self.prepared[index]
        scores = {line['uci']: line for line in scan['lines']}
        if set(scores) != {move.uci() for move in board.legal_moves}:
            raise RuntimeError('Stockfish did not cover every legal move; increase the search time or select bounded ANALYSIS.STOCKFISH_SEARCH_STRATEGY.')
        best = eval_value(scores[scan['best_move']])
        candidates = {row['played']['move'], scan['best_move']} | {
            move for moves in top.values() for move in moves
        }

        def entry(uci):
            child = board.copy()
            child.push_uci(uci)
            value = eval_value(scores[uci])
            if child.is_checkmate():
                value = '#-0' if child.turn else '#0'
            elif child.is_game_over(claim_draw=False):
                value = 0.0
            return {'move': uci, 'san': board.san(chess.Move.from_uci(uci)), 'eval': value,
                    'loss': eval_loss(best, value, row['side'])}

        all_values = {move: entry(move) for move in scores}
        values = {move: all_values[move] for move in sorted(candidates)}
        self.all_scores[row['ply']] = list(all_values.values())
        row.update(
            position_eval=best,
            played=values[row['played']['move']],
            maia={rating: [{**values[move], 'p': float(f'{policies[rating][move]:.5g}')}
                          for move in moves] for rating, moves in top.items()},
            candidate_moves=[
                {**value, 'maia_p': {rating: float(f'{policy[move]:.5g}')
                                    for rating, policy in policies.items()}}
                for move, value in values.items()
            ],
        )

    def _fit(self):
        records = [
            {'move': row['played']['move'], 'position_score': centipawns(row['position_eval']),
             'scores': {entry['move']: centipawns(entry['eval'])
                        for entry in self.all_scores[row['ply']]},
             'policies': policies}
            for row, policies in zip(self.rows, self.rating_policies, strict=True)
        ]
        return fit_game(
            self.game, records,
            lambda board, own, other: self.engines.human_pairs(
                [move.uci() for move in board.move_stack], own, other),
            self.engines.cache.directory, identity(self.engines.signature),
            evaluate_many=getattr(self.engines, 'human_pair_batches', None),
            ratings=self.rating_account_overrides,
            rating_scale=self.rating_context,
        )


def analyze_game(game, engines, side=None, actual_elo=None, *, progress=print, on_position=None, cancel=None,
                 rating_output_dir=None, rating_scale=None):
    """Analyze one game; optionally export into its explicit player-rating folder.

    Without ``rating_output_dir``, the caller controls output publication, as the
    web job does when staging a result for its transactional repository save.
    """
    return GameAnalyzer(game, engines, side, actual_elo, progress=progress,
                        on_position=on_position, cancel=cancel, rating_output_dir=rating_output_dir,
                        rating_scale=rating_scale).run()
