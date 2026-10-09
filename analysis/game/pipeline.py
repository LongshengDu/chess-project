"""Whole-game orchestration over common engines and accuracy curve evidence."""
from __future__ import annotations

import time

import chess
import chess.pgn

from analysis.cache.storage import identity
from analysis.game.cancellation import AnalysisCancelled
from analysis.position_evaluation import RATINGS, eval_value, eval_loss
from analysis.move_hints import add_flags
from analysis.accuracy.performance import refresh_performance
from analysis.position_results import stockfish_result
from analysis.settings import CONFIG

from analysis.accuracy.policies import prepare_policies
from analysis.accuracy.evidence import RATINGS as ACCURACY_RATINGS
from analysis.accuracy.service import refresh_saved_curve
from analysis.maia_context import native_actual_ratings
from analysis.game.metadata import game_metadata

ANALYSIS_VERSION = 9


def prepare_game_policies(game, records, predictions):
    """Prepare common Maia evidence and the policies used by move displays."""
    policy_records = [{'move': row['played']['move'], 'side': row['side'].title()} for row in records]
    prepare_policies(game, policy_records, predictions)
    for row, record in zip(records, policy_records, strict=True):
        row['_policies'] = {str(r): record['policies'][r] for r in RATINGS}


class GameAnalyzer:
    """One game's analysis job, with explicit preparation, scoring and accuracy phases."""

    def __init__(self, game, session, actual_elo=None, *, progress=print,
                 on_position=None, cancel=None, accuracy_output_dir=None, rating_scale=None):
        self.game = game
        self.session = session
        self.progress = progress
        self.on_position = on_position
        self.cancel = cancel
        self.accuracy_output_dir = accuracy_output_dir
        self.headers = dict(game.headers)
        self.metadata = game_metadata(self.headers, actual_elo, rating_scale, game=game)
        self.native_ratings = native_actual_ratings({'game': self.metadata})

    def run(self):
        started = time.perf_counter()
        self.rows = []
        self.history = []
        self.prepared = []
        self.all_scores = {}
        self.boards = []

        self._check_cancelled()
        self._positions()
        self._report_configuration()
        self.progress(f'Maia accuracy evidence: {len(self.boards)} positions, equal-rating profiles from 600–2600 Lichess Blitz.')
        self._prepare_maia()
        prepare_game_policies(self.game, self.rows, self.predictions[:-1])
        jobs = self._jobs()
        scans = self.session.analyze_positions(jobs, cancel=self.cancel)
        try:
            for completed, (index, scan) in enumerate(scans, 1):
                self._check_cancelled()
                if scan is None and index < len(self.rows):
                    raise ValueError('Every played position requires complete Stockfish evidence.')
                self.positions[index]['stockfish'] = stockfish_result(self.boards[index], scan) if scan is not None else None
                if index < len(self.rows):
                    self._apply_scan(index, scan)
                if self.on_position:
                    self.on_position(index, self.positions[index])
                label = self.rows[index]['label'] if index < len(self.rows) else 'final position'
                if scan is None:
                    self.progress(f'Prepared {completed}/{len(self.boards)}: {label} (unscored)')
                else:
                    result = self.positions[index]['stockfish']
                    seconds = result.get('elapsed_seconds')
                    phases = result.get('phases', [])
                    if seconds is None and phases and all('wall_ms' in phase for phase in phases):
                        seconds = sum(phase['wall_ms'] for phase in phases) / 1000
                    duration = 'unavailable' if seconds is None else f'{seconds:.3f}s'
                    self.progress(f'Analyzed {completed}/{len(self.boards)}: {label} '
                                  f'(depth {result["depth"]}, search time {duration})')
        finally:
            scans.close()
        self._check_cancelled()
        record = getattr(self.session, 'record', None)
        references = self.session.position_references(self.boards, ACCURACY_RATINGS)
        # Keep the UI view identical to the pinned measurements; do not round-trip
        # engine payloads into a second cache representation.
        for position, reference in zip(self.positions, references, strict=True):
            position['stockfish'] = self.session.cache.get_reference(reference['stockfish'])
        analysis = {
            'schema_version': ANALYSIS_VERSION,
            'game_id': identity([self.game.board().fen(), self.history]),
            'headers': self.headers,
            'start_fen': self.game.board().fen(),
            'game': self.metadata,
            'coaching': {},
            'analysis_execution': getattr(self.session, 'last_analysis_execution', None),
            'positions': self.positions,
            'position_references': references,
            'moves': self.rows,
        }
        result = refresh_performance(add_flags(analysis, self.all_scores))
        self.progress('Calculating native accuracy curve and absolute deviation...')
        measuring = time.perf_counter()
        refresh_saved_curve(result)
        if record:
            record('accuracy_curve', wall_ms=(time.perf_counter()-measuring)*1000)
        self._check_cancelled()
        if self.accuracy_output_dir is not None:
            from analysis.accuracy.figures import export_saved_figures
            export_saved_figures(result, self.accuracy_output_dir)
        if record:
            record('game_analysis_completed', wall_ms=(time.perf_counter()-started)*1000)
        return result

    def _report_configuration(self):
        engines = getattr(self.session, 'engines', None)
        if engines is None:
            self.progress('Analysis: rebuilding from cache; Maia and Stockfish disabled.')
            return
        device = (engines.signature or {}).get('device', 'unavailable')
        self.progress(f'Maia: model={engines.maia_model}, device={device}, '
                      f'batch_size={CONFIG["MAIA"]["BATCH_SIZE"]}.')
        resources = engines.analysis_pool if engines.analysis_pool is not None else engines
        workers = min(self.session.analysis_workers, len(self.boards))
        limits = self.session.limits
        search_ms = limits.verify_ms if limits.analysis_strategy == 'bounded' else limits.max_ms
        self.progress(f'Stockfish: workers={workers}, '
                      f'threads_per_worker={resources.threads_per_worker}, '
                      f'hash_mb_per_worker={resources.hash_mb}, strategy={limits.analysis_strategy}, '
                      f'depth_ceiling={limits.depth}, search_seconds={search_ms / 1000:g}, '
                      f'max_seconds={limits.max_ms / 1000:g}.')

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
        pending = list(range(len(self.boards)))
        size = max(1, CONFIG['MAIA']['BATCH_SIZE'] // len(ACCURACY_RATINGS))
        for offset in range(0, len(pending), size):
            self._check_cancelled()
            indices = pending[offset:offset+size]
            requests = [(self.boards[i], list(ACCURACY_RATINGS), list(ACCURACY_RATINGS)) for i in indices]
            batches = self.session.human_pair_batches(requests)
            for index, entries in zip(indices, batches, strict=True):
                self.predictions[index] = entries
        self._check_cancelled()
        self.positions = [{'ply': index, 'fen': board.fen(en_passant='fen'),
                           'maia': {f'maia_kdd_{rating}': entry for rating, entry in zip(ACCURACY_RATINGS, entries, strict=True)
                                    if entry is not None}}
                          for index, (board, entries) in enumerate(zip(self.boards, self.predictions, strict=True))]

    def _jobs(self):
        jobs = []
        for row in self.rows:
            policies = row.pop('_policies')
            top = {rating: sorted(policy, key=lambda m: (-policy[m], m))[:5]
                   for rating, policy in policies.items()}
            # Both clients schedule searches using the effective account context.
            level = self.native_ratings[row['side'].title()]
            level = 1500 if level is None else level
            near = str(min(RATINGS, key=lambda rating: abs(rating-level)))
            jobs.append((self.history.copy(), row['played']['move'], top[near][:4]))
            self.prepared.append((policies, top))
            self.history.append(row['played']['move'])
        final = self.positions[-1]['maia'].get('maia_kdd_1500', {}).get('policy', {})
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
            maia={rating: {'moves': [{**values[move], 'p': float(f'{policies[rating][move]:.5g}')}
                                    for move in moves]} for rating, moves in top.items()},
            candidate_moves=[
                {**value, 'maia_p': {rating: float(f'{policy[move]:.5g}')
                                    for rating, policy in policies.items()}}
                for move, value in values.items()
            ],
        )


def analyze_game(game, session, actual_elo=None, *, progress=print, on_position=None, cancel=None,
                 accuracy_output_dir=None, rating_scale=None):
    """Analyze one game; optionally export default SVGs into its game output folder.

    ``actual_elo`` overrides both players' account ratings; omitted ratings come
    from their respective PGN headers. Saved evidence covers both players.
    Without ``accuracy_output_dir``, the caller controls output publication, as the
    web job does when staging a result for its transactional repository save.
    """
    return GameAnalyzer(game, session, actual_elo, progress=progress,
                        on_position=on_position, cancel=cancel, accuracy_output_dir=accuracy_output_dir,
                        rating_scale=rating_scale).run()
