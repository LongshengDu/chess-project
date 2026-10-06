"""Stream and persist the shared game-analysis pipeline for saved web games."""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
from queue import Queue
import threading
import time
from contextlib import nullcontext
from tempfile import TemporaryDirectory

import chess.pgn

from analysis.game.pipeline import analyze_game
from analysis.cache import write_json
from analysis.engine_session import Engines, Limits
from analysis.player_rating.figures import export_saved_figures
from backend.settings import CONFIG


@dataclass
class GameAnalysisRun:
    game_id: str
    run_id: str
    game: chess.pgn.Game
    limits: Limits
    profiler_id: str | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    events: Queue = field(default_factory=Queue)
    started: bool = False


class FullGameAnalysis:
    """HTTP job lifetime only; chess calculations belong to analysis.game.pipeline."""

    def __init__(self, platform):
        self.platform = platform
        self.runs = {}
        self.lock = threading.Lock()

    def prepare(self, game_id, data):
        if not isinstance(data, dict):
            raise ValueError('Expected game-analysis options')
        run_id = data.get('run_id')
        if not isinstance(run_id, str) or not 1 <= len(run_id) <= 100:
            raise ValueError('Provide a game-analysis run identifier')
        limits = self.platform.analysis_limits(data.get('target_depth'), data.get('seconds'))
        saved = self.platform.repository.get(game_id)
        if saved is None:
            raise LookupError('Saved game not found')
        game = chess.pgn.read_game(io.StringIO(saved['snapshot']['plain_pgn']))
        if game is None or game.errors:
            raise ValueError('Saved game has an invalid PGN')
        profiler_id = data.get('profiler_id')
        profiler = self.platform.profiler
        if profiler_id is not None and (not isinstance(profiler_id, str) or profiler is None
                or not profiler.active or profiler_id != profiler.run_id):
            raise ValueError('Unknown profiler run')
        if profiler_id is not None:
            configuration = profiler.data['configuration']
            if (configuration.get('game_id', game_id) != game_id
                    or configuration['target_depth'] != limits.depth
                    or configuration.get('position_budget_seconds', limits.verify_ms / 1000)
                        != (limits.verify_ms / 1000 if self.platform.strategy == 'bounded' else None)
                    or configuration.get('max_search_seconds', limits.max_ms / 1000) != limits.max_ms / 1000
                    or configuration['total_positions'] != len(list(game.mainline_moves())) + 1):
                raise ValueError('Profiler options do not match this game-analysis run')
        run = GameAnalysisRun(game_id, run_id, game, limits, profiler_id)
        with self.lock:
            if run_id in self.runs or any(item.game_id == game_id for item in self.runs.values()):
                raise ValueError('This game already has an active analysis run')
            self.runs[run_id] = run
        return run

    def cancel(self, game_id, run_id):
        with self.lock:
            run = self.runs.get(run_id)
            if run is not None and run.game_id == game_id:
                run.cancelled.set()
                return True
        return False

    def save_positions(self, game_id, positions):
        """Serialize browser autosaves against shared-result publication."""
        with self.lock:
            if any(run.game_id == game_id for run in self.runs.values()):
                return True
            return self.platform.repository.save_analysis(game_id, positions)

    def stream(self, run):
        run.started = True
        worker = threading.Thread(target=self._analyze, args=(run,), name='full-game-analysis')
        worker.start()
        try:
            while True:
                event = run.events.get()
                if event is None:
                    break
                yield json.dumps(event, allow_nan=False) + '\n'
        finally:
            run.cancelled.set()
            worker.join()
            self.release(run)

    def release(self, run):
        """Also handle a response closed before its stream was first consumed."""
        run.cancelled.set()
        with self.lock:
            if self.runs.get(run.run_id) is run:
                del self.runs[run.run_id]
        profiler = self.platform.profiler
        if not run.started and run.profiler_id and profiler:
            with profiler.lock:
                if profiler.active and profiler.run_id == run.profiler_id:
                    profiler.abort({'run_id': run.profiler_id, 'cancelled': True})

    def _analyze(self, run):
        platform = self.platform
        profiler = platform.profiler if run.profiler_id else None
        started = time.perf_counter()
        published = False
        nodes = list(run.game.mainline())

        def record(kind, **values):
            if profiler:
                with profiler.lock:
                    if profiler.active and profiler.run_id == run.profiler_id:
                        profiler.record(kind, **values)

        def progress(message):
            run.events.put({'type': 'progress', 'message': message})
            record('game_analysis_stage', message=message)

        def position(index, value):
            if run.cancelled.is_set():
                return
            run.events.put({'type': 'position', 'index': index, 'position': value})
            if profiler:
                with profiler.lock:
                    if profiler.active and profiler.run_id == run.profiler_id:
                        profiler.position({'run_id': run.profiler_id, 'ply': index, 'fen': value['fen'],
                            'complete': True, 'move': nodes[index].move.uci() if index < len(nodes) else None,
                            'label': nodes[index].san() if index < len(nodes) else 'Final position',
                            'elapsed_ms': (time.perf_counter() - started) * 1000})

        try:
            progress('Preparing shared full-game analysis…')
            # Cold runtime profiling never deletes or contaminates ordinary evidence.
            cache = TemporaryDirectory(prefix='chess-profiler-') if profiler else nullcontext(CONFIG['ANALYSIS']['CACHE_DIR'])
            with cache as cache_dir:
                with Engines.borrowed(run.game.board().fen(), cache_dir,
                        platform.maia, platform.stockfish, limits=run.limits, cancel=run.cancelled,
                        record=record if profiler else None) as engines:
                    result = analyze_game(run.game, engines, progress=progress,
                                          on_position=position, cancel=run.cancelled)
                if profiler and not run.cancelled.is_set():
                    # Keep rating evidence reusable after the isolated cold run.
                    for source in Path(cache_dir).rglob('*.json'):
                        target = CONFIG['ANALYSIS']['CACHE_DIR'] / source.relative_to(cache_dir)
                        write_json(target, json.loads(source.read_text(encoding='utf-8')))
            if run.cancelled.is_set():
                run.events.put({'type': 'cancelled'})
                return
            # Render outside locks, away from caches and the currently published
            # files. Cancellation or deletion can discard this complete draft.
            output_root = platform.repository.output_directory
            output_root.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix='.rating-', dir=output_root) as staged:
                export_saved_figures(result, staged)
                # Publication and cancellation have one ordering. The repository
                # transaction also prevents deletion between its check and save.
                with self.lock:
                    if run.cancelled.is_set():
                        run.events.put({'type': 'cancelled'})
                        return
                    if not platform.repository.save_full_analysis(run.game_id, result,
                                                                  rating_figures=Path(staged)):
                        raise LookupError('The game was deleted while analysis was running')
                    published = True
            run.events.put({'type': 'complete', 'analysis': result})
        except Exception as exc:
            run.events.put({'type': 'cancelled'} if run.cancelled.is_set()
                           else {'type': 'error', 'message': str(exc)})
        finally:
            try:
                if profiler and not published:
                    with profiler.lock:
                        if profiler.active and profiler.run_id == run.profiler_id:
                            profiler.abort({'run_id': run.profiler_id, 'cancelled': run.cancelled.is_set()})
            finally:
                run.events.put(None)
