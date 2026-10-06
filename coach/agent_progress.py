"""Short user-facing progress, separate from reports and private model reasoning."""
from __future__ import annotations

import re
import math
import threading
import time

from coach.settings import CONFIG


PROGRESS_INSTRUCTIONS = (
    'Give brief public progress summaries when starting, changing focus or drafting: current check, '
    'supported finding, next question. No private reasoning, scratch work or raw JSON. '
    'Use Codex commentary. No extra requests/tools just to narrate. '
    'Keep progress outside the final Markdown report.'
)


def print_progress(message):
    print(message, flush=True)


class CoachProgress:
    def __init__(self, sink=None, *, interval=None):
        self.sink = sink
        self.interval = CONFIG['COACH']['PROGRESS_INTERVAL_SECONDS'] if interval is None else interval
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError('Progress interval must be positive.')
        self.started = self.last_update = time.monotonic()
        self.current = 'Waiting for the coach'
        self.stopped = threading.Event()
        self.lock = threading.RLock()
        self.thread = None

    def start(self):
        if self.sink is not None:
            self.thread = threading.Thread(target=self._heartbeat, name='coach-progress', daemon=True)
            self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def emit(self, message):
        if self.sink is None or self.stopped.is_set():
            return
        message = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', message)
        message = ' '.join(re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', message).split())
        if not message:
            return
        if len(message) > 600:
            message = message[:597] + '...'
        with self.lock:
            if self.stopped.is_set():
                return
            elapsed = int(time.monotonic()-self.started)
            try:
                self.sink(f'[Coach {elapsed//60:02d}:{elapsed%60:02d}] {message}')
            except Exception:
                # A closed output pipe or failing UI callback must not ruin a report.
                self.sink = None
            self.last_update = time.monotonic()

    def stage(self, message):
        with self.lock:
            self.current = message.rstrip('.…')
            self.emit(message)

    def idle(self):
        with self.lock:
            self.current = "Waiting for the coach's next update"

    def pulse(self):
        with self.lock:
            if time.monotonic()-self.last_update >= self.interval:
                self.emit(self.current + ' — still in progress.')

    def _heartbeat(self):
        while not self.stopped.wait(self.interval):
            self.pulse()

    def commentary(self, text):
        # Reasoning events/fields are never passed here. Suppress tagged private
        # blocks too, including incomplete tags, instead of summarizing them.
        if isinstance(text, str) and not re.search(r'<(?:think|analysis|reasoning)\b', text, re.I):
            self.emit(text)


def tool_activity(name, arguments, analysis):
    ply = arguments.get('ply')
    label = f'ply {ply}' if type(ply) is int else 'the position'
    if type(ply) is int and 1 <= ply <= len(analysis['moves']):
        label = analysis['moves'][ply-1]['label']
    action = {'get_game_analysis': 'Reviewing the game overview',
              'get_position': 'Inspecting the position',
              'get_leadup': 'Tracing the earlier decisions',
              'maia_analyze': 'Checking human move preferences',
              'maia_compare': 'Comparing Maia choices across ratings',
              'stockfish_analyze': 'Verifying the line with Stockfish',
              'explore_candidate': 'Exploring likely human replies and checking the candidate',
              'compare_played_vs_candidate': 'Comparing likely human replies to the played move and alternative'}[name]
    return action if name == 'get_game_analysis' else f'{action} at {label}.'


def tool_summary(name, result):
    if name in ('compare_played_vs_candidate', 'explore_candidate'):
        branches = [result['played'], result['candidate']] if name == 'compare_played_vs_candidate' else [result]
        ratings = sorted({int(r) for branch in branches for r in (branch.get('maia_after_defense') or {})})
        moves = [b.get('stockfish', {}).get('san') for b in branches]
        checked = ('Checked ' + ' and '.join(moves) + ' with human reply evidence and engine defenses' if all(moves)
                   else 'Checked both branches' if len(branches) == 2 else 'Checked the candidate branch')
        return checked + ('; Maia continuations at ' + '/'.join(map(str, ratings)) + ' Elo.' if ratings else '; no nonterminal continuation to query.')
    if name == 'stockfish_analyze':
        return f"Stockfish returned {len(result.get('lines', []))} checked continuations."
    if name in ('maia_compare', 'maia_analyze'):
        return 'Retrieved Maia choices at ' + '/'.join(result.get('maia', {})) + ' Elo.'
    if name == 'get_leadup':
        return 'Retrieved earlier moves and positional changes for this decision.'
    return 'Local evidence is ready.'
