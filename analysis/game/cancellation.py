"""Job-local cancellation that stops active searches without closing shared engines."""
from contextlib import contextmanager
import threading

from analysis.stockfish_search import SearchControl


class AnalysisCancelled(RuntimeError):
    """The caller cancelled this analysis before a complete result was produced."""


class AnalysisCancellation:
    def __init__(self, event=None):
        self.event = event if event is not None else threading.Event()
        self._aborted = threading.Event()
        self._controls = set()
        self._lock = threading.Lock()

    def check(self):
        if self.event.is_set() or self._aborted.is_set():
            raise AnalysisCancelled('Game analysis cancelled.')

    def abort(self):
        self._aborted.set()
        with self._lock:
            controls = tuple(self._controls)
        for control in controls:
            control.cancel()

    @contextmanager
    def watch(self):
        """Observe the caller's Event even while UCI is waiting for engine output."""
        finished = threading.Event()

        def monitor():
            while not finished.wait(.025):
                if self.event.is_set():
                    self.abort()
                    return

        watcher = threading.Thread(target=monitor, name='analysis-cancellation', daemon=True)
        watcher.start()
        try:
            self.check()
            yield
        finally:
            finished.set()
            watcher.join()

    @contextmanager
    def search(self):
        self.check()
        control = SearchControl()
        with self._lock:
            self._controls.add(control)
        try:
            # Cancellation can race with registering the new lease.
            self.check()
            yield control
            self.check()
        finally:
            with self._lock:
                self._controls.discard(control)
