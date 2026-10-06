"""Shared Stockfish UCI process startup, limits, and cleanup."""
import math
from pathlib import Path

import chess.engine

from engine.settings import CONFIG


def seconds_to_milliseconds(seconds):
    """Keep YAML times in seconds while preserving millisecond tool/API contracts."""
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < .001):
        raise ValueError('Search time in seconds must be finite and at least 0.001.')
    return round(seconds * 1000)


def stockfish_executable(executable, cache_dir):
    """Resolve an explicit executable or the existing local engine cache."""
    if executable is not None:
        return str(executable)
    return next((str(path) for path in sorted(Path(cache_dir).glob('stockfish*'))
                 if path.is_file() and (path.suffix == '.exe' or not path.suffix)), 'stockfish')


def start_stockfish(executable, *, threads=None, hash_mb=None, timeout=CONFIG['STOCKFISH']['START_TIMEOUT_SECONDS']):
    engine = chess.engine.SimpleEngine.popen_uci(str(executable), timeout=timeout)
    try:
        engine.configure({
            "Threads": CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'] if threads is None else threads,
            "Hash": CONFIG['ANALYSIS']['STOCKFISH_HASH_MB_PER_WORKER'] if hash_mb is None else hash_mb,
        })
        return engine
    except Exception:
        engine.close()
        raise


def close_engine(engine):
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            engine.close()
