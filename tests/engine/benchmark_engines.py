"""Reproducible engine timing: uv run python -m tests.engine.benchmark_engines."""
from __future__ import annotations

import argparse
import json
import statistics
import shutil
import threading
import time
from pathlib import Path

import chess
import chess.engine
import torch

from engine.maia import MaiaPolicy
from analysis.settings import CONFIG
from analysis.stockfish_search import SearchControl, resolve_search_seconds, stream_evaluations
from engine.uci import stockfish_executable


def positions():
    board = chess.Board()
    samples = [board.copy()]
    for index, san in enumerate("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7".split()):
        board.push_san(san)
        if index in (9, 19):
            samples.append(board.copy())
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'])
    parser.add_argument("--max-seconds", type=float, default=CONFIG['ANALYSIS']['STOCKFISH_EVALUATION']['DEFAULT_SEARCH_SECONDS'])
    parser.add_argument("--maia-only", action="store_true")
    parser.add_argument("--strategy", choices=("baseline", "bounded", "staged", "exhaustive"), default=CONFIG['ANALYSIS']['STOCKFISH_SEARCH_STRATEGY'])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=CONFIG['MAIA']['DEVICE'])
    args = parser.parse_args()
    args.max_seconds = resolve_search_seconds(args.max_seconds)
    samples = positions()
    policy = MaiaPolicy(CONFIG['MAIA']['MODEL'], CONFIG['MAIA']['CACHE_DIR'], device=args.device)
    timings = []
    ratings = list(range(600, 2601, 100))
    for run in range(4):
        start = time.perf_counter()
        result = policy.batch_evaluate([samples[1].fen()] * len(ratings), ratings, ratings)
        elapsed = time.perf_counter() - start
        assert all(abs(sum(row["policy"].values()) - 1) < 1e-5 for row in result)
        timings.append(elapsed)
        print(f"Maia run {run}: {elapsed:.3f}s", flush=True)
    output = {"torch": torch.__version__, "device": policy._engine.cfg.device,
              "model": CONFIG['MAIA']['MODEL'], "ratings": len(ratings),
              "maia_cold_seconds": timings[0], "maia_warm_seconds": timings[1:],
              "maia_warm_median_seconds": statistics.median(timings[1:]),
              "stockfish_threads": args.threads, "strategy": args.strategy,
              "position_budget_seconds": args.max_seconds, "stockfish": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.maia_only:
        return
    candidates = []
    for board in samples:
        prediction = policy.batch_evaluate([board.fen()], [1500], [1500])[0]
        candidates.append(list(prediction["policy"])[:4])
    executable = stockfish_executable(CONFIG['STOCKFISH']['EXECUTABLE'], CONFIG['STOCKFISH']['CACHE_DIR'])
    binary = Path(shutil.which(executable) or executable)
    with chess.engine.SimpleEngine.popen_uci(str(binary)) as engine:
        engine.configure({"Threads": args.threads, "Hash": CONFIG['ANALYSIS']['STOCKFISH_HASH_MB_PER_WORKER']})
        for depth in (12, 18):
            for index, board in enumerate(samples):
                engine.configure({"Clear Hash": None})
                start = time.perf_counter()
                details = {}
                if args.strategy == "baseline":
                    result = engine.analyse(board, chess.engine.Limit(depth=depth, time=args.max_seconds),
                                            multipv=board.legal_moves.count())
                    achieved = min(info.get("depth", 0) for info in result)
                else:
                    control = SearchControl()
                    timer = threading.Timer(args.max_seconds + (2 if args.strategy == 'bounded' else 0), control.cancel)
                    options = {"maiaCandidateMoves": candidates[index],
                               "forcedCandidateMoves": [["e2e4"], ["f1e1"], ["b1d2"]][index]}
                    result = None
                    timer.start()
                    try:
                        for result in stream_evaluations(engine, board, depth, options, args.strategy, control,
                                                        seconds=args.max_seconds):
                            pass
                    finally:
                        timer.cancel()
                        timer.join()
                    achieved = result["depth"] if result else 0
                    details = {"complete": bool(result and result["complete"]), "options": options,
                               "root_move_depth_vec": result["root_move_depth_vec"] if result else {}}
                row = {"fen": board.fen(), "requested_depth": depth,
                       "completed_depth": achieved,
                       "legal_moves": board.legal_moves.count(),
                       "seconds": time.perf_counter() - start, **details}
                output["stockfish"].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
