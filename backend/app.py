from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask

# Keep both `python backend/app.py` and `python -m backend.app` working.
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.maia import MaiaPolicy
from engine.stockfish import StockfishScorer
from engine.assets import ensure_runtime_assets
from backend.settings import CONFIG
from backend.analysis_positions import PlatformAnalysis
from backend.routes_analysis import AnalysisRoutes
from backend.routes_play import PlayRoutes
from backend.routes_profiler import ProfilerRoutes
from analysis.profiler.runtime import RuntimeProfiler


def create_app(
    platform: PlatformAnalysis,
    static_folder: Path | None = None,
) -> Flask:
    if static_folder is None:
        static_folder = CONFIG['FRONTEND']['STATIC_DIR']
    server = Flask(__name__, static_folder=str(static_folder), static_url_path="")
    server.config["MAX_CONTENT_LENGTH"] = CONFIG['SERVER']['MAX_REQUEST_BODY_MB'] * 1024 * 1024
    server.json.sort_keys = False
    AnalysisRoutes(platform).register(server)
    PlayRoutes(platform.play).register(server)
    if platform.profiler:
        ProfilerRoutes(platform).register(server)

    @server.get("/")
    @server.get("/analysis")
    @server.get("/analysis/<path:route>")
    @server.get("/play")
    @server.get("/play/maia")
    def index(route=None):
        return server.send_static_file("index.html")

    @server.errorhandler(TypeError)
    @server.errorhandler(ValueError)
    def invalid_action(error):
        return {"error": str(error)}, 400

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local Maia play and game analysis")
    parser.add_argument("--host", default=CONFIG['SERVER']['HOST'])
    parser.add_argument("--port", type=int, default=CONFIG['SERVER']['PORT'])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=CONFIG['MAIA']['DEVICE'],
                        help="Maia inference device (auto uses CUDA when available)")
    parser.add_argument("--stockfish-threads-per-worker", type=int,
                        default=CONFIG['ANALYSIS']['STOCKFISH_THREADS_PER_WORKER'],
                        help="CPU search threads for each independent Stockfish worker")
    parser.add_argument("--analysis-strategy", choices=("bounded", "staged", "exhaustive"), default=CONFIG['ANALYSIS']['STOCKFISH_SEARCH_STRATEGY'],
                        help="Analysis search: depth/time budgets (default), uncapped staged, or exhaustive")
    parser.add_argument("--profiler-dir", type=Path, default=CONFIG['ANALYSIS']['PROFILER_DIR'], help="Record opt-in runtime timing reports in this directory")
    parser.add_argument("--analysis-database", type=Path, default=CONFIG['SERVER']['STORAGE']['DATABASE'], help="Use a separate saved-study database")
    args = parser.parse_args(argv)
    if not 1 <= args.stockfish_threads_per_worker <= 256:
        parser.error("--stockfish-threads-per-worker must be between 1 and 256")
    static_folder = CONFIG['FRONTEND']['STATIC_DIR']
    if not (static_folder / "index.html").exists():
        raise SystemExit("Frontend not built. Run: uv run python web/build.py")

    stockfish_path = ensure_runtime_assets()
    stockfish: StockfishScorer | None = None
    try:
        maia = MaiaPolicy(CONFIG['MAIA']['MODEL'], CONFIG['MAIA']['CACHE_DIR'], device=args.device)
        stockfish = StockfishScorer(stockfish_path, threads_per_worker=args.stockfish_threads_per_worker)
        platform = PlatformAnalysis(maia, stockfish, args.analysis_database,
                                    strategy=args.analysis_strategy,
                                    profiler=RuntimeProfiler(args.profiler_dir) if args.profiler_dir else None)
        pool = stockfish.analysis_pool
        print(f"Maia device: {maia._engine.cfg.device}; Stockfish: {pool.workers} workers x "
              f"{pool.threads_per_worker} threads per worker ({pool.total_threads} total search threads); "
              f"search: {args.analysis_strategy}", flush=True)
        if maia._engine.cfg.device == "cpu":
            print("For an NVIDIA GPU, launch with: uv run --extra cuda python backend/app.py", flush=True)
        create_app(platform, static_folder).run(
            host=args.host,
            port=args.port,
            threaded=True,
        )
    finally:
        if stockfish is not None:
            stockfish.close()


if __name__ == "__main__":
    main()
