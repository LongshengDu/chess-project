from __future__ import annotations

import subprocess
from pathlib import Path

import chess.engine
from flask import Flask, request

if __package__:
    from .analysis import MaiaPolicy, PgnAnalysisApi, StockfishScorer
    from .assets import MAIA_MODEL_SPEC, ensure_runtime_assets
    from .game import ChessApi
    from .settings import (
        MAIA_CACHE_DIR,
        MAIA_ELO,
        MAIA_ENGINE_COMMAND,
        MAIA_MODEL_NAME,
        WEB_DIST_DIR,
    )
else:  # Support: python backend/app.py
    from analysis import MaiaPolicy, PgnAnalysisApi, StockfishScorer
    from assets import MAIA_MODEL_SPEC, ensure_runtime_assets
    from game import ChessApi
    from settings import (
        MAIA_CACHE_DIR,
        MAIA_ELO,
        MAIA_ENGINE_COMMAND,
        MAIA_MODEL_NAME,
        WEB_DIST_DIR,
    )


def _request_data() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _analysis_path(data: dict) -> tuple[int, list[str]]:
    try:
        ply = int(data.get("ply", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid PGN position") from error

    moves = data.get("moves", [])
    if not isinstance(moves, list):
        raise TypeError("Invalid variation path")
    return ply, [str(move) for move in moves]


def create_app(
    game: ChessApi,
    pgn_analysis: PgnAnalysisApi,
    static_folder: Path = WEB_DIST_DIR,
) -> Flask:
    server = Flask(__name__, static_folder=str(static_folder), static_url_path="")

    @server.get("/")
    def index():
        return server.send_static_file("index.html")

    @server.get("/api/state")
    def state():
        return game.state()

    @server.post("/api/move")
    def move():
        return game.move(str(_request_data().get("uci", "")))

    @server.post("/api/reply")
    def reply():
        return game.reply()

    @server.post("/api/action")
    def action():
        return game.action(str(_request_data().get("name", "")))

    @server.post("/api/new-game")
    def new_game():
        return game.new_game()

    @server.post("/api/analysis/pgn")
    def import_pgn():
        return pgn_analysis.import_pgn(str(_request_data().get("pgn", "")))

    @server.post("/api/analysis/position")
    def analyse_position():
        data = _request_data()
        ply, variation_moves = _analysis_path(data)
        return pgn_analysis.analyse_position(ply, variation_moves)

    @server.post("/api/analysis/move")
    def analysis_move():
        data = _request_data()
        ply, variation_moves = _analysis_path(data)
        return pgn_analysis.play_variation_move(
            ply,
            variation_moves,
            str(data.get("uci", "")),
        )

    @server.errorhandler(TypeError)
    @server.errorhandler(ValueError)
    def invalid_action(error):
        return {"error": str(error)}, 400

    return server


def _popen_options() -> dict:
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def main() -> None:
    if not (WEB_DIST_DIR / "index.html").exists():
        raise SystemExit("Frontend not built. Run: uv run web/build.py")

    stockfish_path = ensure_runtime_assets()
    game: ChessApi | None = None
    pgn_analysis: PgnAnalysisApi | None = None
    try:
        maia_engine = chess.engine.SimpleEngine.popen_uci(
            MAIA_ENGINE_COMMAND,
            **_popen_options(),
        )
        game = ChessApi(
            maia_engine,
            model_name=MAIA_MODEL_SPEC.display_name,
            model_elo=MAIA_ELO,
        )
        pgn_analysis = PgnAnalysisApi(
            MaiaPolicy(MAIA_MODEL_NAME, MAIA_CACHE_DIR),
            StockfishScorer(stockfish_path),
        )
        create_app(game, pgn_analysis).run(
            host="127.0.0.1",
            port=5000,
            threaded=False,
        )
    finally:
        try:
            if pgn_analysis is not None:
                pgn_analysis.close()
        finally:
            if game is not None:
                game.close()


if __name__ == "__main__":
    main()
