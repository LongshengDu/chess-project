import subprocess
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from flask import Flask, request

if __package__:
    from .analysis import MaiaPolicy, PgnAnalysisApi, StockfishScorer
else:
    from analysis import MaiaPolicy, PgnAnalysisApi, StockfishScorer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIA_CACHE_DIR = PROJECT_ROOT / "backend" / ".cache" / "maia3"
MODEL_NAME = "maia3-5m"
STOCKFISH_PATH = (
    PROJECT_ROOT / "deps" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
)

ENGINE_COMMAND = [
    "maia3-uci",
    "--model",
    MODEL_NAME,
    "--cache-dir",
    str(MAIA_CACHE_DIR),
    "--local-files-only",
    "--use-uci-history",
    "--elo",
    "1500",
]

MODEL_CACHE_COMMAND = [
    "maia3-cache",
    "--model",
    MODEL_NAME,
    "--cache-dir",
    str(MAIA_CACHE_DIR),
]


def ensure_model_cached() -> None:
    MAIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if any(MAIA_CACHE_DIR.rglob(f"{MODEL_NAME}.pt")):
        return
    subprocess.run(MODEL_CACHE_COMMAND, check=True)

PIECES = [
    (chess.QUEEN, "queen", 9),
    (chess.ROOK, "rook", 5),
    (chess.BISHOP, "bishop", 3),
    (chess.KNIGHT, "knight", 3),
    (chess.PAWN, "pawn", 1),
]


class ChessApi:
    def __init__(self, engine: chess.engine.SimpleEngine) -> None:
        self.board = chess.Board()
        self._engine = engine
        self._resigned = False
        self._claimed_draw: str | None = None

    def state(self) -> dict:
        return self._state()

    def move(self, uci: str) -> dict:
        if self.board.turn != chess.WHITE or self._game_over():
            raise ValueError("It is not your turn")

        try:
            move = chess.Move.from_uci(uci)
        except ValueError as error:
            raise ValueError("Invalid UCI move") from error

        if move not in self.board.legal_moves:
            raise ValueError("Illegal move")

        sounds = [self._push(move)]
        if not self._game_over():
            try:
                reply = self._engine.play(self.board, chess.engine.Limit(nodes=1)).move
                if reply is None or reply not in self.board.legal_moves:
                    raise RuntimeError("Maia returned no legal move")
                sounds.append(self._push(reply))
            except Exception:
                self.board.pop()
                raise

        return self._state(sounds)

    def action(self, name: str) -> dict:
        if name == "takeback":
            if self._game_over() or len(self.board.move_stack) < 2:
                raise ValueError("No move pair to take back")
            self.board.pop()
            self.board.pop()
        elif name == "claim-draw":
            if self._game_over() or not self.board.can_claim_draw():
                raise ValueError("A draw cannot be claimed")
            self._claimed_draw = (
                "fifty-move rule"
                if self.board.can_claim_fifty_moves()
                else "threefold repetition"
            )
        elif name == "resign":
            if self._game_over():
                raise ValueError("The game is already over")
            self._resigned = True
        else:
            raise ValueError("Unknown action")
        return self._state()

    def new_game(self) -> dict:
        self.board.reset()
        self._resigned = False
        self._claimed_draw = None
        return self._state()

    def _close(self) -> None:
        try:
            self._engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

    def _game_over(self) -> bool:
        return bool(
            self._resigned
            or self._claimed_draw
            or self.board.is_game_over(claim_draw=False)
        )

    def _push(self, move: chess.Move) -> str:
        captured = self.board.is_capture(move)
        self.board.push(move)
        if self.board.is_checkmate():
            return "checkmate"
        if self.board.is_check():
            return "check"
        return "capture" if captured else "move"

    @staticmethod
    def _material(board: chess.Board) -> dict:
        material = {
            "white": {"pieces": [], "score": 0},
            "black": {"pieces": [], "score": 0},
        }
        score = 0
        for piece_type, role, value in PIECES:
            difference = len(board.pieces(piece_type, chess.WHITE)) - len(
                board.pieces(piece_type, chess.BLACK)
            )
            side = "white" if difference > 0 else "black"
            material[side]["pieces"].extend([role] * abs(difference))
            score += difference * value
        material["white"]["score"] = max(score, 0)
        material["black"]["score"] = max(-score, 0)
        return material

    def _result_status(self) -> tuple[str | None, str, chess.Outcome | None]:
        if self._resigned:
            return "0-1", "You resigned — Black wins", None
        if self._claimed_draw:
            return "1/2-1/2", f"Draw — {self._claimed_draw}", None

        outcome = self.board.outcome(claim_draw=False)
        if outcome is None:
            if self.board.is_check():
                return None, "Check", None
            status = "Your turn" if self.board.turn == chess.WHITE else "Maia is thinking..."
            return None, status, None

        if outcome.termination == chess.Termination.CHECKMATE:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            return outcome.result(), f"Checkmate — {winner} wins", outcome

        reasons = {
            chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
            chess.Termination.SEVENTYFIVE_MOVES: "seventy-five-move rule",
            chess.Termination.FIVEFOLD_REPETITION: "fivefold repetition",
        }
        reason = reasons.get(outcome.termination, "draw")
        return outcome.result(), f"Draw — {reason}", outcome

    def _pgn(self, result: str | None) -> str:
        game = chess.pgn.Game.from_board(self.board)
        game.headers.update(
            Event="Maia Local Chess",
            White="You",
            Black="Maia 1500",
            Result=result or "*",
        )
        return str(game)

    def _state(self, sounds: list[str] | None = None) -> dict:
        result, status, outcome = self._result_status()
        game_over = bool(self._resigned or self._claimed_draw or outcome)
        dests: dict[str, list[str]] = {}
        promotions: dict[str, list[str]] = {}
        replay = self.board.root()
        moves: list[str] = []
        history = [
            {
                "fen": replay.board_fen(),
                "turn": "white",
                "lastMove": None,
                "check": False,
                "material": self._material(replay),
                "sound": None,
            }
        ]

        for move in self.board.move_stack:
            moves.append(replay.san(move))
            captured = replay.is_capture(move)
            uci = move.uci()
            replay.push(move)
            sound = (
                "checkmate"
                if replay.is_checkmate()
                else "check"
                if replay.is_check()
                else "capture"
                if captured
                else "move"
            )
            history.append(
                {
                    "fen": replay.board_fen(),
                    "turn": "white" if replay.turn == chess.WHITE else "black",
                    "lastMove": [uci[:2], uci[2:4]],
                    "check": replay.is_check(),
                    "material": self._material(replay),
                    "sound": sound,
                }
            )

        if not game_over:
            for move in self.board.legal_moves:
                origin = chess.square_name(move.from_square)
                destination = chess.square_name(move.to_square)
                destinations = dests.setdefault(origin, [])
                if destination not in destinations:
                    destinations.append(destination)
                if move.promotion:
                    promotions.setdefault(origin + destination, []).append(
                        chess.piece_symbol(move.promotion)
                    )

        last_move = self.board.peek().uci() if self.board.move_stack else None
        return {
            "fen": self.board.board_fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "lastMove": [last_move[:2], last_move[2:4]] if last_move else None,
            "check": self.board.is_check(),
            "material": self._material(self.board),
            "sound": history[-1]["sound"],
            "gameOver": game_over,
            "result": result,
            "status": status,
            "moves": moves,
            "history": history,
            "dests": dests,
            "promotions": promotions,
            "canTakeback": not game_over and len(self.board.move_stack) >= 2,
            "canClaimDraw": not game_over and self.board.can_claim_draw(),
            "canResign": not game_over,
            "sounds": sounds or [],
            "pgn": self._pgn(result),
        }


def main() -> None:
    dist = PROJECT_ROOT / "web" / "dist"
    if not (dist / "index.html").exists():
        raise SystemExit("Frontend not built. Run: cd web && pnpm install && pnpm run build")

    ensure_model_cached()

    server = Flask(__name__, static_folder=str(dist), static_url_path="")
    popen_args = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
        else {}
    )
    game = ChessApi(chess.engine.SimpleEngine.popen_uci(ENGINE_COMMAND, **popen_args))
    pgn_analysis = PgnAnalysisApi(
        MaiaPolicy(MODEL_NAME, MAIA_CACHE_DIR),
        StockfishScorer(STOCKFISH_PATH),
    )

    @server.get("/")
    def index():
        return server.send_static_file("index.html")

    @server.get("/api/state")
    def state():
        return game.state()

    @server.post("/api/move")
    def move():
        data = request.get_json(silent=True) or {}
        return game.move(str(data.get("uci", "")))

    @server.post("/api/action")
    def action():
        data = request.get_json(silent=True) or {}
        return game.action(str(data.get("name", "")))

    @server.post("/api/new-game")
    def new_game():
        return game.new_game()

    @server.post("/api/analysis/pgn")
    def import_pgn():
        data = request.get_json(silent=True) or {}
        return pgn_analysis.import_pgn(str(data.get("pgn", "")))

    @server.post("/api/analysis/position")
    def analyse_position():
        data = request.get_json(silent=True) or {}
        try:
            ply = int(data.get("ply", -1))
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid PGN position") from error
        variation_moves = data.get("moves", [])
        if not isinstance(variation_moves, list):
            raise ValueError("Invalid variation path")
        return pgn_analysis.analyse_position(ply, [str(move) for move in variation_moves])

    @server.post("/api/analysis/move")
    def analysis_move():
        data = request.get_json(silent=True) or {}
        try:
            ply = int(data.get("ply", -1))
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid PGN position") from error
        variation_moves = data.get("moves", [])
        if not isinstance(variation_moves, list):
            raise ValueError("Invalid variation path")
        return pgn_analysis.play_variation_move(
            ply,
            [str(move) for move in variation_moves],
            str(data.get("uci", "")),
        )

    @server.errorhandler(ValueError)
    def invalid_action(error):
        return {"error": str(error)}, 400

    try:
        server.run(host="127.0.0.1", port=5000, threaded=False)
    finally:
        try:
            pgn_analysis.close()
        finally:
            game._close()


if __name__ == "__main__":
    main()
