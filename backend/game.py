from __future__ import annotations

import chess
import chess.engine
import chess.pgn

if __package__:
    from .chess_utils import material, move_sounds
else:  # Support running backend/app.py directly.
    from chess_utils import material, move_sounds


class ChessApi:
    def __init__(
        self,
        engine: chess.engine.SimpleEngine,
        model_name: str,
        model_elo: int,
    ) -> None:
        self.board = chess.Board()
        self._engine = engine
        self._model_name = model_name
        self._model_elo = model_elo
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
        return self._state(self._push(move))

    def reply(self) -> dict:
        if self.board.turn != chess.BLACK or self._game_over():
            raise ValueError("Maia has no move to play")

        move = self._engine.play(self.board, chess.engine.Limit(nodes=1)).move
        if move is None or move not in self.board.legal_moves:
            raise RuntimeError("Maia returned no legal move")
        return self._state(self._push(move))

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

    def close(self) -> None:
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

    def _push(self, move: chess.Move) -> list[str]:
        captured = self.board.is_capture(move)
        self.board.push(move)
        return move_sounds(self.board, captured)

    def _result_status(self) -> tuple[str | None, str, chess.Outcome | None]:
        if self._resigned:
            return "0-1", "You resigned — Black wins", None
        if self._claimed_draw:
            return "1/2-1/2", f"Draw — {self._claimed_draw}", None

        outcome = self.board.outcome(claim_draw=False)
        if outcome is None:
            if self.board.is_check():
                return None, "Check", None
            status = (
                "Your turn" if self.board.turn == chess.WHITE else "Maia is thinking..."
            )
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
            Black=f"{self._model_name} {self._model_elo}",
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
                "material": material(replay),
                "moveSounds": [],
            }
        ]

        for move in self.board.move_stack:
            moves.append(replay.san(move))
            captured = replay.is_capture(move)
            uci = move.uci()
            replay.push(move)
            history.append(
                {
                    "fen": replay.board_fen(),
                    "turn": "white" if replay.turn == chess.WHITE else "black",
                    "lastMove": [uci[:2], uci[2:4]],
                    "check": replay.is_check(),
                    "material": material(replay),
                    "moveSounds": move_sounds(replay, captured),
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
            "modelName": self._model_name,
            "modelElo": self._model_elo,
            "fen": self.board.board_fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "lastMove": [last_move[:2], last_move[2:4]] if last_move else None,
            "check": self.board.is_check(),
            "material": material(self.board),
            "moveSounds": history[-1]["moveSounds"],
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
