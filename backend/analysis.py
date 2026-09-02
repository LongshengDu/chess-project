from __future__ import annotations

import io
from collections import deque
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch
from maia3.dataset import get_legal_moves_mask, tokenize_board
from maia3.uci import Maia3UCIEngine, parse_args
from maia3.utils import mirror_move
from torch.amp import autocast


MAIA_ELO = 1500
TOP_HUMAN_MOVES = 10
STOCKFISH_DEPTH = 12


def _move_sounds(board: chess.Board, captured: bool) -> list[str]:
    sounds = ["capture" if captured else "move"]
    if board.is_checkmate():
        sounds.append("checkmate")
    elif board.is_check():
        sounds.append("check")
    return sounds


def _material(board: chess.Board) -> dict:
    pieces = [
        (chess.QUEEN, "queen", 9),
        (chess.ROOK, "rook", 5),
        (chess.BISHOP, "bishop", 3),
        (chess.KNIGHT, "knight", 3),
        (chess.PAWN, "pawn", 1),
    ]
    material = {
        "white": {"pieces": [], "score": 0},
        "black": {"pieces": [], "score": 0},
    }
    score = 0
    for piece_type, role, value in pieces:
        difference = len(board.pieces(piece_type, chess.WHITE)) - len(
            board.pieces(piece_type, chess.BLACK)
        )
        side = "white" if difference > 0 else "black"
        material[side]["pieces"].extend([role] * abs(difference))
        score += difference * value
    material["white"]["score"] = max(score, 0)
    material["black"]["score"] = max(-score, 0)
    return material


def _position_state(
    board: chess.Board,
    last_move: chess.Move | None,
    move_sounds: list[str],
) -> dict:
    dests: dict[str, list[str]] = {}
    promotions: dict[str, list[str]] = {}
    if not board.is_game_over(claim_draw=False):
        for move in board.legal_moves:
            origin = chess.square_name(move.from_square)
            destination = chess.square_name(move.to_square)
            destinations = dests.setdefault(origin, [])
            if destination not in destinations:
                destinations.append(destination)
            if move.promotion:
                promotions.setdefault(origin + destination, []).append(
                    chess.piece_symbol(move.promotion)
                )

    if board.is_checkmate():
        status = "Checkmate"
    elif board.is_stalemate():
        status = "Stalemate"
    elif board.is_game_over(claim_draw=False):
        status = "Game over"
    elif board.is_check():
        status = "Check"
    else:
        status = "White to move" if board.turn == chess.WHITE else "Black to move"

    return {
        "fen": board.board_fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "lastMove": (
            [last_move.uci()[:2], last_move.uci()[2:4]] if last_move else None
        ),
        "check": board.is_check(),
        "material": _material(board),
        "moveSounds": move_sounds,
        "ply": board.ply(),
        "dests": dests,
        "promotions": promotions,
        "gameOver": board.is_game_over(claim_draw=False),
        "status": status,
    }


class MaiaPolicy:
    """Lazy direct Maia model access for exact legal-move probabilities."""

    def __init__(self, model: str, cache_dir: Path) -> None:
        config = parse_args(
            [
                "--model",
                model,
                "--cache-dir",
                str(cache_dir),
                "--local-files-only",
                "--use-uci-history",
                "--elo",
                str(MAIA_ELO),
            ]
        )
        self._engine = Maia3UCIEngine(config)

    @staticmethod
    def _history(board: chess.Board, length: int) -> deque:
        replay = board.root()
        history = deque([tokenize_board(replay)], maxlen=length)
        for move in board.move_stack:
            replay.push(move)
            history.append(tokenize_board(replay))
        return history

    def probabilities(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        if board.is_game_over(claim_draw=False):
            return []

        engine = self._engine
        engine.ensure_model_loaded()
        engine.board = board.copy(stack=True)
        engine.history = self._history(board, engine.cfg.history)

        legal_mask = get_legal_moves_mask(board, engine.all_moves_dict)
        tokens = engine._tokens_from_history(engine.history).unsqueeze(0).to(
            engine.cfg.device
        )
        elos = torch.tensor([MAIA_ELO], dtype=torch.long, device=engine.cfg.device)
        assert engine.model is not None
        with torch.no_grad(), autocast(
            "cuda",
            enabled=engine.cfg.use_amp and engine.cfg.device.startswith("cuda"),
        ):
            move_logits, _, _ = engine.model(tokens, elos, elos)

        logits = move_logits[0].float()
        logits = logits.masked_fill(~legal_mask.to(engine.cfg.device), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1).cpu()

        result = []
        for move in board.legal_moves:
            policy_uci = move.uci()
            if board.turn == chess.BLACK:
                policy_uci = mirror_move(policy_uci)
            result.append((move, float(probabilities[engine.all_moves_dict[policy_uci]])))
        return sorted(result, key=lambda item: item[1], reverse=True)


class StockfishScorer:
    def __init__(self, executable: Path) -> None:
        self._executable = executable
        self._engine: chess.engine.SimpleEngine | None = None

    def _get_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            if not self._executable.exists():
                raise RuntimeError(f"Stockfish executable not found: {self._executable}")
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self._executable))
            self._engine.configure({"Threads": 8, "Hash": 512})
        return self._engine

    def score(
        self,
        board: chess.Board,
        moves: list[chess.Move],
    ) -> tuple[dict[str, dict], int | None]:
        if not moves:
            return {}, None

        infos = self._get_engine().analyse(
            board,
            chess.engine.Limit(depth=STOCKFISH_DEPTH),
            multipv=len(moves),
            root_moves=moves,
        )
        if isinstance(infos, dict):
            infos = [infos]

        scores: dict[str, dict] = {}
        depths = []
        for info in infos:
            if not info.get("pv"):
                continue
            move = info["pv"][0]
            score = info["score"].pov(board.turn)
            mate = score.mate()
            cp = score.score()
            label = (
                f"{'+' if mate and mate > 0 else '-'}M{abs(mate)}"
                if mate is not None
                else f"{(cp or 0) / 100:+.2f}"
            )
            scores[move.uci()] = {"cp": cp, "mate": mate, "label": label}
            if "depth" in info:
                depths.append(int(info["depth"]))
        return scores, min(depths) if depths else None

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except chess.engine.EngineTerminatedError:
                pass
            self._engine = None


class PgnAnalysisApi:
    def __init__(self, maia: MaiaPolicy, stockfish: StockfishScorer) -> None:
        self._maia = maia
        self._stockfish = stockfish
        self._boards: list[chess.Board] = []
        self._moves: list[chess.Move] = []
        self._game_state: dict | None = None

    def _board_at(self, ply: int, variation_moves: list[str] | None = None) -> chess.Board:
        if self._game_state is None:
            raise ValueError("Import a PGN first")
        if ply < 0 or ply >= len(self._boards):
            raise ValueError("That PGN position does not exist")

        moves = variation_moves or []
        if len(moves) > 200:
            raise ValueError("Variation is too long")
        board = self._boards[ply].copy(stack=True)
        for uci in moves:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError as error:
                raise ValueError("Invalid variation move") from error
            if move not in board.legal_moves:
                raise ValueError(f"Illegal variation move: {uci}")
            board.push(move)
        return board

    def import_pgn(self, pgn: str) -> dict:
        if not pgn.strip():
            raise ValueError("Paste a PGN to analyse")

        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None:
            raise ValueError("No chess game was found in the PGN")
        if game.errors:
            raise ValueError(f"Invalid PGN: {game.errors[0]}")

        board = game.board()
        boards = [board.copy(stack=True)]
        positions = [_position_state(board, None, [])]
        moves: list[chess.Move] = []
        uci_moves: list[str] = []
        san_moves: list[str] = []

        for move in game.mainline_moves():
            san_moves.append(board.san(move))
            captured = board.is_capture(move)
            board.push(move)
            moves.append(move)
            uci_moves.append(move.uci())
            boards.append(board.copy(stack=True))
            positions.append(
                _position_state(board, move, _move_sounds(board, captured))
            )

        headers = game.headers
        white = headers.get("White", "White")
        black = headers.get("Black", "Black")
        result = headers.get("Result", "*")
        self._boards = boards
        self._moves = moves
        self._game_state = {
            "title": f"{white} – {black}",
            "subtitle": headers.get("Event", "Imported PGN"),
            "white": white,
            "black": black,
            "result": result,
            "moves": san_moves,
            "ucis": uci_moves,
            "positions": positions,
        }
        return self._game_state

    def analyse_position(
        self,
        ply: int,
        variation_moves: list[str] | None = None,
    ) -> dict:
        board = self._board_at(ply, variation_moves)
        distribution = self._maia.probabilities(board)
        selected = distribution[:TOP_HUMAN_MOVES]
        played = (
            self._moves[ply]
            if not variation_moves and ply < len(self._moves)
            else None
        )
        if played is not None and all(move != played for move, _ in selected):
            played_probability = next(
                probability for move, probability in distribution if move == played
            )
            selected.append((played, played_probability))

        stockfish_scores, depth = self._stockfish.score(
            board, [move for move, _ in selected]
        )
        ranks = {move.uci(): rank for rank, (move, _) in enumerate(distribution, 1)}
        moves = []
        for move, probability in selected:
            uci = move.uci()
            moves.append(
                {
                    "uci": uci,
                    "san": board.san(move),
                    "probability": probability,
                    "rank": ranks[uci],
                    "played": move == played,
                    "stockfish": stockfish_scores.get(uci),
                }
            )

        displayed_probability = sum(item["probability"] for item in moves)
        return {
            "ply": ply,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "elo": MAIA_ELO,
            "legalMoveCount": len(distribution),
            "moves": moves,
            "otherProbability": max(0.0, 1.0 - displayed_probability),
            "stockfishDepth": depth,
            "playedMove": played.uci() if played else None,
        }

    def play_variation_move(
        self,
        ply: int,
        variation_moves: list[str],
        uci: str,
    ) -> dict:
        board = self._board_at(ply, variation_moves)
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as error:
            raise ValueError("Invalid UCI move") from error
        if move not in board.legal_moves:
            raise ValueError("Illegal move in this variation")

        san = board.san(move)
        captured = board.is_capture(move)
        board.push(move)
        return {
            "uci": move.uci(),
            "san": san,
            "position": _position_state(
                board, move, _move_sounds(board, captured)
            ),
        }

    def close(self) -> None:
        self._stockfish.close()
