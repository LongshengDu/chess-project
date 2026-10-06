from __future__ import annotations

import io
from pathlib import Path
import chess
import chess.pgn

from engine.maia import MaiaPolicy
from engine.stockfish import StockfishScorer
from analysis.settings import CONFIG
from analysis.game.history import replay
from analysis.position_display import material, move_sounds


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
        "fullFen": board.fen(en_passant="fen"),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "lastMove": (
            [last_move.uci()[:2], last_move.uci()[2:4]] if last_move else None
        ),
        "check": board.is_check(),
        "material": material(board),
        "moveSounds": move_sounds,
        "ply": board.ply(),
        "dests": dests,
        "promotions": promotions,
        "gameOver": board.is_game_over(claim_draw=False),
        "status": status,
    }


class PgnAnalysisApi:
    def __init__(self, maia: MaiaPolicy, stockfish: StockfishScorer) -> None:
        self._maia = maia
        self._stockfish = stockfish
        self._boards: list[chess.Board] = []
        self._moves: list[chess.Move] = []
        self._game_state: dict | None = None
        self._analysis_cache: dict[tuple[int, tuple[str, ...]], dict] = {}

    def _board_at(
        self, ply: int, variation_moves: list[str] | None = None
    ) -> chess.Board:
        if self._game_state is None:
            raise ValueError("Import a PGN or FEN first")
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

        stream = io.StringIO(pgn)
        game = chess.pgn.read_game(stream)
        if game is None:
            raise ValueError("No chess game was found in the PGN")
        if game.errors:
            raise ValueError(f"Invalid PGN: {game.errors[0]}")
        if not game.variations and not pgn.lstrip().startswith("["):
            raise ValueError("No moves or PGN headers were found")
        if chess.pgn.read_game(stream) is not None:
            raise ValueError("Import one PGN game at a time")
        return self._import_game(game, "pgn")

    def import_fen(self, fen: str) -> dict:
        if len(fen.split()) != 6:
            raise ValueError("Enter a complete FEN with all six fields")
        try:
            board = chess.Board(fen.strip())
        except ValueError as error:
            raise ValueError(f"Invalid FEN: {error}") from error
        if not board.is_valid():
            raise ValueError("Invalid FEN position")
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "White"
        game.headers["Black"] = "Black"
        game.headers["Event"] = "Custom FEN position"
        return self._import_game(game, "fen")

    def _import_game(self, game: chess.pgn.Game, source: str) -> dict:

        board = game.board()
        if board.uci_variant != "chess" or board.chess960:
            raise ValueError("Only standard chess positions are supported")
        if not board.is_valid():
            raise ValueError("Invalid starting position")
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
            positions.append(_position_state(board, move, move_sounds(board, captured)))

        # Keep side lines with paths understood by the existing analysis endpoints.
        variations = []
        pending = []
        main_node = game
        for ply, main_board in enumerate(boards):
            pending.extend(
                (node, main_board, ply, []) for node in reversed(main_node.variations[1:])
            )
            if main_node.variations:
                main_node = main_node.variations[0]
        while pending:
            node, parent_board, ply, path = pending.pop()
            if len(path) >= 200 or len(variations) >= 4000:
                raise ValueError("PGN variations are too large")
            branch_board = parent_board.copy(stack=True)
            san = branch_board.san(node.move)
            captured = branch_board.is_capture(node.move)
            branch_board.push(node.move)
            next_path = [*path, node.move.uci()]
            variations.append({
                "ply": ply, "moves": next_path, "uci": node.move.uci(), "san": san,
                "position": _position_state(branch_board, node.move, move_sounds(branch_board, captured)),
            })
            pending.extend(
                (child, branch_board, ply, next_path)
                for child in reversed(node.variations)
            )

        headers = game.headers
        white = headers.get("White", "White")
        black = headers.get("Black", "Black")
        white = "White" if white in ("", "?") else white
        black = "Black" if black in ("", "?") else black
        event = headers.get("Event", "Imported PGN")
        event = "Imported PGN" if event in ("", "?") else event
        result = headers.get("Result", "*")
        self._boards = boards
        self._moves = moves
        self._analysis_cache.clear()
        self._game_state = {
            "source": source,
            "headers": dict(headers),
            "plain_pgn": game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)),
            "title": f"{white} – {black}",
            "subtitle": event,
            "white": white,
            "black": black,
            "result": result,
            "moves": san_moves,
            "ucis": uci_moves,
            "positions": positions,
            "variations": variations,
        }
        return self._game_state

    def analyse_position(
        self,
        ply: int,
        variation_moves: list[str] | None = None,
    ) -> dict:
        variation_path = tuple(variation_moves or ())
        cache_key = (ply, variation_path)
        cached = self._analysis_cache.get(cache_key)
        if cached is not None:
            return cached

        board = self._board_at(ply, list(variation_path))
        position_score, position_depth = self._stockfish.evaluate(board)
        distribution = self._maia.probabilities(board)
        selected = distribution[:10]
        played = (
            self._moves[ply]
            if distribution and not variation_moves and ply < len(self._moves)
            else None
        )
        if played is not None and all(move != played for move, _ in selected):
            played_probability = next(
                (probability for move, probability in distribution if move == played),
                0.0,
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
        result = {
            "ply": ply,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "elo": CONFIG['MAIA']['PLAYER_RATING'],
            "legalMoveCount": len(distribution),
            "moves": moves,
            "otherProbability": max(0.0, 1.0 - displayed_probability) if distribution else 0.0,
            "stockfish": position_score,
            "stockfishDepth": position_depth if position_depth is not None else depth,
            "playedMove": played.uci() if played else None,
        }
        self._analysis_cache[cache_key] = result
        return result

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
            "position": _position_state(board, move, move_sounds(board, captured)),
        }

def load_game(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    stream = io.StringIO(text)
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or not list(game.mainline_moves()):
        raise ValueError('Supply a valid PGN containing at least one legal played move.')
    if game.headers.get('Variant', 'Standard') not in ('Standard', 'Chess') or game.board().chess960:
        raise ValueError('This coach currently supports standard chess only.')
    if not game.board().is_valid():
        raise ValueError('The PGN starting position is invalid.')
    if chess.pgn.read_game(stream) is not None:
        raise ValueError('Supply one game per PGN file.')
    replay(game.board().fen(), [m.uci() for m in game.mainline_moves()])
    return game
