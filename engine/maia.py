"""Official Maia model adapter, legal policy, values, and complete game history."""
from __future__ import annotations

import time
import threading
from collections import OrderedDict, deque
from pathlib import Path

import chess
import torch
from maia3.dataset import get_legal_moves_mask, tokenize_board
from maia3.uci import Maia3UCIEngine, parse_args
from maia3.utils import mirror_move
from torch.amp import autocast

from engine.settings import CONFIG


class MaiaPolicy:
    """Lazy direct Maia model access for exact legal-move probabilities."""

    def __init__(self, model: str, cache_dir: Path, device: str | None = None,
                 *, checkpoint: str | Path | None = None) -> None:
        device = CONFIG['MAIA']['DEVICE'] if device is None else device
        checkpoint = CONFIG['MAIA']['CHECKPOINT'] if checkpoint is None else checkpoint
        config = parse_args(
            [
                "--model",
                model,
                "--cache-dir",
                str(cache_dir),
                "--local-files-only",
                "--use-uci-history",
                "--elo",
                str(CONFIG['MAIA']['PLAYER_RATING']),
            ] + (["--device", device] if device and device != "auto" else [])
        )
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Install GPU support with uv sync --extra cuda, "
                             "then launch with uv run --extra cuda; or use --device cpu.")
        if checkpoint is not None:
            config.checkpoint_path = str(checkpoint)
        self._engine = Maia3UCIEngine(config)
        self._inference_lock = threading.RLock()
        self._prepared_cache = OrderedDict()

    def load(self) -> None:
        """Eagerly load once when an application requires startup readiness."""
        with self._inference_lock:
            self._engine.ensure_model_loaded()

    @property
    def model_signature(self):
        """Stable inference identity for shared analysis caches."""
        with self._inference_lock:
            config = self._engine.cfg
            if config.checkpoint_path is None:
                # Maia resolves its official cached checkpoint during the first
                # load. Reuse that shared model rather than guessing a filename.
                self._engine.ensure_model_loaded()
            checkpoint = Path(config.checkpoint_path).resolve()
            return {'maia': [str(checkpoint), checkpoint.stat().st_mtime_ns],
                    'history_window': config.history, 'device': config.device}

    @staticmethod
    def _history(board: chess.Board, length: int) -> deque:
        # The network only consumes its fixed history window. Walking back that
        # window is equivalent to replaying the whole PGN and dropping its prefix.
        replay = board.copy(stack=max(0, length-1))
        positions = [tokenize_board(replay)]
        while replay.move_stack:
            replay.pop()
            positions.append(tokenize_board(replay))
        return deque(reversed(positions), maxlen=length)

    def probabilities(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        if board.is_game_over(claim_draw=False):
            return []

        rating = CONFIG['MAIA']['PLAYER_RATING']
        result = self.batch_evaluate([board.fen()], [rating], [rating], boards=[board])[0]
        return [(chess.Move.from_uci(move), probability)
                for move, probability in result['policy'].items()]

    def batch_evaluate(self, fens: list[str], ratings: list[int], opponents: list[int],
                       timings: dict | None = None, *, boards=None) -> list[dict]:
        """Return legal policies and White expected scores; optionally fill timings in ms."""
        with self._inference_lock:
            return self._batch_evaluate(fens, ratings, opponents, timings, boards=boards)

    def _batch_evaluate(self, fens, ratings, opponents, timings=None, *, boards=None):
        engine = self._engine
        if not fens or len(fens) != len(ratings) or len(fens) != len(opponents):
            raise ValueError('Supply nonempty matching position and rating arrays')
        if isinstance(boards, (list, tuple)) and len(boards) != len(fens):
            raise ValueError('Supply one history board per requested position')
        cap = CONFIG['MAIA']['BATCH_SIZE']
        if type(cap) is not int or cap < 1:
            raise ValueError('MAIA.BATCH_SIZE must be a positive integer')
        if len(fens) > cap:
            result = []
            for offset in range(0, len(fens), cap):
                part = {} if timings is not None else None
                result.extend(self._batch_evaluate(fens[offset:offset+cap], ratings[offset:offset+cap],
                    opponents[offset:offset+cap], part,
                    boards=boards[offset:offset+cap] if isinstance(boards, (list,tuple)) else boards))
                if timings is not None:
                    for key,value in part.items():
                        timings[key] = timings.get(key,0.) + value
            return result
        tick = time.perf_counter()
        def checkpoint(name):
            nonlocal tick
            if timings is not None:
                if engine.cfg.device.startswith("cuda"):
                    torch.cuda.synchronize(engine.cfg.device)
                now = time.perf_counter()
                timings[name] = (now - tick) * 1000
                tick = now
        engine.ensure_model_loaded()
        checkpoint("model_load_ms")
        # Rating sweeps repeat each FEN 21 times. Tokenize and enumerate its legal
        # moves once, and transfer/normalize the whole batch in one operation.
        prepared = []
        for i,fen in enumerate(fens):
            board = (boards[i] if isinstance(boards,(list,tuple)) else boards[fen]) if boards is not None else chess.Board(fen)
            if board.fen() != chess.Board(fen).fen():
                raise ValueError("Maia history does not match the requested position")
            key = (engine.cfg.history,fen,board.root().fen(),tuple(board.move_stack))
            if key in self._prepared_cache:
                self._prepared_cache.move_to_end(key)
                prepared.append(self._prepared_cache[key])
                continue
            indices = {move.uci(): engine.all_moves_dict[
                move.uci() if board.turn else mirror_move(move.uci())
            ] for move in board.legal_moves}
            item = (board.copy(stack=True), engine._tokens_from_history(self._history(board, engine.cfg.history)),
                             get_legal_moves_mask(board, engine.all_moves_dict), indices)
            self._prepared_cache[key] = item
            if len(self._prepared_cache) > CONFIG['MAIA']['POSITION_CACHE_ENTRIES']:
                self._prepared_cache.popitem(last=False)
            prepared.append(item)
        tokens = torch.stack([p[1] for p in prepared]).to(engine.cfg.device)
        masks = torch.stack([p[2] for p in prepared]).to(engine.cfg.device)
        checkpoint("prepare_ms")
        with torch.inference_mode(), autocast(
            "cuda", enabled=engine.cfg.use_amp and engine.cfg.device.startswith("cuda")
        ):
            moves, values, _ = engine.model(
                tokens,
                torch.tensor(ratings, dtype=torch.long, device=engine.cfg.device),
                torch.tensor(opponents, dtype=torch.long, device=engine.cfg.device),
            )
            checkpoint("forward_ms")
            values = torch.softmax(values.float(), dim=-1).cpu().numpy()
            probabilities = torch.softmax(moves.float().masked_fill(~masks, float("-inf")), dim=-1).cpu().numpy()
            checkpoint("normalize_transfer_ms")
        results = []
        for i, fen in enumerate(fens):
            board, _, _, indices = prepared[i]
            outcome = board.outcome()
            if outcome:
                results.append({"policy": {}, "value": .5 if outcome.winner is None else float(outcome.winner)})
                continue
            policy = {move: float(probabilities[i, index]) for move, index in indices.items()}
            value = float(values[i, 2] + .5 * values[i, 1])
            results.append({"policy": dict(sorted(policy.items(), key=lambda item: -item[1])),
                            "value": value if board.turn else 1 - value})
        checkpoint("format_ms")
        return results
