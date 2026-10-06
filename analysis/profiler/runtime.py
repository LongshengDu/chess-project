"""Runtime profiler: record execution timing for an uncached whole-game run."""
from __future__ import annotations

import json
import platform
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch


class RuntimeProfiler:
    def __init__(self, directory: Path):
        self.directory = directory
        self.lock = threading.RLock()
        self.active = False
        self.data = None
        self.current_search = None

    def start(self, configuration, maia, stockfish, strategy):
        with self.lock:
            if self.active:
                raise ValueError("A runtime profiler is already running")
            self.directory.mkdir(parents=True, exist_ok=True)
            self.run_id = uuid.uuid4().hex
            self.started = time.perf_counter()
            self.data = {"run_id": self.run_id, "complete": False,
                         "started_utc": datetime.now(timezone.utc).isoformat(),
                         "configuration": {**configuration, "strategy": strategy,
                             "stockfish_workers": stockfish.analysis_pool.workers,
                             "stockfish_threads_per_worker": stockfish.analysis_pool.threads_per_worker,
                             "stockfish_total_threads": stockfish.analysis_pool.total_threads,
                             "stockfish_hash_mb_per_worker": stockfish.analysis_pool.hash_mb,
                             "stockfish_total_hash_mb": stockfish.analysis_pool.workers * stockfish.analysis_pool.hash_mb,
                             "model": maia._engine.cfg.model, "device": maia._engine.cfg.device,
                             "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                             "platform": platform.platform(), "processor": platform.processor()},
                         "events": [], "positions": []}
            self.active = True
            self.current_search = None
            self.event_file = self.directory / f"{self.run_id}.jsonl"
            self.record("started", configuration=self.data["configuration"])
            self.snapshot()
            return {"run_id": self.run_id}

    def record(self, kind, **values):
        with self.lock:
            if not self.active:
                return
            event = {"kind": kind, "elapsed_ms": (time.perf_counter() - self.started) * 1000, **values}
            self.data["events"].append(event)
            if kind == "stockfish_search":
                self.current_search = event if event["status"] == "started" else None
            with self.event_file.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event) + "\n")

    def position(self, data):
        with self.lock:
            if not self.active or data.get("run_id") != self.run_id:
                raise ValueError("Unknown profiling run")
            if any(row["ply"] == data["ply"] for row in self.data["positions"]):
                raise ValueError("Position already recorded")
            self.data["positions"].append(data)
            self.record("browser_position", **data)
            self.snapshot()

    def finish(self, data):
        if data.get('cancelled'):
            return self.abort(data)
        with self.lock:
            if not self.active or data.get("run_id") != self.run_id:
                raise ValueError("Unknown profiling run")
            if len(self.data["positions"]) != self.data["configuration"]["total_positions"]:
                raise ValueError("Not all positions were analyzed")
            self.data.update({"complete": True, "browser": data,
                              "server_wall_ms": (time.perf_counter() - self.started) * 1000,
                              "finished_utc": datetime.now(timezone.utc).isoformat()})
            self.record("finished", browser=data)
            self.active = False
            self.snapshot()

    def abort(self, data):
        """Close an interrupted measurement without presenting it as complete."""
        with self.lock:
            if not self.active or data.get('run_id') != self.run_id:
                raise ValueError('Unknown profiling run')
            self.data.update(complete=False, interrupted=data,
                             server_wall_ms=(time.perf_counter() - self.started) * 1000)
            self.record('aborted', **data)
            self.active = False
            self.current_search = None
            self.snapshot()

    def snapshot(self):
        temporary = self.directory / "runtime-profiler.tmp.json"
        temporary.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "runtime-profiler.json")

    def status(self):
        with self.lock:
            return {"enabled": True, "active": self.active,
                    "complete": self.data["complete"] if self.data else False,
                    "positions_completed": len(self.data["positions"]) if self.data else 0,
                    "current_search": self.current_search,
                    "elapsed_seconds": time.perf_counter() - self.started if self.active else None}
