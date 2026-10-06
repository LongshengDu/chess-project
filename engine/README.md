# Chess engine adapters

`engine/` provides Maia inference, Stockfish scoring, downloadable runtime
assets, and UCI process management. It supplies engine operations to the web
backend and coach. Game analysis, search orchestration, accuracy, and rating
calculations belong to [analysis/](../analysis/README.md).

## Modules and interfaces

| Module | Responsibility |
| --- | --- |
| `maia.py` | `MaiaPolicy`: history-aware move probabilities and batched position/rating inference |
| `stockfish.py` | `StockfishScorer`: position/candidate scoring and access to the analysis worker pool |
| `stockfish_pool.py` | `StockfishPool`: persistent workers, leases, and thread/hash allocation per worker |
| `uci.py` | Stockfish executable resolution, startup, cleanup, and time-unit conversion |
| `assets_maia.py` | Resolve the configured checkpoint or populate the official Maia model cache |
| `assets_stockfish.py` | Resolve, verify, download and atomically install official Stockfish binaries |
| `assets.py` | Asset preparation entry point: `python -m engine.assets` |
| `settings.py` | Read root YAML for engine consumers and resolve configured paths |

The `assets_*` modules form the download and installation group; the
`stockfish_*` modules extend the Stockfish adapter with worker management.
`maia.py`, `stockfish.py` and `uci.py` retain their direct engine/protocol names.

Maia inference runs in process. Stockfish runs as native UCI processes.
Applications own their adapters and must close them on shutdown; sharing code
and cached files does not share running processes between applications.

## Setup and configuration

From the repository root, prepare assets and launch the web application:

```powershell
python -m engine.assets
python backend/app.py
```

Each component reads [config.yaml](../config.yaml) through its own settings
module. Engine adapters use `engine/settings.py`; analysis uses
`analysis/settings.py`. There is no global configuration loader.

| Section | Engine-related settings |
| --- | --- |
| `MAIA` | Model/checkpoint, device, inference batch size, sampling temperature, and caches |
| `STOCKFISH` | Executable, asset/download configuration, cache, and startup timeout |
| `ANALYSIS` | Worker count, threads and hash per worker, search limits, and result cache |

Model and executable caches default to `engine/.cache/maia3` and
`engine/.cache/stockfish`. These differ from shared analysis evidence in `ANALYSIS.CACHE_DIR`
and saved web studies in `SERVER.STORAGE.DATABASE`. Changing a report/output
directory does not relocate engine assets.

## Maia batching and history

`MAIA.DEVICE: auto` selects CUDA when available. `MAIA.BATCH_SIZE` limits
position/rating rows per inference batch; its current default is 128. A batch can
evaluate multiple positions and multiple player/opponent rating pairs together.

`MaiaPolicy.batch_evaluate(..., boards=...)` accepts a FEN-to-board mapping or an
aligned board list. Use the list when identical FENs arise through different
histories. Preparation-cache keys retain the complete game history, while model
tokenization uses its recent history window. `MAIA.POSITION_CACHE_ENTRIES` limits
cached CPU preparation. Game-wide batching and reusable policy caches are
coordinated by `analysis.engine_session` and `analysis.player_rating.policies`.
Single-position `probabilities()` uses this same inference path. The adapter
serializes model loading, preparation-cache access and inference internally, so
sharing it between callers cannot race model initialization or cache eviction.

## Stockfish workers and search limits

`ANALYSIS.STOCKFISH_WORKERS` controls independent searches;
`ANALYSIS.STOCKFISH_THREADS_PER_WORKER` controls threads within each engine.
Current defaults are four workers × four threads, giving 16 search threads.
`ANALYSIS.STOCKFISH_HASH_MB_PER_WORKER: 128` allocates 128 MB to each engine.
The default four-worker pool therefore uses 512 MB of hash. Increasing the
worker count increases total hash memory without reducing memory per worker.
A separate direct-scoring process, when started, also receives 128 MB; its hash
is additional to the pool allocation. Engine process overhead is additional to
these hash sizes.

The coach CLI can override this setting with `--hash-mb-per-worker`; the earlier
`--hash-mb` spelling remains an alias with the same per-worker meaning.

Workers start lazily and are leased for individual searches. Cancellation and
errors release leases; closing the pool shuts down its processes. Whole-game
search scheduling belongs to `analysis.stockfish_search`; continuation exploration belongs
to `analysis.stockfish_exploration`. Both consume the common `ANALYSIS` budgets. Depth and
time limits constrain search, so time-limited results can vary between runs even
when configuration is unchanged.

Pool shutdown rejects new leases and hash resets. Concurrent hash resets collect
workers one operation at a time. `StockfishScorer` serializes its separate direct
scoring worker; concurrent searches use `analysis_pool`. Closing the scorer is
idempotent and final: create a new adapter to start another session.

## Tests and measurements

Engine regression tests and the benchmark utility live in `tests/engine/`:

```powershell
python -m unittest discover -s tests/engine -t . -p "test_*.py"
python -m tests.engine.benchmark_engines --help
```

The benchmark is a development tool, separate from application startup. For
whole-game measurements, see the [analysis guide](../analysis/README.md#entry-points)
and [whole-game timing checks](../tests/README.md#single-game-speed).
Player-rating calculations are documented in the
[shared-curve technical paper](../docs/bayesian_shared_curve.md) and the
[estimator interface](../analysis/README.md#player-rating-estimator-interface).
