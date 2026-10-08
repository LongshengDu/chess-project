# Chess engine adapters

`engine/` provides Maia inference, Stockfish adapters, downloadable assets and native process management. This is the development reference; see the root guide for [setup](../README.md#install-and-run), [usage](../README.md#usage) and [configuration](../README.md#configuration).

## Design

Maia inference runs in process; Stockfish runs as native UCI processes. Applications own their adapters and close them on shutdown. Sharing code or cached files does not share running processes between applications.

`EngineRuntime` groups owned or borrowed resources, accepts explicit configuration and manages their lifetime. Entering an owned runtime resolves local asset identities; model weights and Stockfish processes start only when a cache miss needs inference or search. Borrowing server engines leaves their lifetime with the server. The runtime has no analysis-result cache or dependency on `analysis`; [AnalysisSession](../analysis/README.md#design) owns cached requests, search orchestration and job cancellation. Accuracy calculations and game policy belong to analysis.

## Modules

| Module | Responsibility |
| --- | --- |
| [runtime.py](runtime.py) | `EngineRuntime`: owned/borrowed resources and native startup, shutdown and pool lifetime |
| [maia.py](maia.py) | `MaiaPolicy`: history-aware move probabilities and batched position/rating inference |
| [stockfish.py](stockfish.py) | `StockfishScorer`: direct position/candidate scoring and access to the analysis pool |
| [stockfish_pool.py](stockfish_pool.py) | `StockfishPool`: persistent workers, leases and per-worker thread/hash allocation |
| [uci.py](uci.py) | Executable resolution, UCI startup/cleanup and time-unit conversion |
| [assets_maia.py](assets_maia.py) | Resolve the configured checkpoint or populate the official Maia model cache |
| [assets_stockfish.py](assets_stockfish.py) | Resolve, verify, download and atomically install official Stockfish binaries |
| [assets_identity.py](assets_identity.py) | Fingerprint local asset contents without loading models or starting engines |
| [assets.py](assets.py) | `ensure_runtime_assets` and the `python -m engine.assets` preparation entry point |
| [settings.py](settings.py) | Read root YAML and resolve engine paths |

## Interfaces

| Interface | Contract |
| --- | --- |
| `EngineRuntime` context manager | Resolve local assets and signatures, then close any resources started during the session |
| `EngineRuntime.get_stockfish()` | Start the direct Stockfish worker on the first uncached search |
| `EngineRuntime.borrow(maia, stockfish_scorer)` | Attach existing adapters without taking ownership; closing the runtime leaves them running |
| `EngineRuntime.create_pool(workers)` | Allocate the caller's chosen concurrency; pool workers start lazily |
| `MaiaPolicy.batch_evaluate(fens, ratings, opponents, timings=None, *, boards=None)` | Return aligned full legal policies and White expected scores; optional timing fields use milliseconds |
| `MaiaPolicy.probabilities(board)` | Use the same inference path at configured `MAIA.PLAYER_RATING` |
| `MaiaPolicy.asset_signature(...)` | Resolve local identity without constructing a model adapter or requiring an available GPU |
| `StockfishScorer.evaluate(board)` / `score(board, moves)` | Return White-oriented scores and achieved depth from a serialized direct worker |
| `StockfishScorer.analysis_pool` | Lazily create the shared pool for independent searches |
| `StockfishPool.acquire(control=None)` | Lease a worker and release it on completion, cancellation or error |
| `StockfishPool.clear_hash()` / `close()` | Serialize maintenance; closing rejects new leases and hash resets |

Maia's `boards` argument accepts a FEN-to-board mapping or an aligned board list. Use a list when the same FEN occurs with different histories. Preparation-cache keys retain complete history while tokenization uses the model's recent-history window. Inference preserves legal-move alignment; `value` is White's expected game score, `P(win) + 0.5 * P(draw)`, not pure win probability. Loading, preparation-cache access and inference share an internal lock. `model_signature` resolves an existing local checkpoint and its content fingerprint without loading model weights; explicit `load()` remains available for applications requiring startup readiness.

`StockfishScorer` serializes its direct process because overlapping UCI commands can cancel a search. Concurrent searches use pool leases. Closing the scorer is idempotent and final; create another adapter for a new lifetime. Search scheduling and continuation policy remain in `analysis.stockfish_search` and `analysis.stockfish_exploration`; adapters must report achieved depth separately from a requested target.

## Configuration

`settings.py` reads [config.yaml](../config.yaml) directly. Root [configuration rules](../README.md#configuration) govern paths, units and CLI overrides.

| Setting | Engine responsibility |
| --- | --- |
| `MAIA.MODEL`, `CHECKPOINT`, `CACHE_DIR`, `CACHE_EXECUTABLE` | Model identity and asset preparation |
| `MAIA.DEVICE` | `auto` selects available CUDA support; `cpu` provides the CPU path |
| `MAIA.BATCH_SIZE`, `POSITION_CACHE_ENTRIES` | Maximum position/rating rows per inference batch and cached CPU preparations |
| `MAIA.PLAYER_RATING`, `TEMPERATURE` | Default inference conditioning and play sampling |
| `STOCKFISH.EXECUTABLE`, `CACHE_DIR`, `RELEASE_API`, `ASSETS` | Binary resolution and official downloads |
| `STOCKFISH.START_TIMEOUT_SECONDS` | UCI startup timeout |
| `ANALYSIS.STOCKFISH_WORKERS`, `STOCKFISH_THREADS_PER_WORKER`, `STOCKFISH_HASH_MB_PER_WORKER` | Independent worker count and resources per worker |
| `ANALYSIS.STOCKFISH_EVALUATION` | Direct scoring depth and time limits |

Pool search threads equal `workers × threads_per_worker`; hash allocation equals `workers × hash_mb`. Any separate direct process receives another per-worker hash allocation, with process/model overhead additional. Raising worker count does not reduce memory per worker.

Asset caches default to `engine/.cache/maia3` and `engine/.cache/stockfish`. Reusable result measurements belong to [analysis's position cache](../analysis/README.md#coaching-artifacts-and-analysis-cache). Native adapters neither own that storage nor hold its file locks during computation. Batch shape affects scheduling, while individual rating pairs determine reusable Maia request identity.

Engine asset identity uses SHA-256 of file contents, so moving or touching identical assets preserves compatibility. Changed bytes invalidate the identity. Model history/device and Stockfish thread/hash settings remain part of the relevant engine signature.

## Development

Keep model loading, inference and native lifetimes here; use the shared adapters for every consumer. Preserve history and input order when batching, keep a working CPU path, and account for total worker resources when changing concurrency. Close owned resources on failure and cancellation without closing borrowed engines.

Measure model startup, inference, search and elapsed time separately before tuning. Stockfish time limits and achieved depth can vary between runs under the same configuration. The engine benchmark is a development utility; [the test guide](../tests/README.md) owns benchmark and live-run commands.

## Verification

Regression coverage in `tests/engine/` checks adapters, assets, batching/history, resource ownership and concurrent pool cleanup. Use the [shared test guide](../tests/README.md) for offline commands; prefer fake models and UCI processes for focused changes. Whole-game accuracy, cache and profiler validation belongs to [analysis](../analysis/README.md#verification).
