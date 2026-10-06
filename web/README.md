# Local Maia frontend

The application's only frontend combines upstream Maia analysis and play pages
with local Python engines and storage. **Analysis** is the landing page; the
navigation contains **Play Maia** and **ANALYSIS**. Boards, clocks, setup dialogs,
panels, controllers, tutorial, responsive layouts and styles come from the
pinned `deps/maia-platform-frontend` submodule, revision
`a6e52f5c811ee18863cb2f0e81f2433a5b9905de`.

## Build and run

From the repository root:

```powershell
git submodule update --init --recursive
uv run python web/build.py
uv run python backend/app.py
```

Open <http://127.0.0.1:5000/analysis>. For an NVIDIA GPU with a CUDA 13-compatible
driver, retain the CUDA extra on **every** `uv run` command, including builds and
tests, so uv does not synchronize back to the CPU PyTorch build:

```powershell
uv sync --extra cuda
uv run --extra cuda python web/build.py
uv run --extra cuda python backend/app.py --device cuda
```

Stop Python engines before switching PyTorch installations on Windows. The
default `--device auto` selects CUDA when available; `--device cpu` forces CPU
inference. Stockfish uses CPU workers independently. Startup reports the device
and search configuration; Maia loads lazily on its first uncached evaluation.
Stop the server with **Ctrl+C**.

The build helper bootstraps Node.js 22+ and pinned pnpm, installs dependencies in
`web/node_modules`, and builds `web/dist`. It does not modify the submodule.
The root [config.yaml](../config.yaml) controls the application; CLI options
override YAML.

| Configuration | Purpose |
| --- | --- |
| `FRONTEND.STATIC_DIR` | Files served by Python; normally `web/dist` |
| `FRONTEND.BUILD` | Node settings and build-tools directory, normally `web/.tools` |
| `SERVER.HOST`, `SERVER.PORT` | Listening address; default `127.0.0.1:5000` |
| `SERVER.STORAGE.DATABASE` | Saved games and analysis; normally `backend/.cache/analysis.sqlite3` |
| `MAIA` | Shared model, device, opponent rating, batching and sampling |
| `ANALYSIS` | Stockfish workers, search limits and caches |

## User workflows

### Analysis

The first visit creates a saved starting position. **Select Game → Custom →
Analyze Custom PGN/FEN** imports one standard-chess PGN or a complete six-field
FEN. The page retains these upstream functions:

- Board and move-list navigation, exploratory moves, arrows, rotation, legal
  move highlighting, promotions and sounds.
- **Analysis** with Maia probabilities, Stockfish recommendations and evaluations,
  descriptions, win-rate bars, blunders and mistakes.
- **Moves by Rating**, using local inference for all 21 ratings from 600–2600.
  Either Maia selector changes the active rating.
- **Options**, including badges, whole-game depth presets 12/15/18, progress
  and cancellation.
- **Learn from mistakes**, with player selection, corrections, solution reveal
  and navigation; **Export** for copying mainline PGN or current FEN.
- Saved games, custom names, favorites, tutorial and desktop/mobile layouts.

**Analyze Entire Game** runs the same Python full-game pipeline as the
[coach application](../coach/README.md). It saves complete Maia/Stockfish evidence,
move hints, Lichess accuracy and fitted ratings for both players. The page currently
displays the engine evidence; the full saved result, including ratings, is retained
for future rating panels and web coaching. This button does not invoke a coaching LLM.

Games, names, favorites and cached mainline evaluations survive server restarts
in the ignored SQLite database. Completed whole-game runs save immediately;
interactive auto-save is paused during a full-game run, whose completed result is
saved by the server. Keep the tab open during analysis, and wait for **Analysis auto-saved** after cancellation or
interactive analysis. Imported PGN side lines remain in the saved game; new
exploratory moves are session state. Export contains the mainline, and the UI
does not render PGN comments or annotations.

### Play Maia

Open **Play Maia** or `/play` to choose White, Black or random, an opponent from
600–2600, a preset/custom time control and an optional starting FEN. The default
rating is `MAIA.PLAYER_RATING`; the upstream initial time control is 3+0. Play
reuses the configured Maia model and device.

Gameplay supports clocks, premoves, promotion, sounds, board navigation, resign
confirmation, rematches, new games and plain PGN/FEN export. Sampling uses
`MAIA.TEMPERATURE`; disabling sampling or choosing zero temperature selects the
most probable legal move. **Human-like** timing retains the upstream display
delay but does not predict human thinking time.

Moves and results are persisted locally. **ANALYZE GAME** waits for the final
save and opens that game in analysis. Statistics count local games and wins;
the player remains **Unrated** because there is no account-rating system.
Refresh/back follows upstream's fresh-game behavior rather than resuming a live
clock; saved games remain available for analysis.

### Local-service boundaries

Maia account history, Lichess synchronization, the World Championship archive
and the hosted opening-frequency database are external services. Their
game-list tabs do not fetch hosted data; import PGN through Custom Analysis.
Local games need no sign-in, opening predictions use local Maia, and play and
analysis do not send platform analytics. The icon font still loads from Google
Fonts. Results can differ from the hosted site's model, Stockfish version,
search policy and opening database.

## Source and integration

The Python build support follows the same component boundary:

| Module | Responsibility |
| --- | --- |
| `settings.py` | Read frontend settings directly from root YAML and resolve paths |
| `build_node.py` | `NodeRuntime`: discover, verify and install the Node runtime |
| `build_packages.py` | `FrontendPackages`: check pinned dependencies and run the package manager |
| `build.py` | Prepare those dependencies, type-check, then bundle the frontend |

The `build_*` prefix groups the build helpers beside the `build.py` entry point.
Application source remains under `src/`; the Python helpers own build tooling.

Importing these modules performs no downloads or builds. Node archives are
checksum-verified, their filenames and extraction paths are validated, and
unsupported archive entries are rejected before replacing an existing install.

[`src/upstream-app.jsx`](src/upstream-app.jsx) mounts the actual upstream
`src/pages/analysis/[...id].tsx` and `src/pages/play/maia.tsx` with their providers.
Local adapters supply the standalone environment:

| Files in `src/adapters/` | Responsibility |
| --- | --- |
| `router.jsx`, `routes.js`, `next.jsx` | Client-side URLs and Next routing/link/image/head equivalents, including live-game query state |
| `api.js` | Custom games, favorites, cached analysis and upstream GameTree conversion |
| `play-api.js` | Setup, Maia replies, ordered move saves, statistics and post-game analysis |
| `play-transform.js`, `header-transform.js` | Checked source patches for local play and two-entry navigation |
| `engines.jsx` | Maia batching and incremental Stockfish streams through Flask |
| `deep-analysis.js` | Whole-game progress and tree updates from the shared Python pipeline |
| `game-analysis.js`, `analysis-data.js` | Stream/cancellation client and common-result adaptation for upstream nodes |
| `analysis-save.js` | Interactive autosave, paused while the shared full-game job owns publication |
| `auth.jsx` | Local player context and setup modal with configured defaults |
| `hooks.js`, `contexts.js`, `components.js`, `analysis-components.js`, `lib.js` | Narrow upstream exports excluding unrelated platform features |
| `telemetry.js` | Disables hosted analytics |

[`vite.config.mjs`](vite.config.mjs) resolves upstream imports against the local
lockfile and bundles upstream pieces, sounds and branding. In-memory patches
preserve PGN variations, refresh mobile rating charts, avoid duplicate searches,
integrate local play/saves and retain actual search depths and completed updates.
**Builds fail if expected upstream patch anchors change.** Update compatibility
patches and tests when upgrading upstream; do not edit the submodule to implement
local behavior.

The [backend](../backend/README.md) provides `/api/platform/` analysis endpoints
and `/api/platform/play/` sessions, legal-move validation and replies. It shares
`MaiaPolicy` and `StockfishScorer`. Threaded requests and separate engine locks
allow inference and searches to overlap; closing a stream stops its search.
`/api/platform/explore` supports bounded continuation exploration. The former
coaching report/download API is removed; historical report files remain on disk
but are not served by the web application.

Scheduling, budgets and streaming belong to [analysis](../analysis/README.md);
engine operation belongs to [engine](../engine/README.md). The standalone
shared player-rating method is documented in
[Bayesian shared-curve fitting](../docs/bayesian_shared_curve.md).

## Performance

### Current execution and limits

Maia batches position/rating pairs across positions, up to `MAIA.BATCH_SIZE`
rows (128 by default), and reuses history tokens and legal masks. CUDA inference
uses the model's mixed-precision option; White's expected score comes from its
loss/draw/win head. Stockfish searches independent positions through persistent
workers, reusing engines and hash. Current defaults are **four workers × four
threads** and 128 MB hash per worker (512 MB for the pool).

The default `bounded` policy stops each search at its depth ceiling **or** time
allowance, whichever is reached first. All search phases share a position budget:

| Preset | Depth ceiling | Time scale | Default budget | Maximum requested budget |
| --- | ---: | ---: | ---: | ---: |
| Fast | 12 | 0.2 | 2 s | 6 s |
| Balanced | 15 | 0.5 | 5 s | 15 s |
| Deep | 18 | 1.0 | 10 s | 30 s |

`FRONTEND.STOCKFISH_TIME_SCALE_BY_DEPTH` scales both the default and maximum
seconds from `ANALYSIS.STOCKFISH_EVALUATION` (currently 10 and 30). The backend
advertises resolved presets and uses the same limits for individual positions,
whole games, saved-result validation, and profiler metadata. Presets above the
evaluation depth ceiling are unavailable. These scales do not affect coach
analysis or the separately configured continuation previews.

1. Screen all legal moves to d6, using at most 5% of the budget, capped at 0.3 s.
2. Search the best single engine line with up to 40% of the budget.
3. Search the played move first, then four Maia candidates, sharing another 40%.
4. Search four leading engine alternatives with the remaining allowance.

Unused time flows forward. Stopping overhead, startup, inference and transport
fall outside the search budget. This avoids full-depth MultiPV for every legal
move while retaining legal-move coverage, following Stockfish's
[combined-limit and MultiPV behavior](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html).

The UI shows achieved depth and **time limit** when selected candidates did not
all reach the target. Per-move depths survive save/restore in
`root_move_depth_vec`; `target_reached`, `candidate_min_depth` and `stop_reason`
describe coverage and termination. The best-line search supplies the primary
recommendation, protecting it from shallow screening outliers. Classification
and descriptions require depth 12 for the relevant move. Scores use White's
perspective and the upstream win-rate conversion.

Whole-game execution sends one request to the Python pipeline, pauses duplicate interactive
searches, counts completed positions and saves rows in game order with fitted ratings. Cancellation
stops active and queued work; incomplete searches never enter the completed
cache. Completed bounded searches are reusable even when time-limited. Larger
budgets or changed candidates can trigger new work; stronger saved results survive.

`--analysis-strategy staged` deepens selected candidates in phases;
`exhaustive` searches every legal move toward the target depth. Full-game and
interactive root searches retain a watchdog at the preset's maximum time plus cleanup grace.
Bounded/staged caches cannot satisfy exhaustive
requests. `ANALYSIS.STOCKFISH_WORKERS` and `STOCKFISH_THREADS_PER_WORKER` control
concurrency and per-engine threads separately; `STOCKFISH_HASH_MB_PER_WORKER`
sets each worker's hash allocation. More threads may slow short searches. Stockfish runs
on CPU, including NNUE; CUDA accelerates Maia, as described in the
[Stockfish FAQ](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#can-stockfish-use-my-gpu).

### Recorded measurements

Historical measurements used an **i7-10700KF (8 physical cores, 16 logical
processors) and RTX 4070 SUPER (12 GB)**. They retain their original settings
and are **not measurements of today's 4×4, time-scale-2 configuration or current
rating estimator**. Single-run score differences describe a latency/coverage
tradeoff, not equivalent engine strength or ground-truth accuracy.

**Position batching and workers, 2026-10-01.** Cold analysis and the then
Bayesian fitter on all 43 plies of `games/game8.pgn`, using separate caches,
d18/six-second position budgets, eight total search threads and 512 MB hash:

| Allocation | Analysis | Including startup/output | Speedup | Maia calls / time |
| --- | ---: | ---: | ---: | ---: |
| Original 1 worker × 8 threads | 244.87 s | 247.98 s | 1.00× | 301 / 11.36 s |
| 2 workers × 4 threads | 133.40 s | 136.41 s | 1.84× | 88 / 6.36 s |
| 4 workers × 2 threads | 67.75 s | 70.72 s | 3.61× | 88 / 6.11 s |
| 8 workers × 1 thread | 46.58 s | 51.12 s | 5.26× | 88 / 7.88 s |

All runs evaluated 10,578 Maia rows. Stockfish consumed 231.85 s (95%) of the
original analysis. The 4×2 run cut elapsed analysis by 72.3%. Its warm repeat
took 0.23 s for analysis and 3.83 s overall: 43 Stockfish cache hits, no searches
or inference. Browser rendering and coaching-agent generation are excluded;
no LLM requests were made.

The 4×2 run matched the original best move at 36/43 positions, with a 34 cp
median/108 cp maximum best-score difference. Best lines reached d18; candidates
retained individual depths. White/Black accuracy changed from 46.91%/43.70% to
48.90%/44.70%, with six severity changes. The 8×1 run differed by 48 cp
median/162 cp maximum and eight severity labels. An in-process route check
measured 84 Maia rows in 0.65 s, four d12 streams in 1.55 s and cached repeats
in 0.009 s, with full legal-move coverage and no active search controls remaining.

See [single-game benchmark commands](../tests/README.md#single-game-speed)
and preserved [comparison](../tests/coach/output/single-game-speed/comparison.md),
[comparison JSON](../tests/coach/output/single-game-speed/comparison.json),
[4×2 timings](../tests/coach/output/single-game-speed/four-workers/timing.json)
and [route check](../tests/coach/output/single-game-speed/demo-integration.json).
These ignored artifacts may be absent in a fresh checkout. Historical settings
were workers 4, threads per worker 2, total hash 512 MB, time scale 1, Maia batch
size 128 and position cache entries 512. Current code/fitting changes prevent
exact reproduction of the old run.

**Bounded search, 2026-09-21.** With Maia3 79M, Stockfish 19, one eight-thread
engine and 512 MB hash, a fresh 52-position `example.pgn` browser run fell from
902.775 s (uncapped staged d18) to **204.477 s** (bounded d18/six seconds):
**4.42× faster**, retaining all 21 Maia ratings and legal-move coverage.

| Stockfish phase | Uncapped staged | Bounded |
| --- | ---: | ---: |
| All-move screening | 329.398 s | 2.963 s |
| Best single line | — | 41.493 s |
| Intermediate candidate pass | 64.468 s | — |
| Top four engine alternatives | 342.368 s | 72.611 s |
| Remaining selected/played candidates | 156.824 s | 77.873 s |

Before 17...g6, uncapped screening took 293.611 s despite its d10 target
(31 PVs, 1.068 billion nodes, selective depth 70). Three positions consumed
58.07% of the run. Bounded native work peaked at 6.006 s per position; the
slowest end-to-end position took 6.183 s. Best moves matched at 35/51
nonterminal positions; median best/played-score differences were 9/9.5 cp.
The largest played-score difference was 503 cp before 20.Qxe4 at d13. All
selected candidates reached d18 at 27/51 positions. Shallow scores are not
interchangeable with exhaustive full-depth analysis.

Preserved evidence: [uncapped report](../tests/web/benchmarks/example-d18-profile/REPORT.md),
[interactive profile](../tests/web/benchmarks/example-d18-profile/report.html),
[position CSV](../tests/web/benchmarks/example-d18-profile/positions.csv),
[search CSV](../tests/web/benchmarks/example-d18-profile/searches.csv),
[bounded comparison](../tests/web/benchmarks/example-bounded-profile/COMPARISON.md),
[per-position differences](../tests/web/benchmarks/example-bounded-profile/comparison.csv)
and [bounded report](../tests/web/benchmarks/example-bounded-profile/REPORT.md).

Earlier opening-position microbenchmarks measured a warm 21-rating Maia batch
at 627 ms with the original CPU build, 620 ms with updated code on CPU and
24 ms on CUDA (about 26× faster); first calls were 1.32–1.39 s. Warm values
were medians of three calls after one cold call, including preprocessing and
conversion. Three-position staged d18 totals were 6.09/6.82/9.65 s with 4/8/16
threads. One all-move d18 search stopped after 15 s at d17; it is not a completed
d18 timing. These short samples missed the later tactical stalls. Original
JSON data remains in [`tests/web/benchmarks/`](../tests/web/benchmarks/).

### Profiler

Stop competing engine workloads before timing. From the repository root:

```powershell
uv run --extra cuda python -m tests.engine.benchmark_engines --strategy staged --threads 8 --max-seconds 60 --output tests/web/benchmarks/new-staged.json
uv run --extra cuda python -m tests.engine.benchmark_engines --strategy baseline --threads 8 --max-seconds 60 --output tests/web/benchmarks/new-all-moves.json
uv run --extra cuda python -m tests.engine.benchmark_engines --device cpu --maia-only --output tests/web/benchmarks/new-cpu.json
```

`baseline` means original all-legal-move Stockfish search; Maia uses currently
installed code/device. For **Analyze Entire Game**, start a dedicated server
with a new profiler output directory and study database:

```powershell
uv run --extra cuda python backend/app.py --port 5002 --device cuda --profiler-dir tests/web/benchmarks/my-full-game --analysis-database backend/.cache/profiler-my-full-game.sqlite3
```

Import the PGN, append `?profiler=1` to its saved analysis URL and reload before
**Analyze Entire Game → Deep (d18)**. For a cold model, restart after import and
open that URL before any ordinary analysis page. Profiler mode disables automatic
interactive requests, bypasses saved node results and clears server result
caches at Start; it cannot unload an already loaded model.

Keep the tab open until completion. The server checkpoints `runtime-profiler.json` after
each position, appends search events to JSONL and exposes progress at
`/api/platform/profiler`. Restart before another cold run or after cancellation.
Generate timing tables and charts with:

```powershell
uv run --extra cuda python -m analysis.profiler.report tests/web/benchmarks/my-full-game --pgn example.pgn
```

The exporter rejects incomplete runs and verifies recorded positions against the
PGN. Shared-pipeline reports contain Markdown, HTML, CSV, JSON and a standalone
SVG chart. CSVs retain actual depths, termination reasons and cache indicators;
cached results do not replay old search timings. Historical serial profiles remain
supported, including saved `profile.json` files; add `--with matplotlib` to the command when exporting those older
profiles, whose charts use Matplotlib.

Stockfish phase totals accumulate worker time and can exceed whole-run elapsed
time because position jobs overlap. Maia timing belongs to complete batches;
per-position Maia times are left unmeasured. The report includes rating fitting
and total pipeline time, and does not subtract overlapping work to invent
residual overhead. Ordinary analysis does not synchronize CUDA or write runtime-profiler
files; timing measurements include the additional synchronization/logging
overhead.

## Development and validation

With Node.js 22+ and pnpm 10.15.0:

```powershell
cd web
pnpm install --frozen-lockfile
pnpm dev
```

Start Python separately from the repository root. Vite proxies `/api` to port
5000; `MAIA_BACKEND_URL` changes this development-only proxy address. Rebuild and
refresh after production frontend changes; restart Python after backend changes.
From the repository root:

```powershell
node --test tests/web/*.test.mjs
uv run --extra cuda python -m unittest discover -s tests -t . -p "test_*.py"
uv run --extra cuda python web/build.py
```

Omit `--extra cuda` on CPU-only installations. Frontend tests cover completion,
cache reuse, budget upgrades, candidate coverage, depth labels, routing and
upstream patches. Python tests cover integration, persistence, cancellation,
legal play, post-game analysis and routes; see [tests](../tests/README.md).
The build checks the local TypeScript entry point and bundles upstream
TypeScript; it does not perform a full Next.js application type-check.

Upstream code and assets retain their GPL-3.0 notices. See the repository
[LICENSE](../LICENSE) and the submodule's license.
