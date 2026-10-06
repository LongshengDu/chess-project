# Local web backend

`backend/` serves the Maia play/analysis frontend and adapts its requests to local
Python engines. It owns HTTP validation, saved studies, play sessions and analysis
streaming. Reusable chess calculations belong to [analysis](../analysis/README.md);
engine adapters and processes belong to [engine](../engine/README.md).

## Start and configure

Build the frontend once, then start from the project root:

```powershell
python web/build.py
python backend/app.py
```

Open <http://127.0.0.1:5000>. Module execution, `python -m backend.app`, also works.
Startup checks `FRONTEND.STATIC_DIR/index.html`, ensures engine assets exist and
creates one shared Maia adapter plus a Stockfish scorer. Flask handles requests
in threads. **Ctrl+C** exits the application and closes Stockfish workers.

Defaults come directly from root [config.yaml](../config.yaml) through
`backend/settings.py`. `SERVER` controls the address, request-size limit and
SQLite storage; `FRONTEND.STATIC_DIR` locates `web/dist`. Model settings are under
`MAIA`, while search limits and CPU resources are under `ANALYSIS`.

Inspect every supported override without starting a server:

```powershell
python backend/app.py --help
```

| Option | Purpose |
| --- | --- |
| `--host`, `--port` | Listening address |
| `--device auto/cpu/cuda` | Maia inference device |
| `--stockfish-threads-per-worker` | Threads for each Stockfish worker |
| `--analysis-strategy bounded/staged/exhaustive` | Whole-game search policy |
| `--analysis-database <path>` | Separate study/play database |
| `--profiler-dir <directory>` | Enable browser timing capture and profiler routes |

Worker count and hash memory per worker remain configured in YAML. See the
[analysis guide](../analysis/README.md) for search budgets and the
[web performance guide](../web/README.md#performance) for profiling examples.
`FRONTEND.STOCKFISH_TIME_SCALE_BY_DEPTH` defines web preset multipliers for both
the default and maximum evaluation time; the coach does not use these presets.
`ANALYSIS.PROFILER_DIR` supplies the default recording directory;
`null` leaves the profiler disabled.

## Modules and lifecycle

| Module | Responsibility |
| --- | --- |
| `app.py` | CLI, dependency construction, static files, error handling and shutdown |
| `play.py` | Local play state, legal-history checks, Maia replies, clocks/results and statistics |
| `settings.py` | Component-local YAML loading and path resolution |
| `repository.py` | SQLite schema, transactions, studies, favorites, analysis arrays and play persistence |
| `analysis_positions.py` | `PlatformAnalysis`: interactive position requests, in-memory request caches, engine locks and streamed responses; holds shared backend resources |
| `analysis_games.py` | `FullGameAnalysis`: saved-game job lifetime, shared-pipeline invocation, streamed progress, cancellation and atomic database publication |
| `routes_play.py` | Play request validation and HTTP mapping to the play service |
| `routes_analysis.py` | Analysis request validation, imports, study endpoints and streaming transport |
| `routes_profiler.py` | `ProfilerRoutes`: optional HTTP controls for the timing recorder, request validation and cold-run cache reset |

Module groups put their purpose first: `analysis_*` contains the position and
whole-game services, while `routes_*` contains HTTP controllers. `play.py` owns
the separate play-session service; `repository.py` owns persistence.

`create_app(platform, static_folder=None)` accepts an already constructed
`PlatformAnalysis` instance. This keeps route tests independent of actual engines
and listening ports. It registers three small controllers; profiling routes exist
only when a profiler is supplied. Controllers handle HTTP, services own request
and session lifetimes, and `GameRepository` owns SQL. Reusable chess calculations
stay in `analysis/`. Play and analysis share the same model
and lock. `main` owns the scorer and closes it in `finally`, including startup
failures. Stockfish leases and active search controls are released when searches
finish, fail or are cancelled. No service starts its own server or model process.

### Backend services versus shared analysis

`analysis_positions.py` serves interactive position requests from the web page.
It validates requested boards and rating arrays, caches repeated requests in
memory, leases Stockfish workers and serializes search frames for the browser.
Maia inference is performed by `engine/maia.py`; Stockfish search policy is
implemented by `analysis/stockfish_search.py`. Its `PlatformAnalysis` object also
holds the repository, locks, profiler, play service and whole-game service, so
its current scope includes shared backend state as well as position requests.
Interactive requests do not write game profiling events; timing collection belongs
to the shared full-game session, which knows each position's game index.

`analysis_games.py` manages **web jobs** for saved games: run IDs, worker threads,
progress queues, disconnects, cancellation and database publication. It calls
`analysis/game/pipeline.py` through a borrowed `analysis/engine_session.py`
session. The shared pipeline computes move evidence, hints, accuracy and both
player-rating fits; the coach calls that same pipeline. The backend service does
not implement a separate full-game calculation or fitting method.

`routes_profiler.py` exposes **performance timing** controls under
`/api/platform/profiler`.
It validates HTTP requests and asks `analysis/profiler/runtime.py` to start,
record or finish a timing run. The recorder owns timing events and files.
`analysis/profiler/report.py` creates timing reports, and
`analysis/profiler/comparison.py` compares runs. These three analysis modules
handle profiling data; the backend route module only supplies the web controls.

## HTTP interface

All API paths below start with `/api/platform`. The browser routes `/`,
`/analysis`, `/analysis/<route>`, `/play` and `/play/maia` serve the built frontend.

| Method and path | Behavior |
| --- | --- |
| `GET /bootstrap` | Select the latest saved game or create the initial position |
| `GET /config` | Active search policy, position budgets and concurrency settings |
| `GET /games` | Paginated studies; supports custom, play and favorites views |
| `POST /games` | Import one PGN or complete six-field FEN |
| `GET/PATCH/DELETE /games/<id>` | Read, rename/favorite or remove a study |
| `GET/POST /games/<id>/analysis` | Load/save the mainline position-analysis array |
| `POST /games/<id>/analyze` | Run the common full-game pipeline, stream positions/progress, and save ratings, hints and accuracy |
| `POST /games/<id>/analyze/cancel` | Cancel one identified full-game run without closing shared engines |
| `POST /maia` | History-aware legal-move probabilities across requested ratings |
| `POST /stockfish` | Stream search frames as newline-delimited JSON |
| `POST /stockfish/cancel` | Cancel an active search by `search_id` |
| `POST /explore` | Check a candidate continuation under exploration limits |
| `GET /play/config` | Local Maia model, default rating and sampling temperature |
| `POST /play/games` | Create a local play session |
| `GET /play/games/<id>` | Read its persisted state |
| `POST /play/move` | Validate the supplied history and obtain a Maia reply |
| `POST /play/games/<id>/moves` | Save play progress and terminal state |
| `POST /play/games/<id>/analysis` | Save the latest state for analysis handoff |
| `GET /play/stats` | Completed-game counts and wins |

With profiling enabled, `GET /profiler` and `POST /profiler/start`,
`/profiler/position`, `/profiler/finish` record browser and engine timing.
These routes are absent in a normal launch.

FENs, move legality, candidate lists, rating requests and search options are
validated before engine work. Value/type errors become HTTP 400 JSON responses.
Missing saved resources return 404. Stockfish stream failures are reported as an
error frame; clients must inspect frames even after the stream has opened.
Requests are bounded by `SERVER.MAX_REQUEST_BODY_MB`.

## Persistence and caches

The default database is `backend/.cache/analysis.sqlite3`. SQLite stores game
snapshots, favorites, saved analysis arrays and local play state. Connections
enable foreign keys and use `SERVER.STORAGE.LOCK_TIMEOUT_SECONDS` for lock waits.
Study deletion cascades to associated stored records.

Each completed full-game rating run also replaces its SVG figures and fit JSON
under `SERVER.STORAGE.OUTPUT_DIR/<saved game id>-full/player-rating/`. The default
output root is `backend/output`, separate from engine caches and the SQLite
database. For the default `shared_curve_affine` method, `analysis.svg` shows the
current game's accuracy curve alongside both players' affine rating decisions.
The fit uses only that game's measurements and supplied account ratings; it
loads no population asset or other benchmark game. `prior.svg` shows the translated
prior; when ratings use another scale, its density is transformed consistently.
The retained Bayesian method displays posterior densities; the default has point
estimates without a posterior interval. The former corpus-based estimators are
blocked, and cached historical ratings must be refreshed before presenting them
as current results. Rating axes follow the game's rating scale; native Lichess
Blitz axes display 200–3000 while the numerical fitting grid remains 0–3200.
The standalone prior matches the canvas size of one shared-accuracy panel.
Rendering is staged before publication;
cancelled or deleted-game runs cannot replace a previous successful figure set.

Play creation is transactional and retries return the original game. Updating a
play session stores both its state and analysis snapshot in one transaction,
invalidating saved analysis when moves change. Game lists filter and paginate in
SQLite so opening a page does not deserialize the entire game collection.

Maia and Stockfish request caches are process-local. Completed browser analysis is
also stored in SQLite. Search cache keys include board history to preserve
repetition context. Saved analysis validates position objects and search metadata
before replacing an existing cache. Retrieval drops incompatible Stockfish entries from the
response when a saved bounded-search budget differs or an exhaustive request
would otherwise reuse staged/bounded evidence; the stored study remains intact.

The frontend's **Analyze Entire Game** action calls `analysis.game.pipeline.analyze_game`,
the same pipeline used by the [coach CLI](../coach/README.md). It borrows the
server's Maia model and Stockfish pool and calculates complete move evidence,
flags, Lichess accuracy and both player ratings. Ratings are stored even though
the current UI does not display them. No coaching model is invoked by this action.

The request supplies `run_id` and an optional `target_depth`. NDJSON events have
types `progress`, `position` (mainline index plus engine data), `complete` (the
canonical analysis), `error`, or `cancelled`. Position events may arrive out of
order; the final array is ordered. A duplicate active run for the same saved game
is rejected. Cancellation, disconnects and failed runs preserve the previous full
result. Completion writes the full result and UI cache in one SQLite transaction.
`GET /games/<id>/analysis` returns both `positions` and `analysis`; older position-only
caches return `analysis: null`. Playing more moves invalidates both saved forms.
Shared evidence uses `ANALYSIS.CACHE_DIR`, independent of SQLite storage and
report destinations. This full result is available for future web coaching.

## Development and checks

Backend regressions use Flask's test client and fake engines:

```powershell
python -m unittest discover -s tests/backend -t . -p "test_*.py"
```

Coverage includes application startup/cleanup, configuration, study import and
storage, streaming/cancellation, play validation and saved-game handoff. Keep new
route tests under `tests/backend/`; see the [test guide](../tests/README.md). These checks do not
start a listening server or call a coaching model.
