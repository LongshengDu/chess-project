# Backend

The backend serves the local web application and owns HTTP validation, saved games, play sessions and analysis jobs. Installation and user commands are in the [project README](../README.md#install-and-run); this document describes the backend's development contracts.

## Design

Controllers map HTTP requests to services. Reusable chess calculations belong to [analysis](../analysis/README.md), and model inference, engine processes and resource pools belong to [engine](../engine/README.md). Interactive web requests, whole-game analysis and coaching share the same analysis policies and canonical position evidence.

`PlatformAnalysis` holds backend services, the repository, engine locks and optional profiler. Whole-game jobs borrow the server's Maia and Stockfish resources through an `AnalysisSession`; finishing or cancelling a job releases its searches without closing shared engines. `main` owns engine construction and shutdown, including startup failures.

SQLite stores durable web history: game snapshots, favorites, play state and saved UI analysis. It is separate from reusable analysis evidence and generated figures. Clearing the analysis cache must not erase games. Normal requests consult the persistent position cache even when process memory contains a result, so external refreshes become visible without restarting the server.

Full-game publication stores prepared analysis and the UI position array together, keeping the raw array out of the prepared result. Figures are staged before publication under `SERVER.STORAGE.OUTPUT_DIR/<game id>-full/`. Failed, cancelled or deleted-game jobs preserve previously published results. Figure contents and cache identities are defined by the [analysis component](../analysis/README.md).

## Modules

| Module | Responsibility |
| --- | --- |
| [app.py](app.py) | CLI, application construction, static routes, error handling and engine shutdown |
| [settings.py](settings.py) | Read backend configuration directly from root YAML and resolve paths |
| [repository.py](repository.py) | SQLite schema, transactions, saved games, favorites, analysis and play persistence |
| [play.py](play.py) | Play-session state, legal histories, Maia replies, clocks, results and local statistics |
| [analysis_positions.py](analysis_positions.py) | Interactive position requests, canonical cache reuse, engine leases and streamed search frames |
| [analysis_games.py](analysis_games.py) | Whole-game job lifetime, shared-pipeline execution, cancellation and result publication |
| [routes_analysis.py](routes_analysis.py) | Analysis HTTP validation, PGN/FEN import, saved-game endpoints and streams |
| [routes_play.py](routes_play.py) | Play HTTP validation and mapping to the play service |
| [routes_profiler.py](routes_profiler.py) | Optional timing-control endpoints and isolated cold-run setup |

## Interfaces

All API paths below start with `/api/platform`. Browser routes `/`, `/analysis`, `/analysis/<route>`, `/play` and `/play/maia` serve the built frontend.

| Method and path | Contract |
| --- | --- |
| `GET /bootstrap` | Return the latest saved game or create the starting position |
| `GET /config` | Return resolved search policy, depth presets, budgets and concurrency |
| `GET /games` | Paginate custom games, play history or favorites |
| `POST /games` | Import one standard-chess PGN or complete six-field FEN |
| `GET/PATCH/DELETE /games/<id>` | Read, rename/favorite or delete a saved game |
| `GET/POST /games/<id>/analysis` | Retrieve saved positions and prepared analysis, or save interactive positions |
| `POST /games/<id>/analyze` | Stream the shared full-game pipeline and publish its completed result |
| `POST /games/<id>/analyze/cancel` | Cancel an identified full-game run |
| `POST /maia` | Return history-aware probabilities for requested rating pairs |
| `POST /stockfish` | Stream position-search frames as newline-delimited JSON |
| `POST /stockfish/cancel` | Cancel a search by `search_id` |
| `POST /explore` | Evaluate a candidate continuation within exploration limits |
| `GET /play/config` | Return the model, default Maia rating and sampling temperature |
| `POST /play/games` | Create an idempotent local play session |
| `GET /play/games/<id>` | Read persisted play state |
| `POST /play/move` | Validate move history and obtain a Maia reply |
| `POST /play/games/<id>/moves` | Save move history, clocks and terminal state |
| `POST /play/games/<id>/analysis` | Save the latest play snapshot for analysis |
| `GET /play/stats` | Return local game counts and results |
| `GET /profiler` | Read timing-run status; registered only with a profiler |
| `POST /profiler/start`, `/profiler/position`, `/profiler/finish` | Control the optional timing recorder |

Full-game requests contain `run_id` and optional `target_depth`/`seconds`. Events are `progress`, `position`, `complete`, `error` or `cancelled`. Position events may arrive out of order; final positions are ordered. A second active run for the same game is rejected. Disconnects cancel work, and interactive autosave cannot overwrite a running job's publication.

The backend replays supplied starting FENs and move histories, then validates position consistency and search options before engine work. Value/type errors return HTTP 400; missing saved resources return 404. A stream can report an error after HTTP headers have been sent, so clients must inspect its frames.

Play saves update state and analysis snapshots transactionally; changed moves invalidate saved analysis. Local play PGNs set both `WhiteElo` and `BlackElo` to the selected Maia rating and `Site` to `lichess.org`. `TimeControl` records starting seconds plus increment; unlimited games use exactly `300` for analysis while live clocks remain unlimited. These headers survive analysis handoff. The local player remains Unrated in the UI.

## Configuration

[Root configuration](../README.md#configuration) defines application-wide rules. `settings.py` reads [config.yaml](../config.yaml) directly; supported CLI arguments override its defaults.

| Section | Backend use |
| --- | --- |
| `SERVER` | Address, request-size limit and SQLite/output locations |
| `FRONTEND.STATIC_DIR` | Built files served by Flask |
| `FRONTEND.STOCKFISH_TIME_SCALE_BY_DEPTH` | Web presets scaling evaluation default and maximum seconds |
| `MAIA`, `STOCKFISH` | Shared engine construction and assets |
| `ANALYSIS` | Search policies, worker resources, position cache and optional profiler |

`python backend/app.py --help` lists overrides without starting the server. Web presets do not change coach budgets or the separate exploration limits. Configuration owns worker count and hash memory per worker.

## Development

`create_app(platform, static_folder=None)` accepts constructed dependencies, allowing Flask test-client checks without native engines or listening ports. Keep routes concerned with HTTP, services with request lifetimes, and the repository with SQL.

Timing capture is opt-in through `--profiler-dir` or `ANALYSIS.PROFILER_DIR`; the browser selects it with `?profiler=1`. Cold profiling clears transient request/hash state and runs full-game searches in a temporary evidence cache; completed measurements are imported into the shared cache. It does not delete normal cached evidence or unload an already loaded model. The shared full-game session records indexed timing events; interactive requests do not fabricate game timings. Recording, reporting and run comparison belong to `analysis/profiler/`.

## Verification

Run focused regressions from the project environment:

```powershell
python -m unittest discover -s tests/backend -t . -p "test_*.py"
```

Coverage includes startup and cleanup, request validation, persistence, streaming/cancellation, legal play and analysis handoff. Tests use fake engines and Flask's test client. Follow the [shared test guide](../tests/README.md) for broader checks; stop any server started for manual verification.
