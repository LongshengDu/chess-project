# Web

The project's sole frontend adapts the upstream Maia analysis and play pages to the local Python backend. Installation and user workflows are in the [project README](../README.md#install-and-run) and [usage guide](../README.md#usage); this document covers frontend integration and development.

## Design

Analysis is the landing page, with Play Maia and ANALYSIS navigation. Boards, setup dialogs, clocks, move/rating panels, controls, responsive layouts and styles come from the pinned [Maia frontend submodule](../deps/maia-platform-frontend/). Local behavior is implemented through adapters and checked source transforms, without editing the dependency.

The browser sends complete starting-FEN/move histories for the selected mainline or variation. The [backend](../backend/README.md#interfaces) validates them and shares evidence with full-game analysis and coach tools. **Analyze Entire Game** invokes the common Python pipeline, pauses duplicate interactive searches/autosaves, streams progress and saves the completed result. It does not invoke a coaching model.

Play saves are ordered, and analysis handoff waits for the final save. Saved and copied play PGNs use the selected Maia rating for both `WhiteElo` and `BlackElo`, `Site "lichess.org"`, and starting seconds plus increment in `TimeControl`. Unlimited games export exactly `TimeControl "300"` while live clocks remain unlimited. Analysis export preserves these headers, and the local player remains Unrated.

Imported PGN variations remain in saved games; exploratory variations are browser state. Export contains the mainline. Hosted account history, Lichess synchronization, broadcast archives and opening-frequency services are not part of the local application. Play/analysis analytics are disabled; the icon font still loads externally.

## Modules

| Module | Responsibility |
| --- | --- |
| [settings.py](settings.py) | Read frontend configuration directly from root YAML and resolve paths |
| [build_node.py](build_node.py) | Discover or install a checksum-verified Node runtime |
| [build_packages.py](build_packages.py) | Check pinned dependencies and invoke the package manager |
| [build.py](build.py) | Prepare build dependencies, type-check and bundle the frontend |

Imports do not download or build anything. The build helper validates downloaded runtime archives before installation and keeps dependencies in `web/node_modules`.

## Source

[`src/upstream-app.jsx`](src/upstream-app.jsx) mounts the upstream analysis and play pages with their providers. [`vite.config.mjs`](vite.config.mjs) resolves dependency imports against the local lockfile, applies checked patches and bundles upstream assets.

| Source under `src/adapters/` | Responsibility |
| --- | --- |
| `router.jsx`, `routes.js`, `next.jsx` | Local navigation and Next router/link/image/head adapters |
| `http.js` | JSON requests and backend error handling |
| `api.js` | Saved games, metadata, analysis restoration and GameTree conversion |
| `play-api.js` | Game setup, replies, ordered saves, statistics and analysis handoff |
| `play-transform.js`, `header-transform.js` | Checked upstream patches for local play, PGN export and navigation |
| `engines.jsx` | Maia requests, streamed Stockfish frames and shared batch ownership |
| `engine-history.js` | Starting FEN and complete move path for the selected node |
| `search-policy.js` | Saved-search compatibility and achieved-depth labels |
| `deep-analysis.js` | Whole-game progress, tree updates and optional profiler coordination |
| `game-analysis.js`, `analysis-data.js` | Streaming/cancellation client and upstream result adaptation |
| `analysis-save.js` | Interactive autosave coordinated with whole-game publication |
| `auth.jsx` | Local player context and game setup with backend defaults |
| `hooks.js`, `contexts.js`, `components.js`, `analysis-components.js`, `lib.js` | Narrow exports of reused upstream functionality |
| `telemetry.js` | Disable hosted analytics |

Transforms fail when expected upstream anchors change. Review adapters and tests when updating the submodule. Keep requested limits, achieved depths, candidate coverage, terminal outcomes and complete search status intact through conversion and save/restore. Complete compatible results satisfy smaller requested depth and time limits even when the achieved depth is lower; the backend owns selection and promotion of cached evidence. Achieved depth is displayed separately.

## Configuration

See [project configuration](../README.md#configuration) for shared engine and analysis settings. The frontend obtains resolved engine settings from the backend instead of maintaining separate search defaults.

| Setting | Purpose |
| --- | --- |
| `FRONTEND.STATIC_DIR` | Production build directory served by Python |
| `FRONTEND.BUILD` | Node discovery, required version, distribution and build-tools location |
| `FRONTEND.STOCKFISH_TIME_SCALE_BY_DEPTH` | Web presets scaling both evaluation default and maximum seconds |
| `SERVER` | Backend address and saved-game/output storage |
| `MAIA`, `ANALYSIS` | Shared model settings, search limits and worker resources |

Only presets within the evaluation depth ceiling are advertised. They apply to interactive and whole-game web analysis, not coach budgets or exploration previews. Search scheduling and evidence reuse are documented in [analysis](../analysis/README.md).

## Development

With the project installed, run `pnpm dev` from `web/` and start the Python backend separately using the [root instructions](../README.md#install-and-run). Vite proxies `/api` to port 5000; `MAIA_BACKEND_URL` changes only this development proxy. Rebuild production assets after frontend changes and restart Python after backend changes.

Browser timing capture requires an enabled backend profiler and `?profiler=1` on the analysis URL. This disables automatic interactive requests and selects isolated cold full-game profiling. It cannot make a previously loaded Maia model cold. Cache isolation, recorder ownership and outputs are described by the [backend](../backend/README.md#development) and [analysis](../analysis/README.md) components.

## Verification

Run from the project root with the supported Node runtime and project Python environment:

```powershell
node --test tests/web/*.test.mjs
python web/build.py
```

Frontend regressions check adapters, source transforms, PGN export, routing, streams, search completion and restoration. The build checks the local TypeScript entry point and bundles upstream TypeScript; it is not a full upstream Next.js type-check. Python build-helper tests and broader checks are covered by the [shared test guide](../tests/README.md). Stop servers started for manual verification.

Upstream code and assets retain their GPL-3.0 notices; see [LICENSE](../LICENSE) and the submodule license.
