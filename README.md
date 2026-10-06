# Maia Local Chess

A local chess application with Maia play, full-game analysis and a separate
Codex coaching CLI. The sole frontend imports the pinned Maia platform pages and
runs against local Python engines. **Analysis** is the landing page;
**Play Maia** opens game setup.

## Install and run

Requirements: Python 3.10+, Git, uv and a browser. Initial setup downloads
dependencies and engine assets. Run from the repository root:

```powershell
git submodule update --init --depth 1
uv sync
uv run python -m engine.assets
uv run python web/build.py
uv run python backend/app.py
```

Python dependencies for the entire application, including coaching, are declared
in root [pyproject.toml](pyproject.toml) and resolved in [uv.lock](uv.lock).
Components share the root `.venv`; there are no component-specific Python
requirements or installation steps.

The root dependencies also retain Stockfish's Python wrapper and notebook
support for the preserved experiments. Application engine operations use
`python-chess` directly. Historical serial timing charts require the optional
`profiling` extra (`uv sync --extra profiling`, adding `--extra cuda` for GPU
installations); current full-game timing charts use built-in SVG generation.

Open <http://127.0.0.1:5000>. Stop the application and its engines with **Ctrl+C**.
After setup, normal startup only needs `uv run python backend/app.py`. Rebuild
the frontend after changing its source.

For NVIDIA GPU inference, install and retain the CUDA extra:

```powershell
uv sync --extra cuda
uv run --extra cuda python web/build.py
uv run --extra cuda python backend/app.py --device cuda
```

Include `--extra cuda` on subsequent uv runs to retain that PyTorch installation.
Maia can use the GPU; Stockfish uses CPU workers. The startup message reports the
selected device and worker configuration. The build helper uses existing Node.js
22+ or downloads verified tools into `web/.tools`; no global pnpm is required.

## Applications

**Web play and analysis.** Choose **Select Game → Custom → Analyze Custom PGN/FEN**
to import standard-chess PGN or a six-field FEN. The upstream page provides the
board, moves, analysis, moves by rating, options, export and learning from mistakes.
**Analyze Entire Game** runs the same Python pipeline as the coach: Maia policies,
Stockfish searches, move hints, Lichess accuracy and played-strength fitting for
both players. The complete result is saved, including ratings that the current
page does not yet display. Coaching report generation remains a separate action
in the CLI.

**Play Maia** provides side/rating/time-control choices, custom starting positions,
clocks, premoves, promotion, resignation and rematches. It uses the configured
local model. Saved games can open directly in analysis. Local play has no account
rating system.

**Coaching CLI.** The separate [coach](coach/README.md) prepares local evidence,
estimates played strength and uses Codex with the user's ChatGPT sign-in to write
a report. Begin with local analysis:

```powershell
python coach/coach.py games/game1.pgn --side white --elo 1600 --analysis-only
```

See the coach guide for installation, full-report commands, saved-analysis reuse,
tools and budgets. Output defaults to `<PGN parent>/output/<PGN stem>-full`;
`--output-dir` overrides it. Reusable caches remain separate from game output.

## Components and documentation

Each runtime component directory has one Markdown guide, named `README.md`.
Test documentation is consolidated in `tests/README.md`; directories under
`tests/` do not contain their own READMEs or individual test inventories.
The root guide covers setup and cross-component conventions; detailed contracts
belong to their component. Standalone technical papers remain in `docs/`.
Runtime prompts use `.txt` resources. Dependency documentation, game reports,
saved benchmark output and user experiments retain their original files.

Project-wide Python dependency declarations, lockfiles and tooling configuration
belong at the repository root. Dependency-owned manifests stay inside `deps/`.

| Component | Responsibility and guide |
| --- | --- |
| [backend](backend/README.md) | Flask application, local API, persistence, request validation and lifecycle |
| [engine](engine/README.md) | Maia inference, Stockfish workers, UCI processes and engine assets |
| [analysis](analysis/README.md) | Shared game evidence, search, accuracy, hints, player ratings and profiling |
| [coach](coach/README.md) | Codex investigations, evidence selection, coaching policy and report generation |
| [web](web/README.md) | Upstream frontend integration, local adapters, build tools and performance |
| [tests](tests/README.md) | Component regression suites, fixtures, development checks and preserved experiments |

Reusable agent instructions live in [.agents/skills/project-development/SKILL.md](.agents/skills/project-development/SKILL.md), which routes to the [analysis](.agents/skills/chess-analysis/SKILL.md), [player-rating methods](.agents/skills/chess-player-rating/SKILL.md), [backend](.agents/skills/chess-backend/SKILL.md), [coach](.agents/skills/chess-coach/SKILL.md), [docs](.agents/skills/chess-docs/SKILL.md), [engine](.agents/skills/chess-engine/SKILL.md), [games](.agents/skills/chess-games/SKILL.md), [tests](.agents/skills/chess-tests/SKILL.md), and [web](.agents/skills/chess-web/SKILL.md) skills. These capture lasting project instructions; current tasks and configuration remain the source for specific actions and parameter values.

Dependency sources are pinned submodules under `deps/`: Maia, the Maia platform
frontend and Lichess Lila. Update revisions deliberately and check
the compatibility transforms when updating the frontend.

The default **shared-curve affine method** uses only the current game's Maia policies, move qualities and supplied account ratings. It converts arithmetic accuracy through that game's shared curve under a common account-centered prior; no population asset or other game's evidence enters the fit. The [shared-curve affine paper](docs/shared_curve_affine.md) derives the method; the [analysis guide](analysis/README.md#configuration-and-ratings) covers configuration and the retained [Bayesian shared-curve method](docs/bayesian_shared_curve.md). The [rating extension contract](analysis/README.md#player-rating-estimator-interface) describes filename-selected methods.

**Benchmark games are evaluation-only.** Their positions, policies, quality distributions, observed accuracies and reference ratings must not supply training, calibration, priors or population assets for another game's estimator, even without labels or with target-game exclusion. The population-based `hierarchical_affine` and `uncertainty_ensemble` implementations and papers have been removed; `shared_curve_affine` is the current-game-only replacement. `arithmetic_coverage` remains disabled. Affected historical test comparisons remain withdrawn and do not establish test-only performance or justify promotion.

## Configuration

Edit [config.yaml](config.yaml), then restart the application. Relative configured
paths resolve from the project root. Explicit CLI options take precedence;
application settings do not use environment-variable overrides.

| Root section | Ownership |
| --- | --- |
| `MAIA` | Model, checkpoint, device, batch size, playing rating, sampling and inference caches |
| `STOCKFISH` | Executable, binary cache, startup and asset downloads |
| `ANALYSIS` | Shared evidence cache, search strategy/time/depth, Stockfish workers/threads/hash, exploration and rating method |
| `COACH` | Codex model, response/token/time budgets and progress |
| `SERVER` | Host, port, request limit and database |
| `FRONTEND` | Web analysis preset scales, static build location, and Node/build-tool settings |

Every YAML duration is in **seconds**; search depth is in **plies**. Stockfish
time and depth are stopping limits, not a total game timeout.
`STOCKFISH_THREADS_PER_WORKER` applies to each independent worker;
`STOCKFISH_HASH_MB_PER_WORKER` applies to each worker, so pool hash memory is
worker count multiplied by this value (4 × 128 MB = 512 MB by default).
`MAIA.PLAYER_RATING` configures play, whereas `ANALYSIS.PLAYER_RATING` selects
the estimation method. Method-specific numerical settings remain in the selected Python module; the default returns points without an interval.

`ANALYSIS.STOCKFISH_EVALUATION` supplies common scoring and full-game limits.
`FRONTEND.STOCKFISH_TIME_SCALE_BY_DEPTH` scales both default and maximum seconds
for web presets (12: 0.2, 15: 0.5, 18: 1.0); coach analysis uses unscaled limits.
`STOCKFISH_SEARCH_STRATEGY` selects candidate scheduling shared by web and coach.
Single evaluations and continuation previews always use depth/time bounds.

Each component's `settings.py` reads YAML directly and resolves the paths it
uses. Build tooling reads YAML directly as well. There is no global configuration
loader and no dependency on `engine/settings.py` for another component's settings.
Coach reuses the shared analysis/engine sections. Test-only settings stay in
`tests/coach/config.yaml`; dependency and framework manifests retain their native
formats.

## Architecture and stored data

```text
web: upstream Maia pages + local adapters
                    |
           /api/platform/*
                    |
backend: validation, saved games, streaming
                    |
           analysis + engine
       Maia policies / Stockfish workers

web Analyze Entire Game -> shared GameAnalyzer -> saved analysis + player ratings
coach CLI              -> shared GameAnalyzer -> Codex tools -> coaching report
```

The web server uses one configured Maia adapter for play and analysis. Each
launched application owns its engines and closes them on exit; sharing code and
disk caches does not share running processes. Source details and route contracts
are in the component guides.

| Default location | Contents |
| --- | --- |
| `engine/.cache/maia3` | Downloaded Maia checkpoints |
| `engine/.cache/stockfish` | Verified Stockfish binary |
| `backend/.cache/analysis.sqlite3` | Imported studies, played games, favorites and cached web analysis |
| `coach/.cache` | Shared web/CLI evidence; configured by `ANALYSIS.CACHE_DIR`, retaining the existing location |
| `<PGN parent>/output/<PGN stem>-full` | Per-game analysis and coaching output |
| `web/dist` | Generated frontend served by Flask |

Saved studies and caches use separate configuration entries. Choosing a profiling
or report destination does not redirect the default cache.

## Development

Keep each Python module focused on one responsibility. Stateful services own
their resources and lifecycle; pure chess calculations remain small functions.
Use direct imports from the owning module, explicit dependencies and composition
instead of compatibility reexports or application-specific branches in shared
analysis. HTTP controllers, storage, engine operations, analysis and report
rendering have separate modules within their existing components.

Name related Python modules with a shared group first and a descriptive role
second: `assets_maia.py` and `assets_stockfish.py`, `routes_analysis.py` and
`routes_play.py`, or `agent_runner.py` and `agent_budget.py`. Analysis follows
the same convention with `game_*`, `position_*`, `stockfish_*` and `profiler_*`.
Rating modules live together under `analysis/player_rating/` with short role
names such as `interface.py`, `service.py` and `evidence.py`; their documentation
stays in `analysis/README.md`. Keep clear standalone names such as `app.py`,
`settings.py`, `cache.py`, `maia.py` and `stockfish.py`; do not add a prefix
just to repeat the component directory. Each runtime component README must give
every Python module except `__init__.py` its own table row and responsibility.
Do not combine several Python filenames in one row. Test documentation covers
suites and how to run them in `tests/README.md`, without individual test-file
descriptions. Test filenames follow the module or behavior they cover.

From the project root:

```powershell
python -m unittest discover -s tests -t . -p "test_*.py"
node --test tests/web/*.test.mjs
```

Use the project's Python environment; frontend tests need dependencies installed
by `web/build.py`. See [tests](tests/README.md) for component commands,
[web development](web/README.md#development-and-validation) for frontend checks, and
[performance](web/README.md#performance) for measurements and reproduction.

Keep tests in `tests/<component>/`. Preserve user scratch code, notebooks, chess
SVGs and historical results; `tests/main.py`, `tests/notebook.ipynb` and
`tests/chess.svg` are intentional user work. Validate agent changes offline first,
then at most a small live report when needed. Full live coaching reports require
an explicit request during development. Close engines and owned processes after
checks, and leave server startup to the user.
