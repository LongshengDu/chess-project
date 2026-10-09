# Maia Local Chess

Play against Maia, analyze games with local Maia and Stockfish engines, and get a chess coaching report from Codex. The web application uses the Maia platform's analysis and play pages with a local Python backend. Web and coaching share one full-game analysis pipeline.

## Install and run

Install Python 3.10+, Git and uv, then run these commands from the repository root:

```powershell
git submodule update --init --depth 1
uv sync
uv run python -m engine.assets
uv run python web/build.py
uv run python backend/app.py
```

Open <http://127.0.0.1:5000>. Stop the application and its engines with **Ctrl+C**. Later starts only need `uv run python backend/app.py`; rebuild after frontend source changes. The build helper finds Node.js 22+ or downloads verified tools into `web/.tools`, so no global pnpm installation is required.

All Python components share the root environment, [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). Initial setup downloads dependencies, Maia checkpoints and Stockfish.

For NVIDIA GPU inference:

```powershell
uv sync --extra cuda
uv run --extra cuda python backend/app.py --device cuda
```

Keep `--extra cuda` on subsequent uv runs to retain the CUDA installation. Maia uses the GPU; Stockfish uses CPU workers. Startup reports the selected device and worker configuration.

## Usage

### Web play and analysis

**Analysis** is the landing page. Choose **Select Game → Custom → Analyze Custom PGN/FEN** to import a game or position, then explore moves, evaluations, moves by rating and alternative lines. **Analyze Entire Game** saves analysis for both players using the shared pipeline. Coaching reports are generated through the CLI.

**Play Maia** opens game setup: select a side, Maia rating and time control, then play locally and open the saved game in analysis. Exported PGNs use the selected Maia rating for both `WhiteElo` and `BlackElo`, `Site "lichess.org"`, and the selected time control. Unlimited games export `TimeControl "300"` for analysis while the live game remains untimed.

### Analysis and coaching

Run from the repository root, replacing `game.pgn` with your PGN path. Coaching uses Codex with your ChatGPT sign-in; run `codex login` if needed. Local analysis requires no LLM account.

```powershell
# Analyze locally, reusing compatible cached measurements.
uv run python coach/coach.py game.pgn --analysis-only

# Analyze and coach White; use --side black to coach Black.
uv run python coach/coach.py game.pgn --side white

# Generate another report from the existing analysis.json.
uv run python coach/coach.py game.pgn --side white --coach-only

# Rebuild saved analysis strictly from its original cache, without engines.
uv run python coach/coach.py game.pgn --analysis-only --rebuild-from-cache
```

Normal analysis reuses matching cached evidence and computes missing requests. `--rebuild-from-cache` regenerates analysis from saved measurements without engines and stops if required evidence is missing. `--coach-only` skips the full-game pass, but coaching tools can investigate new positions. Add `--refresh-cache` only when you explicitly want fresh engine measurements; it cannot be combined with `--rebuild-from-cache` or `--coach-only`.

Without rating flags, the CLI uses PGN `WhiteElo`, `BlackElo`, `Site` and `TimeControl`. To override both players' ratings, supply `--elo 1600 --rating-scale lichess_blitz` together. Supported scales are `lichess_blitz`, `lichess_rapid`, `chess_com_blitz` and `chess_com_rapid` (aliases `lb`, `lr`, `cb`, `cr`). Missing or invalid rating context stops the run and asks for both flags. The original PGN headers are preserved, and the same analysis can coach either side.

Output defaults to `<PGN parent>/output/<PGN stem>-full`; `--output-dir` changes the destination without moving caches. Analysis writes `analysis.json`, a copy of the PGN, `accuracy-curve.svg`, and `accuracy-by-move-1600.svg`, `accuracy-by-move-1800.svg`, `accuracy-by-move-2000.svg`. Coaching adds `coaching.md`, diagrams and run logs. Use `uv run python coach/coach.py --help` for all options.

## Configuration

Edit [config.yaml](config.yaml) and restart the application. Paths resolve from the project root; explicit CLI options override YAML. Application settings do not use environment-variable overrides.

| Section | Settings |
| --- | --- |
| `MAIA` | Model, device, checkpoint, batching and local play |
| `STOCKFISH` | Executable and downloaded engine assets |
| `ANALYSIS` | Shared evidence cache, Stockfish workers, threads and hash per worker, search strategy and limits |
| `COACH` | Codex model, response/token/time budgets and progress |
| `SERVER` | Host, port, request limits and saved-game storage |
| `FRONTEND` | Analysis preset scales, static files and build tools |

YAML durations use seconds and search depth uses plies. Maia inference and Stockfish analysis resources are independent. Web presets scale the shared evaluation limits; coaching uses the unscaled limits. Components read their own configuration directly or through their local `settings.py`.

## Project structure

Component READMEs contain development documentation, module inventories, interfaces and checks.

| Directory | Responsibility |
| --- | --- |
| [analysis](analysis/README.md) | Shared game and position analysis, accuracy curves, move hints, evidence cache and profiling |
| [backend](backend/README.md) | HTTP API, play sessions, saved games and server lifecycle |
| [coach](coach/README.md) | Codex investigations, chess tools, report generation and usage accounting |
| [engine](engine/README.md) | Maia inference, Stockfish processes, worker pools and asset downloads |
| [web](web/README.md) | Maia frontend integration, local adapters and build tooling |
| [tests](tests/README.md) | Component tests, bounded live checks, benchmarks and preserved experiments |
| [docs](docs/) | Standalone technical papers, including [Accuracy Curves](docs/accuracy_curves.md) |
| [deps](deps/) | Pinned Maia, Maia platform frontend, Lichess Lila and opening-database submodules |
| [games](games/) | PGNs and generated game outputs |
| [.agents/skills](.agents/skills/project-development/SKILL.md) | General project and component guidance for development agents |

## Design and data

```text
Web pages → Backend ─┐
                    ├→ Shared analysis → Maia / Stockfish
Coach CLI ──────────┘         │
                              ├→ Saved evidence and accuracy figures
                              └→ Codex chess tools → Coaching report
```

Analysis owns chess calculations and reusable evidence; engines own model inference and native processes. The web server and CLI share code and disk caches, while each application closes the processes it creates. Codex receives prepared evidence and requested tool results for coaching.

Accuracy curves describe expected move quality at Maia's native Lichess Blitz anchors from 600–2600. Analysis also records actual arithmetic accuracy, Lichess game accuracy and probability-weighted absolute deviation. Curve intersections are descriptive coordinates, not player-rating estimates. The [technical paper](docs/accuracy_curves.md) explains these measurements.

| Default location | Contents |
| --- | --- |
| `engine/.cache/maia3` and `engine/.cache/stockfish` | Downloaded checkpoints and engine binaries |
| `analysis/.cache/positions/<FEN hash>.json` | Shared Maia/Stockfish evidence, one file per position with distinct history and search contexts |
| `analysis/.cache/games` and `analysis/.cache/game-metadata` | Saved-game measurement references and analysis bookkeeping |
| `backend/.cache/analysis.sqlite3` | Durable web studies, played games, favorites and saved analysis |
| `backend/output/<game ID>-full` | Web game analysis artifacts |
| `<PGN parent>/output/<PGN stem>-full` | CLI analysis, figures and coaching output |
| `web/dist` | Built frontend served by the backend |

Caches, web history and generated reports are separate. Interactive analysis, full-game analysis and coaching reuse compatible position evidence; saved games retain references to their original measurements. See the [analysis development guide](analysis/README.md) for the cache and saved-data contracts.

Stockfish reuse compares requested depth and time, not achieved depth. A compatible complete result covers a request when both limits are at least as large. Otherwise, the next search keeps the maximum of each limit and includes all required candidates. Engine identity, search strategy and relevant history must match. Runs with `--rebuild-from-cache` stop on missing evidence; they never start engines.
