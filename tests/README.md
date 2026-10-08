# Tests and development checks

First-party regression suites, fixtures, experiments and benchmarks live under `tests/`, grouped by component. This is their shared guide; test subdirectories have no separate READMEs. Use the [root installation](../README.md#install-and-run) and shared Python environment. Dependency-owned tests remain in `deps/`.

## Suites

| Directory | Coverage |
| --- | --- |
| `analysis/` | PGN/FEN, shared game analysis, search budgets, hints, accuracy/curves, profiling, prepared artifacts and concurrent caches. |
| `backend/` | Configuration, lifecycle, HTTP analysis/play, persistence, cancellation and full-game publication. |
| `engine/` | Maia batching/history, Stockfish limits and worker concurrency, asset verification and cleanup. |
| `coach/` | Compact evidence, investigations, report validation, diagrams, budgets, progress and output paths. |
| `web/` | Build tooling, routing, adapters, streaming, cache restoration, autosave and upstream compatibility. |

Regressions exercise shared production interfaces with synthetic inputs, fake engines, scripted model responses, temporary storage, mocked downloads and Flask's in-process client. Ordinary checks do not spend model tokens, download assets, start listening servers or perform frontend builds. Accuracy tests distinguish native Maia expectations, arithmetic player accuracy and Lichess accuracy; cache tests cover immutable observations, history/rating compatibility and strict reconstruction without engines.

## Run checks

Run from the repository root with the project Python environment active. Node tests use dependencies installed by `python web/build.py`:

```powershell
python -m unittest discover -s tests -t . -p "test_*.py"
node --test tests/web/*.test.mjs
```

For a focused suite, use `python -m unittest discover -s tests/<component> -t . -p "test_*.py"`; `-t .` preserves package imports from the repository root. `pnpm test` inside `web/` runs the same Node suite. See the [web guide](../web/README.md) for frontend-specific development checks and browser profiling.

## Development

Keep regressions and fixtures with their component, use meaningful behavioral assertions, and isolate output from production data and accepted results. Test filenames follow the covered module or workflow; benchmark entry points use `benchmark_*` and small-report helpers use `smoke_*`. Production code must not import test helpers or expose test-only modes.

Game collections and reference labels are evaluation-only, including unlabeled positions. They must not supply reusable training, calibration, normalization or population assets. Use synthetic inputs for invariants and each game's own compatible cached evidence for requested analysis. Preserve [main.py](main.py), [notebook.ipynb](notebook.ipynb), [chess.svg](chess.svg), scratch scripts, source PGNs and historical outputs; user work is outside `test_*.py` discovery.

The commands below are opt-in development operations. Use existing local engine assets for engine measurements, record settings and cache state, compare output quality alongside speed, and close owned processes. Validate offline before a live check; full live coaching requires an explicit request.

## Batch analysis and saved reconstruction

[analyze_pgn_batch.py](analysis/analyze_pgn_batch.py) accepts one valid standard game per supplied PGN, with a common starting position across the batch. Ordinary runs share one `AnalysisSession`, reuse compatible evidence and calculate missing requests. Existing results require `--replace-output`, which replaces generated analysis, the copied PGN and normal plots while preserving coaching reports and other files. It does not refresh or clear the cache.

```powershell
python -m tests.analysis.analyze_pgn_batch games/game0.pgn games/game1.pgn --timing-output tests/analysis/output/full-analysis-batch/run.json
python -m tests.analysis.analyze_pgn_batch games/game0.pgn games/game1.pgn --check-cache
python -m tests.analysis.analyze_pgn_batch games/game0.pgn games/game1.pgn --cache-only --replace-output --timing-output tests/analysis/output/cached-analysis-rerun/run.json
```

`--check-cache` validates all inputs and required cached evidence without engines, output changes or a timing file; existing outputs need no replacement flag for this check. `--cache-only` performs the same preflight before writing and reuses the validated sessions. It prefers each game's pinned manifest; if none exists, it requires complete compatible observations from one coherent engine profile. Missing local assets allow an unambiguous recorded profile, while missing evidence, broken references and ambiguous engine mixtures fail. See the [analysis contract](../analysis/README.md#coaching-artifacts-and-analysis-cache) for discovery rules and [root usage](../README.md#usage) for single-game workflows. `--replace-output` controls output handling alone.

Outputs use each PGN's normal game directory and include prepared `analysis.json`, the copied PGN and default accuracy SVGs. Raw observations stay in the configured analysis cache. The timing record includes configuration, cache hits, Maia execution and curve measurements; no coaching model is called. Engine ownership ends with the batch context.

## Offline cache migration and audit

[migrate_position_cache.py](analysis/migrate_position_cache.py) converts the previous normalized cache into the current readable structure without engines. With cache writers stopped, first stage and validate into a fresh directory outside the cache; `--artifacts` supplies directories containing saved `analysis.json` files whose private evidence associations must be migrated.

```powershell
python -m tests.analysis.migrate_position_cache analysis/.cache --output tests/analysis/output/cache-migration-preview --artifacts games/output backend/output
python -m tests.analysis.audit_position_cache tests/analysis/output/cache-migration-preview/staged --output tests/analysis/output/cache-audit.json
```

Migration leaves the source unchanged unless `--apply` is supplied. To publish after inspection, rerun with `--apply` and a fresh `--output` directory; the original `positions`, `games` and `game-metadata` directories remain under that output's `original/` backup. This is a one-time conversion for the previous format. [audit_position_cache.py](analysis/audit_position_cache.py) reads only the current format, checks all observations and manifest pins including inactive evidence, and writes its report without changing the cache; use `analysis/.cache` as its input to check the installed cache.

## Paper figures

Render both accuracy figures from saved measurements without engines:

```powershell
python -m tests.analysis.render_accuracy_paper
```

The [renderer](analysis/render_accuracy_paper.py) defaults to `games/output/game10-full/analysis.json`; `--analysis` changes the source, and `--figure`/`--move-figure` select the publication SVG destinations. It renders an anonymous White/Black copy, preserves source identity/evidence and discards temporary extra by-move figures. Paper examples and figures must omit real player names and handles.

## Small live report

When a live development check is necessary after offline validation, use saved analysis for one selected-player decision:

```powershell
python tests/coach/smoke_test.py games/output/game2-full/analysis.json --side white
```

This prepares both comparison branches and requests one short report with a valid diagram through the normal `run_coach` interface. It does not repeat full-game analysis, but its focused investigation can perform engine work. The default report is `tests/coach/output/game2-full-codex-smoke/coaching-smoke.md`; an explicit `--output-dir` must differ from the source analysis directory. `--side` supplies the runtime target; saved evidence has no target side. Use compatible saved analysis and the existing ChatGPT sign-in.

[coach/config.yaml](coach/config.yaml) owns isolated token, time, tool-call, output-estimate and word limits. The [scenario](coach/smoke_scenario.py) supplies a `CoachingRequest` with one response, one draft and model tools disabled after local preparation; its prompt remains a test `.txt` resource. Usage can arrive after a threshold is exceeded. Diagnose a failed live check before retrying and preserve the source analysis and accepted full report.

## Performance measurements

Engine and game benchmarks use local Maia/Stockfish, without a coaching model or listening server. With a CUDA PyTorch installation, replace `python` with `uv run --extra cuda python` so uv preserves that installation. Inspect any runner's `--help` without launching engines.

### Full-game and web routes

```powershell
python tests/coach/benchmark_game_speed.py games/game8.pgn --workers 4 --threads-per-worker 2 --cache-dir tests/coach/output/speed-cache --output-dir tests/coach/output/speed-cold
python tests/coach/benchmark_game_speed.py games/game8.pgn --workers 4 --threads-per-worker 2 --cache-dir tests/coach/output/speed-cache --output-dir tests/coach/output/speed-warm
python tests/coach/benchmark_web_parallel.py games/game8.pgn --output tests/coach/output/web-parallel.json
python tests/coach/benchmark_game_compare.py tests/coach/output/speed-cold tests/coach/output/speed-warm --output tests/coach/output/speed-comparison.json
```

Choose a fresh cache directory for a cold run and reuse it for the warm run. Without `--cache-dir`, game benchmarks use shared `ANALYSIS.CACHE_DIR`, independently of output location. Full-game runs save `analysis.json` and `timing.json` with position/search timing, model calls/rows, hits, worker counts and device. Analysis wall time excludes startup and coaching; summed parallel search time measures work rather than elapsed time, and GPU batch time is recorded as a batch. The web runner measures batched Maia, concurrent Stockfish streams and cached repeats through in-process routes.

The comparison reads saved runs, treating the first as reference, and writes JSON/Markdown covering time, scores, best moves, achieved depths, hints and player accuracy. Check each run's recorded configuration when interpreting a comparison; changed scores alone do not show which run is more accurate, and a one-worker run does not reproduce an older implementation. Keep current measurements separate from preserved historical results in `coach/output/` and `web/benchmarks/`.

### Engine strategies

```powershell
python -m tests.engine.benchmark_engines --help
python -m tests.engine.benchmark_engines --device cpu --maia-only --output tests/engine/output/engine-cpu.json
python -m tests.engine.benchmark_engines --strategy staged --threads 8 --max-seconds 60 --output tests/engine/output/engine-staged.json
```

These opt-in measurements compare cold/warm Maia batches or Stockfish strategies, achieved depths and elapsed times. They require local assets and an explicit output path and are excluded from regression discovery. [Root config.yaml](../config.yaml) supplies production defaults; explicit benchmark arguments describe the experiment rather than new application defaults.
