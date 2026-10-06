---
name: chess-engine
description: Maintain this chess project's Maia inference, Stockfish adapters and workers, engine assets, and process lifecycle. Use for engine operation or performance changes; keep game analysis and rating calculations in analysis.
---

# Chess engine operations

Apply the [project instructions](../project-development/SKILL.md) and consult the [engine contract](../../../engine/README.md). Read [analysis documentation](../../../analysis/README.md) when changing the interface consumed by search orchestration.

## Component boundary

- Keep `engine/` responsible for model loading, inference, Stockfish/UCI operations, worker leases, engine assets, and process cleanup.
- Keep full-game analysis, search scheduling policy, move classification, accuracy, rating estimation, and profiler reports in `analysis/`.
- Provide reusable adapters to both backend and coach. Applications own running resources; sharing code or cached assets is not sharing a live engine process.
- Apply the common naming and artifact-preservation rules; engine tests and experimental drivers belong under `tests/engine/`.

## Inference and search performance

- Use batched Maia inference across positions and requested rating pairs when available. Reuse preprocessing and legal-move masks without discarding game history.
- Use the configured GPU for Maia when supported, with a functional CPU path. Stockfish concurrency uses CPU workers and threads; do not describe ordinary Stockfish search as GPU accelerated.
- Reuse persistent Stockfish workers and their configured hash allocation. Parallelize independent position searches through the common analysis scheduler.
- Treat worker count, threads per worker, and hash memory per worker as separate resources. Consider their aggregate CPU and memory use when changing parallelism; more threads are not automatically faster.
- Preserve accurate achieved depths, termination reasons, and search limits. A time-limited result must not be labeled as reaching an unachieved target depth.
- Measure actual per-component and per-position costs before optimizing. Distinguish cold startup, warm inference, cache reuse, engine worker time, and total elapsed time.

## Configuration and interfaces

- Follow the common YAML ownership and unit conventions. Engine-local settings use `MAIA` and basic `STOCKFISH` configuration; search strategy, time/depth budgets, and worker resources belong in `ANALYSIS`.
- Share search-policy semantics across relevant Stockfish operations. Frontend depth presets scale shared evaluation limits; they are not a second engine configuration.
- Take numeric values and cache locations from current configuration, not an old benchmark or its output directory.

## Lifecycle and verification

- Read the engine README for the current initialization, history/batch alignment, lease, and cancellation contracts when changing those features; do not treat particular implementation mechanisms as permanent user requirements.
- Do not leave native engines or servers started for the task running after finishing unless the user asks to keep them running.
- Add meaningful regression coverage in `tests/engine/` for changed adapter behavior, alignment, limits, resource ownership, or cleanup. Use fakes for lifecycle checks and bounded real-engine runs only when necessary.
- Update the module table and interface/lifecycle explanations in `engine/README.md` when code structure changes; do not create extra component Markdown documents.
