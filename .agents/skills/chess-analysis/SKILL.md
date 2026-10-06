---
name: chess-analysis
description: Develop the shared game-analysis pipeline, accuracy metrics, rating estimators, caches, or runtime profiler in this chess repository. Use for analysis component changes and estimator experiments; coaching prose and engine adapters have separate skills.
---

# Chess analysis

Apply the [project guidance](../project-development/SKILL.md). Read [analysis/README.md](../../../analysis/README.md) for the current contracts and configuration before changing them; source and configuration determine present values, not historical experiment settings.

## Component boundaries

- Web and coach must share the same full-game pipeline, including evidence, hints, performance statistics, and both players' rating fits. A frontend that does not display a result still uses the complete shared calculation.
- Keep actual analysis in `analysis/`; `engine/` owns model inference, engine adapters, and process operations. Backend routes orchestrate requests rather than duplicating analysis.
- Group game functionality in `analysis/game/`, rating functionality in `analysis/player_rating/`, and timing tools in `analysis/profiler/`. Use `profiler` for profiling functionality and `runtime profiler` for execution-time measurements.
- Apply the common configuration ownership: analysis owns shared search budgets and worker resources; basic engine startup and frontend presets stay with their own sections.

## Evidence and metrics

- Calculate cheap per-move hints during game analysis and store only short labels in each move's `flags` array in `analysis.json`. Empty arrays are valid; do not duplicate hint evidence in unrelated artifacts.
- Reuse the existing Lichess accuracy calculation and judgment metric for inaccuracies, mistakes, and blunders. Preserve both game and phase performance statistics for reports.
- Distinguish arithmetic mean move accuracy used by a rating method from Lichess full-game accuracy, which uses a different aggregation. Label them correctly rather than silently substituting one for the other.
- The observed statistic and the expected Maia curve must measure the same quantity and use consistent position selection.
- Do not treat a forced move's perfect accuracy as evidence of playing strength. Preserve the estimator's explicit forced-position policy.
- Maia policies describe human move likelihood; Stockfish scores describe objective quality. Keep both available rather than collapsing them into one interpretation.
- Cache reusable measurements independently of estimator settings and output directories. Refit from valid same-game evidence without rerunning engines merely because the method, account rating, or display scale changes.

## Rating method interface

For a new fitting method or changes to a method's mathematics, use [chess-player-rating](../chess-player-rating/SKILL.md). It covers universal model design, ordering, account sensitivity, scale consistency, the drop-in API, and method evaluation without training on test games.

- `ANALYSIS.PLAYER_RATING.METHOD` selects `analysis/player_rating/<METHOD>.py`; that file supplies `Rating(PlayerRating)` implementing the shared abstract interface.
- A new conforming method should require its file and configuration selection, not edits to application callers or a hardcoded method registry.
- Keep pure fitting separate from engine calls, PGN loading, caching, validation, and rendering. Use the service layer for those integrations.
- Method-specific mathematical arguments belong in the method's Python file. Do not turn experimental sigma, prior, interval, or probability-cutoff values into project-wide configuration or permanent skill rules.
- Record method identity, version, and arguments so changed calculations invalidate stale fits. Do not relabel an old estimate as a freshly calculated result.
- Report intervals only when the method supplies them. A narrow central display interval is not a calibrated error guarantee, and point-only decisions must not be drawn as posteriors.
- Favor universal, interpretable models based on declared evidence. Do not introduce exceptions for named games or particular moves to improve a comparison.

## Strict testing boundary

- Apply the project's strict test-only boundary to every estimator, including unlabeled accuracy curves, moments, normalizers, covariance estimates, and derived assets. Only the current game's evidence and declared inputs may enter its inference; leave-one-out construction is not an exception.
- Commercial estimate headers are comparison labels only, joined after predictions. They must never be estimator inputs or targets for automatic coefficient fitting.
- Requested comparisons may inform discussion of method behavior or explicitly requested minor adjustments; they do not turn the collection into independent validation after repeated selection.
- Do not restore withdrawn population assets or disabled methods as a shortcut to passing historical tests.
- Preserve interpretable diagnostics such as shared-curve intersections, observed accuracy, and variance. Investigate differences without treating a commercial reference as unquestionable ground truth.

## Rating coordinates

- Maia's native rating anchors are Lichess Blitz. Read PGN `Site` and `TimeControl` when determining the account and output scale; do not assume Chess.com Rapid ratings are native Maia ratings.
- Use the common scale-conversion code, including the declared site/time-control classification. Lichess duration is initial seconds plus 40 increments.
- Normalize inputs once into the native scale and convert decisions and display coordinates back once. Preserve approximate consistency when the same game and account inputs are represented in another supported scale.
- Keep scale provenance and explicit extrapolation or missing-metadata assumptions visible. Do not refit conversion coefficients from test-game references.

## Saved results and figures

- When application fitting or saved-analysis refresh is performed, regenerate the rating figures and replace known old generated versions, including on cache hits. Keep the pure numerical fit interface free of file output.
- Rating charts are SVG. Preserve unrelated user diagrams; do not clear an output directory indiscriminately.
- Put the shared accuracy curve beside a combined White/Black result panel. Mark curve intersections separately from final estimates and show applicable uncertainty parameters.
- Show the prior in a separate compact chart. Distinguish unnormalized native prior weights from a transformed density on another rating scale.
- Use the same accuracy scale across comparable figures and the declared rating coordinates. Consult the current renderer for established plot limits rather than fixing obsolete experiment settings here.

## Verification and speed work

- Keep experiments and regressions in `tests/analysis/`, using the [tests skill](../chess-tests/SKILL.md). Use current-game or synthetic evidence for mathematical invariants.
- Prefer batched Maia inference, available GPU execution, and configurable concurrent Stockfish workers when improving speed; measure rather than assume the benefit.
- Record component timing and cache behavior. Parallel work totals are not wall-clock duration, and a batch's time is not independently measured time for each row.
- Compare quality as well as speed when changing search budgets. A shorter run or different evaluation alone does not establish a better analysis.
- Running all games or starting engines is task-dependent work, not an automatic consequence of loading this skill.
