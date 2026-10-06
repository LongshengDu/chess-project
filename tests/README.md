# Tests and development checks

All first-party regression suites, fixtures and development tools live under
`tests/`, grouped by component. This is their shared guide; component test
directories do not have separate READMEs. Dependency-owned tests remain in `deps/`.

## Suites

| Directory | Coverage |
| --- | --- |
| `analysis/` | PGN/FEN, shared full-game results, search budgets, hints, accuracy, rating estimators, profiling and atomic caches. |
| `backend/` | Configuration, startup/cleanup, HTTP analysis and play, persistence, cancellation and full-game publication. |
| `engine/` | Maia batching/history, Stockfish limits and worker concurrency, asset verification and cleanup. |
| `coach/` | Evidence, chess investigations, report validation, diagrams, agent budgets, progress and output paths. |
| `web/` | Build tooling, routing, frontend adapters, streaming, terminal cache restoration, autosave and upstream compatibility. |

Shared-analysis checks cover identical web/coach evidence and ratings, cancellation
without disrupting unrelated jobs, and history-sensitive caching. Rating checks
use frozen evidence and mathematical invariants. For new estimators, follow the
[extension contract](../analysis/README.md#player-rating-estimator-interface).
Benchmark games are evaluation-only. Neither their labels nor their unlabeled
positions, policies, quality distributions or observed statistics may supply
training, calibration, priors or population assets. Excluding the current target
while using the remaining benchmark games is still prohibited. Synthetic unit
fixtures and current-game inference remain valid; test fixtures must never become
runtime dependencies. Commercial references enter scoring only after predictions.

**Withdrawn research:** earlier population-based rating comparisons reused these
benchmark games as a fitting corpus. Their results and associated promotions
are withdrawn as evidence of test-only performance. The affected papers have
been replaced by the current-game-only [shared-curve affine paper](../docs/shared_curve_affine.md).
Historical experiment scripts and results remain for inspection, not as endorsed
methods or external validation. The production population asset and the
hierarchical-affine and uncertainty-ensemble implementations have been removed.
Do not rebuild that asset to rerun an archive or make a retired test pass.

## Offline tests

Run from the repository root using the project's Python environment and Node
dependencies installed by `python web/build.py`:

```powershell
python -m unittest discover -s tests -t . -p "test_*.py"
node --test tests/web/*.test.mjs
```

Run one Python suite with
`python -m unittest discover -s tests/<component> -t . -p "test_*.py"`.
Keep `-t .` so package imports resolve from the repository root. `pnpm test`
inside `web/` runs the same Node suite; no separate Node project is needed under
`tests/web/`.

Regressions use fake engines, scripted model responses, temporary files/databases,
mocked downloads and Flask's in-process client. They do not call a live coaching
model, start listening servers or download model assets. Build regressions mock
processes rather than installing tools or performing a frontend build.

## Saved-game rating refresh

The production default is `shared_curve_affine`. It uses the current game's shared arithmetic accuracy curve and conditional variance, plus a common account-centered prior. It does not load another game, a population curve or a calibration asset. `bayesian_shared_curve` remains an asset-free alternative using the same current-game evidence.

To refit saved games with the configured method and replace their rating figures:

```powershell
python -m tests.analysis.compare_shared_curve_games
```

The runner validates saved engine evidence and uses fresh PGN rating context. It refreshes `games/output/<game>-full/analysis.json` and the game's `player-rating/` figures without engine searches or coaching calls. Old analyses are preserved under the comparison output. Commercial headers enter evaluation only after predictions; previous retired estimates are historical snapshot values, not recomputed baselines.

The main SVG compares the game's accuracy curve with both players' results. The default displays affine point decisions without a posterior interval; the Bayesian method displays its conditional posterior. The separate prior figure uses the declared rating scale, with density transformed when required. Output rating coordinates follow the PGN's site and time-control scale; inference remains in native Lichess Blitz coordinates. Output is SVG, with obsolete renderer-owned files replaced and unrelated images preserved.

`--games-dir` and `--output-dir` select input and comparison locations. `--figures-only` renders compatible saved results without changing analysis bytes. It does not make a retired method valid or relabel old estimates as current. The scale-conversion tests independently cover inference boundaries, extrapolation, cache reuse and coordinate consistency.

The old `rating-scales`, `rating-methods-games0-18`, hierarchical and uncertainty comparisons are preserved archives. Their corpus-based entries do not satisfy the test-only requirement; they are not current refresh procedures. Existing mathematical Bayesian-paper illustrations may still be rendered from their own fixed example evidence; they must not provide reusable fitting inputs for other games.

## Top-probability shared curve

The top-probability experiment is run with:

```powershell
python -m tests.analysis.experiment_shared_curve_top_probability --top-probability 0.68 --sigma-scale 0.5
```

At each position and rating, it keeps the probability-ranked moves through
cumulative probability strictly above the cutoff, includes all ties at the last
probability, and renormalizes. Both expected accuracy and adaptive variance use
that distribution. Actual played moves remain in observed accuracy even when
outside the selected set. Only positions with a single legal move are omitted
from both observed and expected averages. The runner now uses the production
fourth-power prior, flat on 800–2400, with zero-weight endpoints configured by
`--prior-range` (default 200 and 3000). There is no midpoint-weight parameter.
The conditional central
interval remains 20%.

`tests/analysis/output/shared-curve-top-probability-current-prior/` contains a JSON/CSV table,
per-game fit data, and SVG figures. A full-probability control uses the same
sigma multiplier, prior, and forced-position rule. Retained candidate counts,
probability masses, raw curve reversals, and isotonic adjustments are recorded.
Reference headers are read only after inference; parameters are not fitted to
them. Production settings, game analyses, and the paper are unchanged. No engine
or coaching calls are made. The estimator is isolated in
`tests/analysis/shared_curve_top_probability.py`; the runner is
`tests/analysis/experiment_shared_curve_top_probability.py`.

## Lichess-accuracy shared-curve experiment

```powershell
python -m tests.analysis.experiment_shared_curve_lichess
```

This isolated experiment compares an arithmetic shared curve with a
Lichess-accuracy curve on all saved games. It includes all played positions,
including forced moves. At each position and rating, it retains the top 99%
probability mass with cutoff ties and renormalizes the candidate probabilities.
For candidate accuracy `Q`, fixed actual-game volatility weight `w`, and `n`
positions, the expected score is
`0.5 * (sum(w * E[Q]) / sum(w) + n / sum(E[1 / max(1, Q)]))`.
The observed score uses the same formula on the played moves and is checked
against the local Lichess full-game implementation. Saved volatility weights
are also independently reconstructed and checked.

The harmonic term is an expected-reciprocal approximation, not the exact
expectation of a random game's harmonic mean. It keeps the actual game's
positions and volatility weights; it does not simulate alternative game paths.
An independent-position first-order delta-method variance supplies the adaptive
Gaussian sigma, including the covariance between candidate accuracy and its
floored reciprocal. Both fits retain top probability 0.99, sigma scale 1.0 and
the same production prior. These explicit 99% settings are experimental; production
now defaults to all legal moves. The exponential tails, monotone curve fitting and
posterior calculation are reused. Reference headers are read
only after both fits; no parameters are fitted to the references.

The estimator is `tests/analysis/shared_curve_lichess.py`; its runner writes
CSV/JSON results, per-game experimental and arithmetic fits, combined accuracy
and posterior SVGs, and separate unnormalized prior SVGs under
`tests/analysis/output/shared-curve-lichess-top99-current-prior/`. `--output-dir` must stay
under `tests/analysis/output/`. Production code, configuration, saved analyses,
rating fits, figures and evidence are checked with SHA-256 before and after.
No engine or coaching calls are made, and production outputs are not refreshed.

## Arithmetic shared-curve parameter comparison

```powershell
python -m tests.analysis.experiment_shared_curve_sweep
```

This separate experiment uses the current arithmetic-accuracy estimator on all
saved games with the fixed grid of top probabilities 95%, 96%, 97%, 98%, 99%,
and 100% (the full legal-move policy), crossed with sigma scales 0.5 and 1.0.
These are the original defaults. Supply `--top-probabilities` and `--sigma-scales`
for another explicit grid, with a separate output directory to preserve older runs:

```powershell
python -m tests.analysis.experiment_shared_curve_sweep --top-probabilities 0.99 1.0 --sigma-scales 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 --output-dir tests/analysis/output/shared-curve-arithmetic-sigma-sweep
```

The existing prior, curve extension, forced-move exclusion, and posterior median
remain unchanged. A game qualifies only when **both** players' observed arithmetic
accuracies intersect its monotone measured curve within 600–2600, including the
endpoints. Intersections that require extrapolated tails do not qualify.

The primary ranking uses the intersection of qualifying game sets across all
requested settings. A secondary ranking uses each setting's own qualifying games and
records the count, preventing changes in game selection from being hidden.
Ranking uses mean absolute error against commercial PGN estimates, followed by
RMSE, maximum error, and a stable variant name for ties. Reference ratings never
enter inference or the intersection filter; this is a requested benchmark
comparison, not an independent validation of the winning parameters.

`tests/analysis/experiment_shared_curve_sweep.py` saves `comparison.json`,
`ranking-common.csv`, `ranking-individual.csv`, `players.csv`, a ranking SVG, and
each variant/game's `fit.json`, `analysis.svg`, and unnormalized `prior.svg` below
`tests/analysis/output/shared-curve-arithmetic-sweep-current-prior/`; earlier sweep
outputs keep their recorded prior. Per-game curves and
posteriors are rendered with four local processes. File hashes check that
production code, configuration, PGNs, saved analyses, fits, figures, and cached
evidence remain unchanged. There are no engine or coaching API calls.

## Current rating checks

The active default is `shared_curve_affine`. Its checks cover the current-game curve and conditional variance, translated common prior, affine moment calculation, monotone within-game ordering, missing or uninformative evidence, rating-scale conversion, saved-fit refresh and point-only output. No test-derived population asset is required. Missing legacy assets must not be restored as a test workaround.

A valid whole-collection evaluation runs each game independently: the estimator receives that game's numeric evidence and permitted account inputs, then the runner joins reference ratings to completed predictions. Altering or removing other games must not change its result. Prior results that used any part of the benchmark collection as reusable fitting data are not valid baselines for a clean test-only comparison.

| Maintenance module in `analysis/` | Purpose |
| --- | --- |
| `refresh_current_game_ratings.py` | Discover all game PGNs and recompute their saved ratings and figures from each game's own matching cached evidence, even when fit signatures match; verify PGNs, non-rating analysis fields and caches remain unchanged, without scoring references or running engines |
| `retired_population.py` | Reject withdrawn population-building and corpus-dependent research entry points before their original file access |
| `shared_curve_affine_paper_figures.py` | Generate the shared-curve affine paper's explanatory SVGs from a declared synthetic curve, with no PGN or cached game inputs |
| `rating_evidence_fixture.py` | Build synthetic policies and move qualities for rating invariants, without loading games or calibration data |

The current-game-only refresh runs with `python -m tests.analysis.refresh_current_game_ratings` and records its preservation checks under `tests/analysis/output/shared-curve-affine-current-game-only/verification.json`.

Ordering acceptance has one direction: equal commercial White/Black ratings permit a fitted gap strictly below 50 Elo; any nonzero commercial difference requires the same fitted sign, even if the fitted gap is smaller than 50. Above/below-actual classification is a separate exact-sign diagnostic. Neither rule supplies inputs to the estimator.

## Withdrawn rating research archive

Earlier experiments explored population curves, context variance, likelihood mixtures, common account priors, alternative accuracy metrics and paired decisions. Their scripts, numerical outputs and figures remain under `tests/analysis/` and `tests/analysis/output/`. Where another benchmark game's measurements entered fitting, the test-only separation was violated even without commercial labels and even with target exclusion. Those comparisons and resulting promotions are withdrawn. Historical numbers must not be presented as independent validation or as results of the new default.

| Preserved output family | Historical scope and status |
| --- | --- |
| `edge-rating-methods`, `universal-rating-methods` | Edge likelihoods, population moments, noise assumptions and ensembles; corpus-based comparisons withdrawn |
| `simple-rating-restart`, `joint-rating-directions` | Simpler accuracy maps and ordering diagnostics; affected corpus-based candidates withdrawn, archived cohort sizes remain distinct |
| `lichess-blitz-methods`, `rating-methods-games0-18` | Earlier scale/reference comparisons including retired corpus methods; preserved historical records |
| `intuitive-curves`, `rating-scales` | Shared-curve candidate and scale-aware comparisons; corpus-based selection/promotion withdrawn |
| `hierarchical-affine-production`, `arithmetic-coverage-production` | Historical promotion/parity outputs; implementation agreement did not establish valid evaluation separation |
| `hierarchical-accuracy` | Lichess aggregation, center/contrast and joint-feature trials using a test-derived population; not valid test-only evidence |

`hierarchical_affine` and `uncertainty_ensemble` have been removed from production; `shared_curve_affine` is the current-game-only replacement. `arithmetic_coverage` remains disabled. The old paper renderers and population-asset builder are historical tools, not supported regeneration or deployment commands; the obsolete papers and their dedicated figures have been removed. In particular, `build_rating_calibration.py` must not rebuild a runtime asset from benchmark fixtures. Removing labels, anonymizing contexts, hashing inputs or leaving one game out does not repair the boundary violation.

### Accuracy-measurement research retained for inspection

The scalar and joint measurement helpers can still illustrate mathematical properties on synthetic inputs. They do not authorize corpus construction or another benchmark-dependent rating trial.

| Research module in `analysis/` | Mathematical purpose |
| --- | --- |
| `lichess_measurement.py` | Arithmetic and Lichess aggregates, exact policy moments, plug-in approximations and policy-draw covariance on supplied positions |
| `hierarchical_accuracy_fusion.py` | Native pair-center replacement retaining an arithmetic contrast, with explicit clipping and missing-data behavior |
| `hierarchical_joint_accuracy.py` | Historical two-feature covariance and affine derivation; corpus-based rating application withdrawn |
| `experiment_hierarchical_accuracy.py` | Historical benchmark/corpus runner; its saved comparisons are withdrawn |

Lichess volatility weights remain conditional on the reached game path. Independent policy draws do not form alternative legal games. Expected aggregates differ from aggregates of expected move qualities; numerical integration error differs from conditional variance. Center/contrast constraints do not establish a joint posterior. These mathematical distinctions remain useful despite withdrawal of the corpus-based rating comparisons.

Any future population or learned model requires an independently sourced development corpus kept separate from benchmark games, including overlapping games and derived measurements. A source hash or a claim of label-free fitting is not a substitute for that separation. No such corpus is currently used by the default estimator.

## Small live report

Validate offline first. During development, use at most this short live check
when needed; generating a full live coaching report requires an explicit request.
Do not repeatedly retry a failed live check. Use Codex with the existing ChatGPT
sign-in and close owned model/engine processes on exit.

```powershell
python tests/coach/smoke_test.py games/output/game2-full/analysis.json
```

This uses saved analysis and prepares one selected-player decision, including
both comparison branches. It requests one 100–180 word report with a diagram,
without repeating full-game analysis. The default output is
`tests/coach/output/game2-full-codex-smoke/coaching-smoke.md`. An explicit
`--output-dir` must differ from the source analysis directory; existing analysis
and accepted full reports are preserved. Regenerate an incompatible snapshot
with `coach/coach.py ... --analysis-only` before the check.

[coach/config.yaml](coach/config.yaml) contains isolated limits: 20,000 cumulative
tokens, 90 seconds, three local tool calls and at most 250 report words. The
scenario requests one response and one draft, with model tools disabled after
preparation. Token events can arrive after an in-flight response exceeds a limit.
The test prompt is a `.txt` resource. Production has no smoke flags or test imports;
both scenarios use the common runner through a caller-supplied `CoachingRequest`.

## Single-game speed

These opt-in measurements use local Maia and Stockfish without a coaching model
or listening server. Model assets must already be available. Engines close on
exit. With a CUDA PyTorch installation, use `uv run --extra cuda python` in place
of `python` so uv preserves that installation.

```powershell
python tests/coach/benchmark_game_speed.py games/game8.pgn --workers 4 --threads-per-worker 2 --cache-dir tests/coach/output/speed-cache --output-dir tests/coach/output/speed-cold
python tests/coach/benchmark_game_speed.py games/game8.pgn --workers 4 --threads-per-worker 2 --cache-dir tests/coach/output/speed-cache --output-dir tests/coach/output/speed-warm
python tests/coach/benchmark_web_parallel.py games/game8.pgn --output tests/coach/output/web-parallel.json
```

Use a fresh cache directory for the cold run and the same directory for the warm
run. Without an override, caches use shared `ANALYSIS.CACHE_DIR`, independently
of output paths. Full-game measurements save `analysis.json` and `timing.json`
with position/search-phase timing, model calls and rows, cache hits, workers and
device. Analysis wall time excludes startup and coaching; summed parallel search
times measure work, not elapsed time. Batched GPU time is measured as a batch,
not attributed arbitrarily to individual positions. The web-route benchmark
checks batched Maia, concurrent Stockfish streams and cached repeats in process.

Compare saved runs with the reference first:

```powershell
python tests/coach/benchmark_game_compare.py tests/coach/output/speed-serial tests/coach/output/speed-cold --output tests/coach/output/speed-comparison.json
```

The JSON/Markdown comparison includes wall/component time, scores, best moves,
achieved depths, hints and player accuracy. Changed scores alone do not establish
which run is more accurate. A one-worker run of current code does not reproduce
a retired implementation. Browser profiling and historical comparisons are
documented in [web performance](../web/README.md#performance).

## Engine benchmarks

Inspect available options without launching engines:

```powershell
python -m tests.engine.benchmark_engines --help
```

Measurements require local assets and an explicit output path. For example:

```powershell
python -m tests.engine.benchmark_engines --device cpu --maia-only --output tests/coach/output/engine-cpu.json
python -m tests.engine.benchmark_engines --strategy staged --threads 8 --max-seconds 60 --output tests/coach/output/engine-staged.json
```

These compare cold/warm Maia batches or Stockfish search strategies, depths and
elapsed times. They are separate from automatic regression discovery. Use the
same CUDA invocation rule as the full-game benchmarks, preserve previous outputs,
and close engine processes after runs.

## Fixtures and output

- `analysis/data/` contains frozen rating evidence; shared fake engines and
  scripted agent fixtures live in `coach/`.
- `coach/config.yaml` and test `.txt` prompts are isolated from production settings.
- `coach/output/` and `web/benchmarks/` hold generated measurements and preserved
  historical results. Recorded settings describe those runs, not current defaults.
- [main.py](main.py), [notebook.ipynb](notebook.ipynb) and [chess.svg](chess.svg)
  are preserved user work and are excluded from `test_*.py` discovery.

Preserve scratch scripts, notebooks, chess diagrams and existing output when
reorganizing tests. Add regressions and fixtures to the matching component.
Focused test names retain the runtime module's functional group; broader checks
name their workflow. Benchmark commands use `benchmark_*`, and small-report
helpers use `smoke_*`. Keep fixtures separate from test cases and runtime code.
