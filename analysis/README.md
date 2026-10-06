# Shared chess analysis

`analysis/` turns PGNs and engine measurements into reusable game evidence,
performance statistics, and played-strength estimates. The web application's
**Analyze Entire Game** and the coach CLI call the same `analyze_game` pipeline,
including rating fitting for both players. Engine inference and process management belong to
[engine/](../engine/README.md), HTTP routes to [backend/](../backend/README.md),
and coaching investigations and reports to [coach/](../coach/README.md).

## Modules

Game, profiler, and player-rating modules live in the `game/`, `profiler/`, and
`player_rating/` packages, so their filenames do not repeat the package name.
Position and Stockfish modules use the `position_` and `stockfish_` prefixes. Each
filename identifies one responsibility. The names `cache.py`, `settings.py`, `engine_session.py`,
`lichess_accuracy.py`, and `move_hints.py` already identify their standalone
roles. There are no duplicate implementations hidden behind the grouped names.

| Group | Module | Responsibility |
| --- | --- | --- |
| Game | `game/pipeline.py` | `GameAnalyzer` and `analyze_game`: the shared complete-game pipeline, including rating fitting |
| Game | `game/study.py` | PGN/FEN study import, mainline and variation navigation, interactive scoring, and PGN file loading |
| Game | `game/history.py` | Reconstruct and validate complete histories and compact branch references |
| Game | `game/context.py` | Describe position facts and the earlier decisions leading to a critical moment |
| Game | `game/summary.py` | Select balanced investigations and produce compact views of saved evidence |
| Game | `game/performance.py` | Attach Lichess game/stage statistics and refresh quality labels in saved analysis |
| Game | `game/cancellation.py` | Cancel this job's active searches without closing shared engines |
| Position | `position_display.py` | Material-display data and move sounds |
| Position | `position_evaluation.py` | Score units and losses, material stages, and rating metadata helpers |
| Position | `position_results.py` | Convert complete engine evidence into the saved/web position contract |
| Stockfish | `stockfish_search.py` | Root-move search policies, budgets, deadlines, and streamed evaluations |
| Stockfish | `stockfish_exploration.py` | Bounded continuation searches and legal-line validation |
| Player rating | `player_rating/interface.py` | Abstract `PlayerRating` contract implemented by every estimator |
| Player rating | `player_rating/service.py` | Load an estimator, prepare evidence, validate results, and refresh saved fits |
| Player rating | `player_rating/evidence.py` | Build and validate estimator-independent numeric move-quality evidence |
| Player rating | `player_rating/context.py` | Attach actual-rating inputs and before-position evaluations without changing cached engine evidence |
| Player rating | `player_rating/scale.py` | Normalize account ratings to Maia coordinates and convert fit decisions, intervals and plotting coordinates to the declared scale |
| Player rating | `player_rating/policies.py` | Batch and cache history-aware, equal-rating Maia policies |
| Player rating | `player_rating/parameters.py` | Measured Maia rating grid shared by evidence collectors and estimators |
| Player rating | `player_rating/bayesian_shared_curve.py` | Bayesian shared-curve estimator mathematics |
| Player rating | `player_rating/shared_curve_affine.py` | Default estimator: current-game arithmetic curve and conditional variance, with a common-prior minimum-MSE affine rating decision; no population asset |
| Player rating | `player_rating/arithmetic_coverage.py` | Retired corpus-based coverage estimator; blocked from runtime selection |
| Player rating | `player_rating/uncertainty_measurement.py` | Arithmetic shared curves and conditional variances, with an optional historical weighted-measurement helper |
| Player rating | `player_rating/uncertainty_likelihood.py` | Bounded Gaussian/Beta likelihoods, posterior means, and monotone native-coverage decisions |
| Player rating | `player_rating/calibration.py` | Disabled historical population-asset interface; no benchmark-derived asset is available to runtime inference |
| Player rating | `player_rating/figures.py` | Export accuracy, prior, and method-appropriate posterior or component-estimate figures |
| Profiler | `profiler/runtime.py` | Record timing events and manage a profiling run's lifetime |
| Profiler | `profiler/report.py` | Validate and summarize timings; generate CSV, Markdown, HTML, and charts |
| Profiler | `profiler/comparison.py` | Compare completed runs' timing, score, and depth tradeoffs |
| Shared services | `engine_session.py` | Cached Maia inference and Stockfish searches through owned or borrowed engines |
| Shared services | `cache.py` | Content identities, atomic JSON publication, and reusable evidence storage |
| Chess metrics | `lichess_accuracy.py` | Lichess accuracy, judgment thresholds, and game-phase calculations |
| Chess metrics | `move_hints.py` | Cheap per-move attention labels from available chess evidence |
| Rating conversion | `elo_convert.py` | Monotone four-scale conversion, PGN site/time-class inference and conversion provenance |
| Configuration | `settings.py` | Read root YAML directly for analysis consumers |

The `profiler/` package groups timing tools. `RuntimeProfiler` in `profiler/runtime.py`
records execution timing; `profiler/report.py` and `profiler/comparison.py`
summarize and compare its saved measurements.

`game/study.py` supports interactive navigation; it does not implement another
complete-game pipeline. Both applications use `game/pipeline.py` for that work.
`engine_session.py` coordinates analysis requests; the lower-level model adapters
and UCI process management remain in the `engine` component.

## Entry points

Run from the repository root:

```powershell
python coach/coach.py games/game1.pgn --side white --elo 1600 --analysis-only
python backend/app.py
```

The first command saves local analysis without a coaching LLM call. The second
starts the Maia play/analysis application, with analysis as the landing page.
Use these commands to inspect profiling options:

```powershell
python -m analysis.profiler.report --help
python -m analysis.profiler.comparison --help
python -m tests.engine.benchmark_engines --help
```

`profiler/runtime.py` provides `RuntimeProfiler`, which records browser analysis when the backend is launched with
`--profiler-dir <directory>`; it is not a standalone command. Frontend timing
and performance details are in [web/README.md](../web/README.md#performance).

Current full-game profiles render SVG charts without a plotting dependency.
Re-exporting historical serial profiles also produces PNG and requires the
root `profiling` extra: `uv sync --extra profiling` (add `--extra cuda` to
retain GPU support). Reports describe the recorded run's configuration and
measurements; they do not assume the hardware or outcome of an earlier benchmark.

## Configuration and ratings

### Site and time-control scales

Maia policies and every estimator's internal mathematics use **Lichess Blitz**. The shared game pipeline infers the declared account/result scale from the current PGN `Site` and `TimeControl`. Site matching is case insensitive: a value containing `lichess.org` selects Lichess; one containing `Chess.com` selects Chess.com. A `Link` header does not override `Site`.

`elo_convert.py` supports `lb` (Lichess Blitz), `lr` (Lichess Rapid), `cb` (Chess.com Blitz), and `cr` (Chess.com Rapid). Numeric time controls use initial seconds plus 40 increments. Lichess classes are UltraBullet below 30 seconds, Bullet below 180, Blitz below 480, Rapid below 1500, and Classical thereafter. Chess.com uses Bullet below 180, Blitz below 600, and Rapid thereafter. Textual `blitz` and `rapid` also work. Recognized unsupported classes raise a clear error; missing metadata retains the native scale with an explicit `assumed_native` label. Python callers may pass `rating_scale='cr'` to `analyze_game` or `fit_game`; the CLI equivalent is `--rating-scale cr` and applies to both actual-rating inputs and results.

Let `F` convert native Lichess Blitz to the declared scale. The displayed shared curve satisfies `C_display(F(r)) = C_native(r)`. Actual ratings are normalized with `F⁻¹` before fitting. The reported decision is `F(native decision)`, and interval endpoints are converted the same way, without rounding intermediate values. Thus changing only the coordinate representation of the inputs commutes with fitting. Recomputing a posterior mean on a nonlinear display axis would violate this property, so the estimator's native decision convention is retained. Posterior density plots use `p_display(F(r)) = p_native(r) / F′(r)`.

The supplied conversion coefficients are unchanged and are not fitted to the game corpus or its commercial estimates. Their fitted native domain is 400–2800. The extended shared-curve support requires explicit extrapolation of the same analytic conversion equation; metadata identifies extrapolated account inputs and estimates. This numerical consistency does not establish the empirical accuracy of conversion between sites. Conversion-model uncertainty is not included in the existing conditional intervals.

Saved `played_elo_scale` records the selected scale and its provenance. Player summaries retain `canonical_estimate` alongside the displayed estimate. Method parameters and detailed diagnostics remain explicitly native; renderers convert coordinates and densities for presentation. Maia tools, rating-dependent hints and search scheduling normalize actual/fitted levels before selecting native Maia ratings. Scale or actual-rating changes invalidate only the fit context, preserving reusable engine evidence.

```powershell
python -m analysis.elo_convert 1500 cr --to lb --metadata
```

Each component reads root [config.yaml](../config.yaml) through its own settings
module. `ANALYSIS` owns search policy and resources; `MAIA` owns model, device,
and batch settings; `STOCKFISH` owns executable and startup settings. All configured
search durations are seconds, and depths are plies (half-moves).

| `ANALYSIS` setting | Scope |
| --- | --- |
| `STOCKFISH_WORKERS`, `STOCKFISH_THREADS_PER_WORKER` | Independent engine processes and search threads within each process |
| `STOCKFISH_HASH_MB_PER_WORKER`, `STOCKFISH_CACHE_ENTRIES` | Hash memory allocated to each Stockfish worker; completed native position-search cache |
| `STOCKFISH_SEARCH_STRATEGY` | Shared web/coach root-candidate scheduling: `bounded`, `staged`, or `exhaustive` |
| `STOCKFISH_EVALUATION` | Shared depth and time limits for full-game analysis, individual scoring, and coach investigations |
| `STOCKFISH_EXPLORATION` | Fast continuation previews after a candidate move |
| `PLAYER_RATING` | Estimator module; method arguments belong to its Python file |
| `CACHE_DIR` | Shared web/CLI inference and analysis-evidence cache |
| `PROFILER_DIR` | Optional output directory for browser profiler recordings |

Evaluation and exploration each have `MAX_DEPTH`, `DEFAULT_SEARCH_SECONDS`,
and `MAX_SEARCH_SECONDS`. These bound one search, not an entire game or agent
run. Frontend presets scale the evaluation limits; coach analysis uses them directly:

| Search | Time limit with current defaults | Depth limit |
| --- | --- | --- |
| Coach full-game analysis, individual scoring, and coach investigations | Evaluation default 10 s, maximum 30 s; bounded full-game phases share one position allowance | Evaluation ceiling 28 |
| Web Fast preset | Evaluation limits × 0.2: default 2 s, maximum 6 s | 12 |
| Web Balanced preset | Evaluation limits × 0.5: default 5 s, maximum 15 s | 15 |
| Web Deep preset | Evaluation limits × 1.0: default 10 s, maximum 30 s | 18 |
| Web continuation previews | Exploration default 2 s, maximum 10 s; independent of analysis presets | Exploration ceiling 18 |

Each timed search stops at its depth limit or time limit, whichever comes first.
`FRONTEND.STOCKFISH_TIME_SCALE_BY_DEPTH` contains the web-only multipliers:
`{12: 0.2, 15: 0.5, 18: 1.0}`. Each multiplier applies to both evaluation time
fields. The backend resolves these limits for full-game requests, interactive
root searches, profiler recordings, and cache validation. Presets above the
evaluation depth ceiling are unavailable. Changing this frontend mapping does
not change coach limits. Coach overrides use `--depth`, `--verify-ms`, and
`--max-ms`; there is no separate initial-analysis depth or global time multiplier.

The strategy setting selects candidate scheduling, not a Stockfish UCI option.
`bounded` shares the requested position allowance across screening, engine-best,
played/human candidates, and alternative searches. Individual evaluations and
continuation previews always use depth/time bounds; they do not perform this
multi-phase candidate scheduling.

The `staged` and `exhaustive` policies pursue target depths without the bounded
policy's time allocation. The shared game pipeline and web root searches still
have a watchdog at the applicable evaluation maximum, plus two seconds of
process-cleanup grace; exceeding it aborts the position. These are search budgets, excluding worker-queue waits,
Maia inference, rating fitting, and file output. See the
[engine guide](../engine/README.md) for worker/hash allocation.

```yaml
ANALYSIS:
  PLAYER_RATING:
    METHOD: shared_curve_affine
```

`METHOD` selects `analysis/player_rating/<METHOD>.py`. Active estimators use the same interface and current-game legal-move evidence:

| Method | Calculation | Output |
| --- | --- | --- |
| `shared_curve_affine` (default) | Current-game shared arithmetic curve and conditional variance; common translated account prior; minimum-MSE affine rating decision | Point estimates; no calibrated interval |
| `bayesian_shared_curve` | Current-game shared-curve Gaussian posterior, independent of actual ratings | Posterior medians and central 20% display intervals |

The default **Shared-curve affine fit** uses the current game's curve `C(r)`, each player's arithmetic accuracy `A`, and the pooled conditional variance `v` of mean accuracy. Maia probabilities retain all legal moves; positions with a single legal move are excluded from both observed and expected averages. The variance describes alternative move qualities under those policies, not the sample variance of the player's played moves.

The fourth-power prior is translated toward the mean available actual rating, restricted to native 0–3200 and normalized. With no supplied accounts its original center is retained. Under that common prior, the estimate is `E[R] + b*(A - E[C(R)])`, where `b = Cov(R,C(R))/(Var(C(R))+v)`. The final decision is clipped to native 0–3200 before conversion to the declared display scale. The model reads no other game, population curve or calibration asset.

For a linear curve, this is a reliability-weighted average of the prior mean and the accuracy-implied rating. For a nonlinear curve, it is the minimum squared-error decision among affine maps under the specified working moments, not an exact curve inverse or a posterior mean. A curve intersection is unnecessary. Both players use the same map, so higher arithmetic accuracy cannot produce a lower estimate; clipping or integer rounding can create ties. Changing either account can change both estimates through the common prior. No fixed small account-sensitivity bound or calibrated rating interval is claimed.

**Evaluation separation is mandatory.** Benchmark games must not supply training, calibration, priors or population assets, including unlabeled positions, Maia distributions, move qualities or observed statistics. Leaving out only the current target does not make reuse of the remaining benchmark games acceptable. Current-game inference is permitted; other benchmark games remain outside its inputs. The pretrained engine and independently supplied rating-scale conversion are not refitted on these games.

The [Shared-Curve Affine Estimation paper](../docs/shared_curve_affine.md) derives the current method and illustrates its calculation. It replaces the removed population-based hierarchical-affine and uncertainty-ensemble papers. The `hierarchical_affine` and `uncertainty_ensemble` implementation files have also been removed; their names are not aliases for `shared_curve_affine`. `arithmetic_coverage` remains disabled, and the test-derived population asset is absent. Historical test scripts and affected saved comparisons are preserved as withdrawn records, not evidence of test-only performance.

The retained Bayesian shared-curve fit
uses Maia's full legal-move distribution at the game's actual positions, with a common
move-quality curve for both players. It uses neither account ratings nor
commercial reference ratings. Method-specific settings live
in the selected Python file, not global YAML. The Bayesian module defines
`Args(rating_range=(0., 3200.), flat_prior_range=(800., 2400.), prior_range=(200., 3000.),
central_interval=.20, accuracy_sigma_scale=1., top_probability=1.)`
and `ARGS = Args()`. Its measured curve is preserved on 600–2600 and extended
with smooth exponential tails bounded by 0–100 accuracy. The prior is flat on
800–2400 with symmetric fourth-power shoulders: `S(t) = t**4 / (t**4 + (1-t)**4)`
for normalized shoulder position `t` clipped to 0–1. Raw weight is zero at and
beyond 200/3000, approximately 0.059 at 400/2800, 0.941 at 600/2600, and one
throughout 800–2400. Its integral is 2200; no midpoint parameter is needed.
The native numerical curve and posterior grid span 0–3200; native graphs display 200–3000, converted to the declared scale when applicable.
Posterior calculations normalize this prior;
the native prior graph shows its original 0–1 weights. On another scale the prior graph shows the transformed density divided by its peak, labeled as relative density.
The reported central
20% interval contains the posterior's 40th–60th percentiles; it is a deliberately
narrow display interval, not a calibrated prediction-error guarantee.

Bayesian version 8 uses this fourth-power prior and all legal moves, weighted by Maia's full probabilities,
for both expected accuracy and variance. Optional `top_probability` values below
1 keep the smallest probability-ranked set whose cumulative probability exceeds
the cutoff, include ties, and renormalize. Actual played moves remain in observed accuracy
even when outside the selected set. Positions with only one legal move are
excluded from both averages; a one-move selected set is not a legally forced move.
The Gaussian likelihood uses the resulting Maia-derived sigma multiplied by 1.0.
Curve slope and the prior still affect Elo spread. The expected accuracy defines
a game-specific reference trend, not an empirically calibrated rating scale. Entirely flat
curves and players without informative observations receive no estimate.

The graph's observed accuracy is an arithmetic mean of per-move Lichess
accuracy values after forced-position exclusion. It is not Lichess full-game
accuracy: the performance snapshot separately combines a volatility-weighted
mean and a harmonic mean over the game's moves. The reference curve uses the
same arithmetic statistic as the observed value; replacing only the observed
value with full-game accuracy would compare different quantities.

The [Bayesian shared-curve paper](../docs/bayesian_shared_curve.md) documents the
retained method. The interface below allows future estimators without changing
application callers.

## Saved artifacts and caches

Coach output defaults to `<PGN parent>/output/<PGN stem>-full/`; `--output-dir`
overrides it. `analysis.json` stores game and per-move evidence, flags, performance,
and player-rating results. Detailed numeric evidence stays outside the compact
context sent to the coaching agent. Timing runs also save timing data and
configuration snapshots.

Python callers can use `player_rating.figures.export_figures(fit, output_dir)`
to save `fit.json`, `analysis.svg`, and `prior.svg`. The main image places the
current game's accuracy curve beside a combined White/Black result panel. The
default shows the common prior level, accuracy adjustments and final point
decisions; the Bayesian method shows posterior densities and its conditional
intervals. A point-only fit must not be displayed as a posterior distribution.
Accuracy-curve intersections are labeled separately from the estimator's final
points, including extrapolated or absent crossings. Rating axes use the declared
scale, with the native 200–3000 plotting domain converted accordingly. All rating
figures are SVG. Completed exports replace known generated files atomically and
preserve unrelated images.

`export_saved_figures(analysis, output_dir)` renders a supported saved fit without
engines. Normal coach output places figures in the game's `player-rating/`
directory. Refreshing a retired fit requires the active estimator; a historical
corpus-based rating must not be presented as a newly calculated current result.
The numeric evidence cache remains separate from output, so changing the rating
method does not require new engine searches when saved measurements are valid.

## Shared pipeline and boundaries

Both web and coach use `game/pipeline.py` for full-game engine evidence, hints,
Lichess performance and player-rating fitting. `game/study.py` handles interactive
positions and variations; it does not own a second complete-game pipeline.
Engine owners manage process cleanup. Rendering and profiling consume completed
analysis rather than implementing alternate rating calculations.

## Player-rating estimator interface

The `analysis/player_rating/` package contains the estimator implementation and
its supporting modules. Each filename identifies one responsibility:
`interface` defines the abstract `PlayerRating` contract,
`service` connects that contract to applications and saved results, `evidence`
constructs the common numeric inputs, `policies` obtains Maia distributions, and
`parameters` defines the measured Maia rating grid. The
`shared_curve_affine` and `bayesian_shared_curve` modules implement the active estimators. Removed method names fail as unavailable; the surviving historical `arithmetic_coverage` module rejects use.

[player_rating/interface.py](player_rating/interface.py) contains no fitting,
loading, or cache logic. [player_rating/service.py](player_rating/service.py)
prepares evidence, loads an implementation, validates its result, and attaches
metadata. These are separate responsibilities, not duplicate implementations.
The estimator calculates from numeric evidence; it does not load PGNs, start
engines, render reports, or read reference labels.
It must not import test fixtures, load benchmark-derived assets or use other
benchmark games as fitting context. A future external model requires a separate,
independently sourced development corpus and an explicit data-provenance audit;
the active estimators require no such asset.

### Add a method

Create `analysis/player_rating/my_method.py` containing a module-local concrete class named
`Rating`. This minimal example delegates to the existing calculation; replace
that delegation with the new method's mathematics:

```python
from analysis.player_rating.interface import PlayerRating
from analysis.player_rating.bayesian_shared_curve import summarize


class Rating(PlayerRating):
    name = "My rating method"  # Optional; otherwise derived from the filename.
    version = 1              # Increment when the calculation changes.

    def fit(self, evidence):
        return summarize(evidence)
```

Set `ANALYSIS.PLAYER_RATING.METHOD: my_method` and restart the application. The
value is a filename without `.py`, matching `[a-z][a-z0-9_]*`; paths and arbitrary
method IDs are rejected. No registry, import list, or algorithm branch in an
application needs changing.
Adding a method must respect the current-game evidence boundary; no method may restore benchmark-derived population inputs.

`get_estimator(method=None, *, args=None)` loads an explicit filename or the configured choice
from `analysis.player_rating`.
It requires a module-local concrete subclass, a no-argument constructor, and a
synchronous `fit(evidence)` with no other required arguments.
The module filename supplies the method ID; the inherited name is its readable
form and the inherited version is `1`. Names must be nonempty and versions
positive integers. Each lookup creates a fresh instance, so keep constructors
inexpensive and instances stateless. Python caches imports; restart after editing
method code. Invalid modules, implementations, and metadata fail explicitly.
Methods with settings override `parameters` to return a finite JSON mapping of
their effective settings. The inherited default is `{}`. For example, a method
using a frozen dataclass can return `dataclasses.asdict(self.args)`. Optional
`args` supplied by a Python caller pass unchanged to the implementation's
`Rating(args=args)` constructor; application code does not interpret them.

### Evidence contract

`fit` receives a dictionary with exactly `White` and `Black`. Each side contains:

| Field | Contract |
| --- | --- |
| `schema_version` | Current numeric evidence schema: `1` |
| `conditioning` | `equal_opponent`: both Maia rating inputs use the same rating |
| `rating_grid` | `[600, 700, ..., 2600]` in ascending order |
| `observations` | Ordered analyzed decisions for that side; may be empty |
| `actual_rating` | Optional finite actual rating in `[0, 4000]`, or `None`; attached at fitting time rather than stored in the engine-evidence cache |

Each observation contains:

| Field | Contract |
| --- | --- |
| `played_index` | Integer index of the played move in the aligned legal-move arrays |
| `qualities.position` | Complete legal-move accuracy array in `[0, 100]`; played moves use the next decision's position evaluation, with the final move using its played-root score |
| `qualities.root` | Same legal-move order, using root candidate-search accuracy |
| `maia_probabilities` | 21 rating rows × legal moves; finite nonnegative values, each row summing to one |
| `weight` | Positive finite volatility weight, available to methods that use it |
| `position_win_probability` | Optional White-relative before-position winning probability in `[0, 1]`; restored from canonical/saved position scores for compatible numeric evidence; active estimators do not use it to weight move accuracy or identify a population context |

All move arrays share one order. Forced moves have one column with probability
one; empty sides remain present. Insufficient evidence must produce an explicit
unavailable estimate. The shared curve omits single-legal-move positions, uses
`qualities.position`, and takes unweighted arithmetic means. Other methods may
use the available quality views and weights according to their documented design.

[player_rating/evidence.py](player_rating/evidence.py) validates and copies evidence, stripping
arbitrary metadata. Only the explicit `actual_rating` field passes actual Elo;
reference Elo, player names, and game IDs never reach the estimator. The context
adapter reads `WhiteElo`/`BlackElo` and explicit application overrides, without
inventing a fallback rating. These optional additions remain compatible with
numeric schema 1; the service attaches missing context before fitting. Observations describe actual game positions, not generated
alternative histories. [player_rating/policies.py](player_rating/policies.py) batches
history-aware, equal-rating Maia policies and reuses those from the analysis pass.
A method needing different evidence requires an explicit schema extension;
estimators do not request engine runs or silently reinterpret common fields.

### Result contract

Return `players`, `prior`, `interval_scope`, `rating_range`, and `central_interval`.
`rating_range` is a pair of finite increasing support bounds;
`central_interval` is a finite probability strictly between zero and one, or
`None` for a point-only method. When it is `None`, every player's `interval` and
`uncertainty` must also be `None`.
`players` contains exactly
`White` and `Black`, each with:

| Field | Contract |
| --- | --- |
| `estimate` | Integer within the method's declared `rating_range`, or `None` when unavailable |
| `uncertainty` | Nonnegative finite display uncertainty, or `None` without an estimate or for a point-only method |
| `interval` | Two finite ordered bounds within `rating_range`, containing any estimate; `None` for a point-only method |
| `moves_used` | Integer count, no greater than the side's available observations |
| `identifiable` | Boolean, true exactly when an estimate is available |
| `at_rating_limit` | Boolean indicating a supported rating boundary |

`prior` describes the prior or explicit absence of one. `interval_scope` is
nonempty text explaining represented and excluded uncertainties. Optional
`diagnostics` is a mapping; optional `method` is explanatory text; optional
`account_ratings_used` is a boolean. Application callers must not need to
interpret these diagnostics. The complete result must be finite JSON data.

A player with zero used observations cannot receive an estimate. Invalid keys,
nonfinite values, malformed intervals, inconsistent availability, and unsupported
ratings fail validation rather than being clamped or replaced by another method.
The loader then adds method ID/name and version metadata. Method output declares
its central interval and support explicitly; the loader does not substitute a
global interval or assume that Maia's measured grid bounds every estimator.

### Pipeline and cache identity

Application-facing functions in `analysis.player_rating.service` are:

```python
fit_game(game, records, evaluate, cache_directory, model_signature,
         *, evaluate_many=None, args=None, output_dir=None, ratings=None)
fit_evidence(evidence, *, evidence_key=None, args=None)
refresh_saved_rating(analysis, cache_directory, *, output_dir=None)
```

`fit_game` collects or reuses evidence; `fit_evidence` recalculates from compatible
saved evidence without engine calls. Both return the same normalized two-player
result. `ratings` optionally maps `White`/`Black` to explicit actual-rating
overrides; absent entries use PGN actual-rating headers. `fit_evidence` callers
provide the optional context fields themselves. Pass a rating-artifact directory as `output_dir` to `fit_game` or
`refresh_saved_rating` to regenerate figures, including cache hits and unchanged
saved-fit signatures. The application pipeline's `rating_output_dir` performs
the same export after its final cancellation check. Coach and browser job
entrypoints supply normal output locations; pure `fit_evidence` remains free of
filesystem output so other estimators and numerical tests can use it.
Canonical game records contain the played UCI `move`, White-perspective
`position_score`, and `scores` for every legal UCI move, using centipawns or signed
mate notation. Optional `policies[rating][uci]` reuses complete Maia distributions
already prepared by game analysis.

The numeric-evidence key covers schema, Maia model signature, starting FEN, moves,
scores, and policies. Files live under the caller's cache root at
`player-rating/rating-<key>.json`. Estimator identity/version and method settings
are excluded so conforming methods can reuse expensive measurements.

`rating_signature` separately identifies estimator ID/name/version, evidence
schema, and the method's `parameters` mapping. `rating_fit` records that signature
and the evidence key. A separate `context_signature` identifies the attached
actual ratings. Changing the method, its settings, or
supplied actual ratings refreshes a saved fit without repeating engine measurements.
Increment the estimator version when its calculation changes; increment the
evidence schema when its meaning or required structure changes. Old estimates
must not be presented as newly calculated under a changed method.

## Tests and extension checks

PGN, search, exploration, hints, profiling, rating tests, and evaluation-only fixtures
live in `tests/analysis/`. Engine and HTTP checks live in `tests/engine/` and
`tests/backend/`. The [test guide](../tests/README.md) documents suite coverage,
coaching checks, profiling and project-wide commands. Experimental scripts and fixtures also belong under
`tests/`, not in the runtime package.

`python -m tests.analysis.compare_shared_curve_games` refits saved game evidence with the configured method,
refreshes each game's `player-rating/` figures, and compares fitted ratings with
PGN reference headers after inference. Aggregate JSON/CSV results and comparison
charts go to `tests/analysis/output/selected-rating-production/`. Reference labels
never enter the estimator, and this comparison does not run chess engines or a
coaching model.

For a new estimator, test both players, empty and one-sided games, invalid
inputs/results, rating bounds, deterministic evidence reuse, and declared
invariants such as color symmetry. A loader integration test should pass another
conforming estimator through the normal application and saved-refresh paths.
Compare commercial labels only after fitting; never use test references as
hidden inference inputs. Start with small offline checks.
