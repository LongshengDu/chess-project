# Shared chess analysis

`analysis/` turns game histories and engine measurements into reusable move evidence, accuracy metrics and figures. This is the development reference; see the root guide for [setup](../README.md#install-and-run), [usage](../README.md#usage) and [project boundaries](../README.md#design-and-data).

## Design

Web whole-game analysis and the coach call `analyze_game`, implemented by `GameAnalyzer`. The pipeline prepares history-aware Maia policies, schedules Stockfish searches, builds move records, and derives hints, performance and accuracy curves. Interactive navigation in `game/study.py` uses the same adapters and measurements without becoming a second full-game pipeline.

`AnalysisSession` owns the `PositionCache`, request identities, search limits and job cancellation. Its [EngineRuntime](../engine/README.md) owns or borrows native resources. Closing an owned session closes its engines; closing a borrowed web session cancels its own work while leaving server engines running. `CachedAnalysisSession` validates existing game evidence before reconstruction and never starts engines.

Raw measurements, prepared analysis and rendered figures have separate responsibilities. `cache/` owns evidence storage and reuse; `accuracy/` owns accuracy measurements and figures. Cache reuse requires compatible engine/history inputs, complete evidence and sufficient requested search limits. Prepared analysis and figures remain readable without native engines. HTTP jobs and durable web storage belong to [backend](../backend/README.md), and investigations and reports belong to [coach](../coach/README.md).

## Modules

Each runtime Python module is listed below, excluding package initializers.

| Module | Responsibility |
| --- | --- |
| [game/pipeline.py](game/pipeline.py) | `GameAnalyzer` and `analyze_game`: shared full-game preparation, searches and derived evidence |
| [game/metadata.py](game/metadata.py) | Effective player, rating and game context without changing PGN headers |
| [game/opening.py](game/opening.py) | ECO opening names from reached positions in the pinned Lichess opening database |
| [game/study.py](game/study.py) | PGN/FEN import, mainline/variation navigation, interactive scoring and file loading |
| [game/history.py](game/history.py) | Complete history replay and compact branch references |
| [game/context.py](game/context.py) | Position facts and earlier decisions leading to a critical moment |
| [game/summary.py](game/summary.py) | Compact evidence views and balanced investigation selection |
| [game/cancellation.py](game/cancellation.py) | Cancel one job's searches without closing shared engines |
| [accuracy/lichess.py](accuracy/lichess.py) | Lichess accuracy, judgment thresholds and game phases |
| [accuracy/performance.py](accuracy/performance.py) | Lichess game/stage accuracy and saved move judgments |
| [accuracy/evidence.py](accuracy/evidence.py) | Validate full legal policies and reconstruct aligned move accuracies |
| [accuracy/policies.py](accuracy/policies.py) | Attach complete native-anchor policies from shared batched inference |
| [accuracy/service.py](accuracy/service.py) | Prepare or aggregate per-move measurements into game curves |
| [accuracy/figures.py](accuracy/figures.py) | Render saved side/shared accuracy and deviation curves as SVG |
| [accuracy/plot_style.py](accuracy/plot_style.py) | Shared selectable SVG text and side colors |
| [accuracy/by_move.py](accuracy/by_move.py) | Persist per-rating move measurements and render selected-anchor move graphs |
| [accuracy/comparison.py](accuracy/comparison.py) | Same-Elo side comparisons and chronological/hardest/easiest position queries |
| [position_display.py](position_display.py) | Material-display data and move sounds |
| [position_evaluation.py](position_evaluation.py) | Score units, losses, material stages and supplied-rating helpers |
| [position_results.py](position_results.py) | Convert Stockfish scan lines to and from cached/web result records |
| [stockfish_search.py](stockfish_search.py) | Root-move scheduling, budgets, deadlines and streamed evaluations |
| [stockfish_exploration.py](stockfish_exploration.py) | Bounded continuation searches and legal-line validation |
| [profiler/runtime.py](profiler/runtime.py) | Timing events and profiling-run lifetime |
| [profiler/report.py](profiler/report.py) | Validate recorded timings and produce CSV, Markdown, HTML and SVG reports |
| [profiler/comparison.py](profiler/comparison.py) | Compare recorded timing, score and depth tradeoffs |
| [session.py](session.py) | `AnalysisSession` and `Limits`: cached requests, batching and search orchestration |
| [cache/artifacts.py](cache/artifacts.py) | `AnalysisStore`: publish prepared analysis and resolve pinned raw measurements |
| [cache/session.py](cache/session.py) | Rebuild a saved game from pinned or discovered measurements without engine work |
| [cache/storage.py](cache/storage.py) | Content identities, atomic JSON publication and internal manifest storage |
| [cache/positions.py](cache/positions.py) | Immutable measurements, history contexts and locked merges in one file per FEN |
| [cache/requests.py](cache/requests.py) | Engine-specific Maia rating-pair and Stockfish search identities |
| [cache/policy.py](cache/policy.py) | Requested-limit dominance, compatible engine histories and evidence selection |
| [cache/structure.py](cache/structure.py) | Readable grouped position documents, immutable identities and shared result bodies |
| [cache/validation.py](cache/validation.py) | Completeness, probability and legality checks before caching or reuse |
| [move_hints.py](move_hints.py) | Per-move attention labels from measured evidence |
| [maia_context.py](maia_context.py) | Resolve supplied account context and normalize ratings for Maia conditioning |
| [elo_convert.py](elo_convert.py) | Four-scale account conversion, site/time-class inference and provenance |
| [settings.py](settings.py) | Read analysis configuration directly from root YAML |

## Interfaces

| Interface | Contract |
| --- | --- |
| `analyze_game(game, session, actual_elo=None, *, progress=print, on_position=None, cancel=None, accuracy_output_dir=None, rating_scale=None)` | Analyze both sides; optionally publish accuracy SVGs. Callers otherwise control artifact publication |
| `AnalysisSession.borrowed(...)` | Use the backend's Maia adapter and Stockfish pool without taking ownership |
| `CachedAnalysisSession(game, cache_directory, *, engine_signature=None)` | Validate pinned game evidence, or discover complete compatible observations when no manifest exists; missing required evidence is an error |
| `AnalysisStore.save(path, analysis)` / `load(path)` | Save/load prepared analysis; keep raw positions and internal bookkeeping outside public JSON |
| `AnalysisStore.load_positions(path)` | Explicitly resolve pinned raw evidence; never substitute a newer measurement |
| `refresh_saved_curve(analysis, *, output_dir=None)` | Recalculate from supplied raw positions, or aggregate existing per-move measurements; optionally render figures |
| `export_saved_figures(analysis, output_dir)` | Replace the overview and three default by-move SVGs from prepared data |
| `expected_accuracy_by_move(analysis, maia_elo=1600)` / `export_move_curve(analysis, output_dir, maia_elo=1600)` | Read selected-anchor move data or write its SVG |
| `AccuracyComparison(analysis)` | Query prepared evidence with native-rating, stage and inclusive game-relative ply filters |

### Prepared and raw evidence

For `N` played half-moves, raw `positions` contains `N+1` boards, including the final board. `moves[i]` describes the decision at `positions[i]` with game-relative `ply=i+1`. The in-memory pipeline includes raw positions for browser transport; public `analysis.json` contains prepared evidence only.

| Prepared field | Meaning |
| --- | --- |
| `headers` | Original PGN headers, unchanged by analysis overrides |
| `game.white`, `game.black` | Effective `{name, elo}` for each player; unavailable values are `null` |
| `game.rating_scale` | Effective account scale: `lichess_blitz`, `lichess_rapid`, `chess_com_blitz` or `chess_com_rapid` |
| `game.result`, `game.date`, `game.eco`, `game.opening` | Supported game metadata, nullable; opening is resolved from the ECO and reached database positions |
| `coaching` | Game-wide coaching information, currently `{}`; selected coaching side is a runtime request |
| `moves` | Played/candidate evaluations, ranked Maia choices, per-anchor expected accuracy and absolute deviation, hints and judgments |
| `performance` | Lichess accuracy and game/stage statistics |
| `accuracy_curve` | Native anchors, shared/side expectation and deviation curves, observed arithmetic/Lichess accuracy and eligible/forced counts |

Consumers read effective context from `game`. Account conversion selects Maia context without changing the curve's native scale. Missing names may display as White/Black; missing results stay unknown. The source `Opening` header is preserved but does not supply the resolved opening name.

| Raw field | Units and orientation |
| --- | --- |
| `maia.maia_kdd_<rating>.policy` | Complete legal UCI move probabilities for the side to move; equal own/opponent rating at each anchor |
| `maia.maia_kdd_<rating>.value` | White expected game score, `P(win) + 0.5 * P(draw)`, in 0–1 |
| `stockfish.cp_vec` | Legal-root centipawn scores, positive for White; mate display sentinels are ±10000 |
| `stockfish.mate_vec` | Mate distances signed from the side to move, unlike the White-oriented centipawn scores |
| `root_move_depth_vec`, `depth`, `best_depth`, `candidate_min_depth` | Achieved search depths in plies |
| `target_depth`, `target_reached` | Requested ceiling and whether it was reached |
| `complete`, `coverage_complete` | Scheduling finished and all legal moves scored; neither promises equal depth |
| `phases`, budget/time/stop fields | Search diagnostics and termination information |

### Coaching artifacts and analysis cache

`PositionCache` stores one `ANALYSIS.CACHE_DIR/positions/<SHA-256 of board.fen()>.json` per canonical six-field FEN. The `position-evidence-v2` document groups readable observations under `evidence` by namespace, rating pair or search type, and history. Each observation keeps its request, immutable measurement identity, active status and result; repeated bodies alone use `shared_results`. `histories` retains starting FEN and complete UCI moves, and `engines` holds shared signatures. A FEN alone is not a result identity.

Maia requests include content-based model identity and own/opponent ratings, independent of batch shape. Stockfish requests include executable identity/resources, search kind, strategy, requested depth/time and candidate/root options. Reuse requires compatible engine semantics and both requested limits to meet or exceed the new request; achieved depth remains diagnostic. Candidate coverage and exploration restrictions remain part of compatibility. Maia histories must preserve the model's input window; Stockfish histories must preserve relevant reversible play and terminal state. Unknown engine histories use exact pins, never guessed identities.

`games/` stores the latest ordered pinned manifest keyed by starting FEN and played moves. `game-metadata/` associates artifact bookkeeping and exact references with the resolved artifact path plus public-content identity, so identical prepared files can retain different observations. `load_positions(path)` uses only that artifact's association; an unassociated copy or external edit never silently adopts a newer game manifest. Prepared data remains readable without raw evidence. `AnalysisStore.save` strips raw positions/references and internal analysis, performance and curve bookkeeping from public JSON.

Normal lookup selects complete compatible observations and keeps their original request and measurement identity. Stronger requested limits can supersede weaker active observations while immutable older bodies remain available to saved pins. Explicit refresh bypasses older reusable observations without deleting them; new measurements remain reusable within that run. Normal analysis may compute missing requests; see [usage](../README.md#usage).

When no observation satisfies both requested limits, the next compatible Stockfish search uses `max(cached depth, requested depth)` and `max(cached time, requested time)`, preserving the union of required candidates. New input limits are validated before promotion; a smaller preset does not erase a previously larger limit. Search kinds, strategies, engine profiles and exploration restrictions remain separate.

Cache-only reconstruction prefers the existing game manifest, including its original observations when settings or engines change; a broken manifest fails instead of falling back. With no manifest, it discovers all required native Maia rating pairs and complete legal-move Stockfish evaluations from compatible position records. Each namespace requires one coherent profile across the game, retaining engine, search kind, strategy, policy version and other semantic options while allowing different requested limits/candidates. The configured engine is preferred only when it identifies one complete profile; multiple complete strategies remain ambiguous. Missing local asset files permit a unique recorded-profile fallback without starting engines or assigning a guessed identity. PV-only exploration records cannot substitute for full evaluations.

Validation rejects unavailable placeholders, invalid policies, incomplete legal-move evaluations and illegal continuations. Every played position requires Stockfish evidence and every non-forced played position requires all 21 native Maia policies; forced policies follow chess rules without fabricated measurements. Final unplayed-board evidence may be absent. Cache-only sessions preflight the complete game before returning, and the [batch runner](../tests/README.md#batch-analysis-and-saved-reconstruction) validates every session before changing outputs; `--check-cache` performs only that engine-free check.

Thread/process locks protect read–merge–write updates followed by atomic replacement; engine work runs outside those locks. Corrupt records are misses, and recovery preserves the unreadable file under `.position-corrupt`. Cache storage remains separate from engine assets, durable web studies and output directories; changing a report destination does not relocate it.

### Accuracy curve evidence

Measurements use the game's full legal policies at 21 native **Lichess Blitz 600–2600** anchors, with equal player/opponent conditioning. For legal-move accuracy `q(m)` and normalized probability `p_r(m)`, expected accuracy is `mu(r) = sum(p_r(m) * q(m))` and absolute deviation is `d(r) = sum(p_r(m) * abs(q(m) - mu(r)))`. Average each quantity over a side's non-forced decisions, then weight nonempty sides equally. Observed arithmetic accuracy averages that side's actual choices over the same decisions. Empty/forced-only sides have `null` metrics, not zeros.

Absolute deviation measures within-position spread in accuracy points; it is not uncertainty or playing strength. Lichess game accuracy separately combines volatility-weighted and harmonic accuracy and retains forced moves under its own rules. The played move uses the next unrestricted position score, or the final played-move score; alternatives use root scores, and legal terminal outcomes override search scores. Unequal search depths and time limits constrain the evidence. The [technical paper](../docs/accuracy_curves.md) owns the full derivation.

Per-move measurements are saved at `moves[i].maia[str(rating)]`, including forced positions. Curves, by-move views and `AccuracyComparison` exclude forced decisions when aggregating/comparing, require existing measurements, and perform no engine work. Compare sides at the same anchor and game scope; expected accuracy is descriptive positional context, not a winning chance or a player-rating estimate.

### Saved refresh and figures

`refresh_saved_curve` with raw positions prepares per-move measurements; with prepared data alone it aggregates them. Missing prepared values require explicit raw evidence. `export_saved_figures` writes `accuracy-curve.svg` and `accuracy-by-move-{1600,1800,2000}.svg` directly into the chosen output directory, normally beside `analysis.json`, without separate graph-data files.

The overview has White, Black, shared and absolute-deviation panels. Bands show mean ± absolute deviation, clipped to 0–100; they imply no probability coverage. `xelo` legend values are geometric intersections interpolated between measured anchors, without extrapolation or a player-rating estimate. By-move graphs compare selected-anchor expectations and actual move accuracy using original PGN fullmove numbers; dotted connections span omitted forced turns. SVG text remains selectable, with consistent White/Black colors.

## Configuration

[Root configuration](../README.md#configuration) defines units, paths and override rules. `settings.py` reads [config.yaml](../config.yaml) directly; analysis consumes engine settings without a second configuration gateway.

| Setting | Analysis responsibility |
| --- | --- |
| `ANALYSIS.CACHE_DIR` | Shared position evidence, game manifests and artifact bookkeeping |
| `ANALYSIS.STOCKFISH_WORKERS`, `STOCKFISH_THREADS_PER_WORKER`, `STOCKFISH_HASH_MB_PER_WORKER` | Search concurrency and per-worker resources; [engine](../engine/README.md#configuration) owns allocation details |
| `ANALYSIS.STOCKFISH_SEARCH_STRATEGY` | `bounded`, `staged` or `exhaustive` root scheduling |
| `ANALYSIS.STOCKFISH_EVALUATION` | Depth/default/maximum time for full-game analysis and scoring |
| `ANALYSIS.STOCKFISH_EXPLORATION` | Separate continuation-search limits |
| `ANALYSIS.STOCKFISH_CACHE_ENTRIES`, `PROFILER_DIR` | Completed interactive search capacity and optional browser timing output |
| `MAIA`, `STOCKFISH` | Model/batching and executable/startup settings from the engine component |

Bounded search shares one position allowance across screening and deepening. Other strategies retain a maximum-time watchdog; inference, queue waits, accuracy calculations and output are outside search budgets. Frontend presets scale evaluation default/maximum seconds through `FRONTEND.STOCKFISH_TIME_SCALE_BY_DEPTH`; coach uses unscaled evaluation limits. YAML durations use seconds and depths use plies, while some tool/session interfaces use milliseconds.

## Development

Keep calculations and request reuse here, native lifetimes in engine, and transport/publication in consumers. Preserve full legal-move alignment, rating scale provenance and history when extending evidence. Use `AnalysisStore` for saved analysis and keep renderers dependent on prepared data. Evaluation games may test behavior but cannot supply reusable training or calibration assets.

Figure and timing tools operate on saved data; their argument reference is available without analysis:

```powershell
python -m analysis.accuracy.by_move --help
python -m analysis.profiler.report --help
python -m analysis.profiler.comparison --help
```

The single-figure CLI accepts native anchors in steps of 100, defaults to 1600 and replaces only `accuracy-by-move-<elo>.svg`. `RuntimeProfiler` is an integration service; report tools read recorded timings and do not start a profiling run.

## Verification

Use [the shared test guide](../tests/README.md) for commands and explicit live-run scope. Analysis tests cover complete legal policies, rare-outlier deviation, forced/empty games, color symmetry, arithmetic/Lichess distinctions, history-sensitive cache reuse, pinned artifacts and figure replacement. Prefer synthetic evidence and fake engines for calculation or integration changes; documentation and figure work do not require new searches or a cache refresh.
