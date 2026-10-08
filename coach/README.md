# Chess coach

Codex investigates local chess evidence and writes illustrated coaching reports using the existing ChatGPT sign-in. The coach shares the web application's [analysis pipeline](../analysis/README.md) and [engine adapters](../engine/README.md). See the root guide for [installation](../README.md#install-and-run), [workflows and rating input](../README.md#usage), and [output/cache locations](../README.md#design-and-data).

## Design

The CLI prepares or loads a game's analysis, then passes a runtime target side to the common coaching runner. Saved analysis remains reusable for either player. `AnalysisSession` owns reusable evidence requests and an `EngineRuntime`; the CLI closes the engines it creates. Cache compatibility, immutable observations, saved-game reconstruction and accuracy calculations belong to `analysis`.

Before generation, the coach prepares played-versus-alternative comparisons, immediate human replies, strongest-defense checks, diagrams and preceding-move context. Selection spreads meaningful decisions across the stages reached; prepared investigations count toward coverage and execution budgets. Codex receives compact evidence and requests additional chess facts through typed tools. Complete investigation results remain local, while repeated baselines and reply distributions use references in model responses.

The Codex adapter starts the SDK's pinned runtime in a temporary directory over local stdio, with unrelated tools and integrations disabled. It accepts ChatGPT authentication and exposes only the chess interface. Python validates and publishes the report. The provider process, progress worker and owned engines close on completion, failure or interruption; a failed provider turn is not automatically restarted.

## Modules

| Module | Responsibility |
| --- | --- |
| [coach.py](coach.py) | CLI validation, output paths, saved-analysis compatibility and engine lifetime. |
| [agent_runner.py](agent_runner.py) | Shared `run_coach` workflow and caller-supplied `CoachingRequest`. |
| [agent_codex.py](agent_codex.py) | Codex process, authentication, dynamic tools and streamed usage events. |
| [agent_budget.py](agent_budget.py) | Response, cumulative-token and elapsed-time accounting and limits. |
| [agent_progress.py](agent_progress.py) | Progress callbacks, public commentary and the background status worker. |
| [tools_chess.py](tools_chess.py) | Stateful investigations, bounded calls, result reuse and local trace. |
| [tools_schema.py](tools_schema.py) | Typed tool declarations, JSON schemas, argument validation and session bindings. |
| [tools_evidence.py](tools_evidence.py) | Initial investigation preparation and compact evidence for Codex. |
| [report_diagrams.py](report_diagrams.py) | SVG board rendering and report image provenance. |
| [report_output.py](report_output.py) | Evidence coverage, report validation and atomic publication. |
| [report_performance.py](report_performance.py) | Deterministic Markdown performance table from shared statistics. |
| [settings.py](settings.py) | Component-local reading and path resolution of root YAML. |

[prompt.txt](prompt.txt) contains the production coaching instructions. Tests, test prompts and small-report scenarios belong to [tests/coach](../tests/README.md).

## Interfaces

### Evidence

| Contract | Meaning |
| --- | --- |
| Saved `analysis.json` | Prepared game context, move evidence, performance and native accuracy curves from `AnalysisStore`; canonical raw measurements remain in the shared analysis cache. The [analysis guide](../analysis/README.md) owns the schema and cache contract. |
| `game` | Authoritative names, effective account ratings, rating scale, result and opening context. Original PGN `headers` remain saved but are omitted from the agent overview; unknown values remain `null`. |
| Request context | The overview adds `coaching.side`, per-player converted `maia_elo` and `game.maia_rating_scale`. All chess-tool ratings use native Lichess Blitz; saved analysis contains no coaching target. |
| Move evidence | Evaluations use White-perspective pawns; `#-2` means Black mates and `#0`/`#-0` means checkmate. Mate comparisons have null loss; negative numeric loss preserves a search discrepancy. Saved flags are attention hints. |
| `human_replies` | Immediate replies after the candidate, their probabilities, checked evaluations and legal mate-in-one continuations. Vectors follow `ratings`; `covered_probability` is retained probability mass and `engine_reply_p` is the separate engine defense's likelihood. Missing values mean unknown. |
| `leadup_context` | Earlier moves, saved evaluations/flags and factual board changes before a decision. Features such as pins, pawn structure and king-zone attacks require interpretation and verification. |

Human reply curves use equal-rating conditioning; explicit `maia_analyze` queries specify both players' ratings. Follow a likely reply by appending its UCI move to `human_replies.position.line`. `after_defense` identifies the separate engine branch. `baseline_ref` and `reply_ref` refer to evidence already supplied, rather than missing results.

Accuracy tools read prepared evidence without engines or raw-cache access. They compare both sides at common native anchors and exclude forced decisions. Higher expected accuracy describes easier preservation of move quality for Maia; absolute deviation describes move-quality spread. Neither yields a player-rating estimate. Match White/Black observations by PGN `move_number` and `side`; `ply` remains a separate game-relative half-move index, including custom starting positions. Whole-game Lichess accuracy stays separate from filtered arithmetic comparisons. See the [accuracy paper](../docs/accuracy_curves.md) for definitions.

### Chess tools

The `@chess_tool` declarations in [tools_schema.py](tools_schema.py) are the authoritative signatures and validation rules. They bind once to `ChessTools`; the Codex adapter and batch validator share them. Positions use one-based `ply` before the move and an optional legal UCI `line`; `ply=N+1` selects the final board. History is reconstructed locally. Arguments use UCI, while returned continuations provide checked numbered SAN and achieved search depths.

| Tool | Purpose and key bounds |
| --- | --- |
| `get_game_analysis` | Retrieve the compact overview already supplied initially. |
| `compare_position_difficulty` | Compare both sides' expected and actual accuracy at 1–6 native 600–2600 anchors, with optional stage and inclusive ply filters. |
| `get_accuracy_by_move` | Retrieve saved per-move expectation, actual accuracy and deviation; filter by side/stage/plies, sort chronologically or by expectation, and return up to 40 decisions. |
| `get_position` | Inspect an actual or hypothetical board and render an SVG without engine searches. |
| `get_leadup` | Retrieve 1–16 preceding half-moves and factual changes, excluding the target move; continue from `from_ply` to walk farther back. |
| `maia_analyze` | Predict 1–5 choices with explicit mover/opponent ratings on the 600–3000 Lichess Blitz scale. |
| `maia_compare` | Compare up to six mover ratings and 1–5 choices per level. |
| `stockfish_analyze` | Check 1–5 lines, optionally restricted to legal candidates, with 1–16 continuation plies and a bounded millisecond allowance. |
| `explore_candidate` | Investigate a move's immediate human replies and strongest resistance, with 2–16 continuation plies. Default Maia levels use the converted player rating and higher native levels. |
| `compare_played_vs_candidate` | Compare actual and alternative choices with human replies, engine checks and decision/defense diagrams. |
| `investigate_batch` | Execute 1–6 independent `{tool, arguments}` requests in input order; no nested batches. Each child counts toward the tool budget. |

Batches validate schemas and declared bounds before execution. An illegal branch produces an item error while other independent items can complete. Reuse supplied evidence before asking for more; dependent investigations wait for their preceding result.

### Runner and artifacts

`run_coach(analysis, session, output_dir, *, side, request=CoachingRequest(...))` is the shared execution interface. A request can supply instructions, task preparation, a report validator, report name, tool access and execution limits. The normal request produces a full report; test scenarios supply their own request through the same interface. `progress=callback` receives progress messages and `progress=None` disables them.

| Artifact | Contents and update policy |
| --- | --- |
| `analysis.json`, `game.pgn`, accuracy SVGs | Shared analysis artifacts; filenames and regeneration rules belong to the [analysis guide](../analysis/README.md). |
| `initial_evidence.json` | Prepared compact package supplied before generation. |
| `investigations.json` | Detailed results from the latest investigation session. |
| `coaching.md` / `coaching.draft.md` | Accepted report / latest proposed draft. Validation precedes atomic replacement of the accepted report. |
| `positions/*.svg` | Verified boards oriented to the coached side; orange played-move and green alternative arrows. |
| `agent_run.json` | Latest provider, model, status, investigated plies, usage and failure detail. |
| `agent_trace.jsonl` | Current run's tool arguments, timing, result identifiers and status. |
| `response_usage.jsonl` | Current run's per-response usage deltas, elapsed time and context/tool-response sizes. |
| `usage.jsonl` | Append-only run accounting. |

Logs retain public diagnostics and chess evidence, without raw provider responses or reasoning. Usage uses cumulative-event deltas, ignores duplicate/stale events and counts cached input as a subset of input. Missing or interrupted usage is explicitly estimated.

## Configuration

[Root config.yaml](../config.yaml) supplies current defaults; [root configuration](../README.md#configuration) covers precedence and paths. `COACH.CODEX` selects model, runtime binary and reasoning effort. Analysis search limits come from `ANALYSIS.STOCKFISH_EVALUATION`; coach analysis uses these unscaled limits. Engine settings remain in `MAIA` and `STOCKFISH`. YAML durations use seconds; CLI `--verify-ms`/`--max-ms` and tool time arguments use milliseconds, and search depth uses plies.

| Setting | Meaning |
| --- | --- |
| `MAX_MODEL_RESPONSES` | Agent response allowance; the default chess-call allowance is four times this value, including prepared investigations. One investigation may contain several engine searches. |
| `TOTAL_TOKEN_BUDGET` | Cumulative input, output and fallback estimates for the coaching run. |
| `OUTPUT_TOKEN_ESTIMATE` | Fallback accounting when usage is unavailable; it does not cap response length. |
| `RUN_TIMEOUT_SECONDS` | Coaching-run limit after initial full-game analysis, including local evidence preparation. |
| `REPORT_ATTEMPTS` | Total allowed drafts, including the first draft and corrections. |
| `PROGRESS_INTERVAL_SECONDS` | Status-message interval during quiet waits; status messages add no model requests. |

Limits are independent. Usage arrives after work has occurred, so an in-flight response can exceed a threshold before interruption. Model-response limits do not limit full-game positions or engine search depth. Test limits are isolated in [tests/coach/config.yaml](../tests/coach/config.yaml).

## Development

Keep coaching orchestration in `agent_*`, investigations in `tools_*`, and rendering/validation in `report_*`. Chess calculations, accuracy, hints, cache persistence and engine operation remain in their owning components. Production modules have no test-mode flags or imports from tests.

The [prompt](prompt.txt) grounds both praise and criticism in likely human replies, practical counterplay and executable continuations, with Stockfish checking tactical soundness and stronger resistance. Investigate earlier positional choices and the opponent's setup. Attribute Maia support to specific checked choices and ratings; an engine PV does not establish human findability. Saved flags and accuracy metrics guide investigation rather than determine a coaching verdict.

Reports cover the stages reached, Performance snapshot, Main pattern, Prioritized improvement plan and Exercises; equivalent headings are accepted. Best/Worst decisions are conditional on evidence. The renderer inserts the shared two-player performance table. Pair checked numbered SAN with verified SVGs and explanations of buildup, replies and resulting positions; captions distinguish actual play from hypothetical lines. There is no production word quota or diagram cap.

The default validator requires three investigated selected-player decisions, or all if fewer, coverage of reached stages, valid diagram paths and the mandatory section groups. Validation establishes evidence coverage and structure; it cannot prove the truth of chess explanations. Rejected drafts remain separate, and repairs stay within configured budgets.

## Verification

Use the [shared test guide](../tests/README.md) for offline coach regressions, bounded live report checks and performance measurements. Validate with fake engines and scripted responses first. A small live report is appropriate only when needed; a full live coaching report requires an explicit request. Preserve saved games, accepted reports and user diagrams during development.
