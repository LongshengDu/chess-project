# Local Chess Coach

The same local Stockfish/Maia full-game pipeline used by the web application
analyzes the game, computes accuracy and hints, and fits both player ratings.
Codex investigates selected
positions and writes coaching using your existing ChatGPT sign-in. Codex is the
only coaching agent.

## Setup and running

Coaching uses the project's root Python environment and dependency files.
Complete the [project installation](../README.md#install-and-run), then sign in
from the repository root:

```powershell
codex login
```

The root `uv sync` installs the coaching SDK together with the rest of the
application. For GPU inference, use `uv sync --extra cuda` as described in the
project guide. There is no separate coach dependency installation.

The official `openai-codex==0.155.1` SDK includes a pinned Codex runtime. If
`codex login status` already reports ChatGPT sign-in, no new login is needed.
Codex uses your account allowance; the coach rejects API-key authentication for
this provider. No application environment-variable overrides are used.

All coach tests, fixtures, test prompts, limits and output live in `tests/coach/`;
instructions are in the [test guide](../tests/README.md).
Use the small-report check during development; run a full live coaching report only when explicitly requested.

For a new game, generate local analysis first, without any LLM calls:

```powershell
.\.venv\Scripts\python.exe coach/coach.py game.pgn --side white --elo 1400 --analysis-only
```

Use `coach/coach.py game.pgn --side white --elo 1400 --coach-only` when ready
for a full report. Omit both mode flags to analyze and coach in one run. Saved analysis must match the game's starting
FEN, moves and current analysis schema. Current PGN headers, selected side and supplied actual Elo replace the saved account context before refitting. Older baselines
must be regenerated with `--analysis-only` for full coaching; the isolated smoke
script can still test with an older saved snapshot. `--output-dir` selects another folder.
From `coach/`, use `python coach.py ../game.pgn ...`; module execution also works.

`--elo` and the PGN actual ratings use the scale inferred from `Site` and `TimeControl`; played-level estimates use that same scale. `--rating-scale lb|lr|cb|cr` explicitly overrides it for both players (Lichess Blitz/Rapid or Chess.com Blitz/Rapid). The shared curve and intervals are displayed in those coordinates. Chess tools normalize actual/fitted levels back to Maia's native Lichess Blitz anchors, which are labeled separately in the report. A saved game can therefore be refitted after a scale/header correction without repeating its engine analysis.

## Configuration and architecture

Edit the project [config.yaml](../config.yaml). Explicit CLI values override it.
Relative paths resolve from the YAML file's directory. Restart after edits.

```yaml
COACH:
  CODEX:
    MODEL: gpt-6.1-sol     # --model overrides
    BINARY: null          # SDK's pinned runtime; optional executable path
    REASONING_EFFORT: high
```

Codex uses the nine typed `@chess_tool` functions with local JSON schemas and validation through the documented
experimental dynamic-tool protocol. Python executes chess operations and saves
reports. The SDK runs a temporary local stdio process, with no listening web
port, a temporary working directory and disabled unrelated tool integrations.
The Codex process and engines close after completion, failure or interruption.

Modules use a functionality-first name: `agent_*` runs the coaching session,
`tools_*` supplies checked chess evidence, and `report_*` renders or validates
the result. Game analysis, rating fitting and engine operation remain in their
own components; the coach calls their public interfaces.

| Module | Responsibility |
|---|---|
| [coach.py](coach.py) | CLI, output paths, saved-analysis compatibility and engine lifetime. |
| [agent_runner.py](agent_runner.py) | Common `run_coach` workflow and `CoachingRequest` contract. |
| [agent_codex.py](agent_codex.py) | Codex process, dynamic tools and usage events. |
| [agent_budget.py](agent_budget.py) | Response, token and elapsed-time accounting and limits. |
| [agent_progress.py](agent_progress.py) | Progress messages and the background status worker. |
| [tools_chess.py](tools_chess.py) | Stateful `ChessTools`: chess investigations, bounded calls, result reuse and trace. |
| [tools_schema.py](tools_schema.py) | Typed tool schemas, argument validation and bindings to an investigation session. |
| [tools_evidence.py](tools_evidence.py) | Initial investigation preparation and compact evidence sent to the agent. |
| [report_diagrams.py](report_diagrams.py) | `BoardDiagrams`: SVG rendering and image provenance. |
| [report_output.py](report_output.py) | Evidence coverage, report validation and atomic publication. |
| [report_performance.py](report_performance.py) | Deterministic Markdown snapshot from shared analysis statistics. |
| [settings.py](settings.py) | Component-local reading of root YAML. |
| [prompt.txt](prompt.txt) | Runtime coaching instructions. |

## Saved analysis and compact evidence

Each saved move has its pre-move FEN
and position evaluation, the played move, Maia top-five lists for 1000–2600,
and the union of played/Stockfish-best/all Maia top-five candidates with full
`maia_p` curves. Evaluations are White-perspective pawns; `#-2` means Black mates,
`#0`/`#-0` is checkmate and loss is null for mate comparisons. Negative numeric
loss preserves a search discrepancy instead of silently clamping it.

Only a compact overview is sent initially. Tools keep the same move-field names,
but default to top-three choices at four relevant ratings. `maia_compare` can
retrieve other ratings or topk=5. Branch-aware position tools accept `ply` plus an optional
short UCI `line`, reconstructing complete history locally. `explore_candidate`
returns `after_defense={ply,line}` to continue the investigation without repeating
full game histories. Branch results carry verified numbered SAN lines and actual
depths. The prompt encourages checking replies and exploring subsequent human
moves before explaining an idea.

Detailed fitting surfaces and initial search diagnostics stay in engine caches.
Investigations, provider status and token accounting are separate from analysis.json.
Tool-response sizes are recorded in the local trace. Repeated baseline data is
replaced by `baseline_ref`; exact checker-to-king square paths support tactical explanations.

### Prepared investigations

Before the first model response, the coach prepares checked comparisons for three
selected-player decisions, adding missing stages (at most five moments). Short
games use all meaningful decisions. Each comparison includes immediate Maia replies
to BOTH choices, strongest-defense checks, continuations and diagrams. Quiet stages get a
fallback position, so coverage does not depend on mistakes being present. These
local investigations count toward report validation and the tool/time budget.

Selection uses expected-score deterioration instead of raw pawn loss in already
winning positions. Continuing an existing forced mate gets only a small priority
bonus. A soft proximity discount spreads initial examples across the game while
allowing a major nearby decision to remain selected. Examples are presented in
game order and missing stages remain covered; no move names are special-cased.

The compact overview includes the leading saved Maia reply after each player's
move being coached, even outside the prepared set: `likely_reply` contains
`[rating, SAN, probability, eval]`. It uses the nearest saved rating to the
opponent's supplied/fitted level and equal-rating conditioning. This is a lead
for investigation, not a guaranteed reply; no engine calls or flag recalculation
are required. The last position may lack saved reply data.

### Human reply branches

`explore_candidate` and both comparison branches return `human_replies` immediately
after the candidate, rather than only asking Maia after the engine's defense.
This includes at most five replies, probability curves near the opponent's actual
and fitted levels and at 2000/2200/2600, evaluations, and legally verified
`allows_mate_in_one` moves. Ratings near actual/fitted levels are rounded/clamped
to the saved 1000–2600 grid. All sampled ratings' first choices are retained.
`covered_probability` exposes omitted mass; it is not a winning percentage.
`engine_reply_p` indicates how often Maia chooses the separate engine defense.
Null values mean unknown. These reply curves use equal-rating conditioning,
matching saved analysis (R versus R); other Maia queries state their conditioning.

Append a reply to `human_replies.position.line` to examine the player's continuation
with the existing tools. Check the player's Maia choices before calling a line easy;
a verified mate does not establish its human findability. Render that branch with
`get_position` when explaining it. In model evidence, probability vectors follow
the `ratings` array. Repeated distributions use `reply_ref` to a prior `reply_id`.
Full results remain local; engine-defense continuations send only their leading
Maia choice per rating. Saved replies need no new engine inference; hypothetical
positions use one batched Maia query and at most one bounded Stockfish MultiPV
search, rather than a recursive reply tree.

### Position history and lead-up

The initial package also contains `leadup_context`: the preceding eight half-moves
for each critical moment, saved evaluations/flags, and changes in concrete board
features. Overlapping moves and endpoint snapshots are stored once in that package.
Both sides are included, so the coach can trace the opponent's setup as well as the
player's earlier choices. Quiet moves are included even without flags. Snapshots
record king positions, castling rights, material, pawn squares and pawns adjacent
to kings, minor pieces on home squares, doubled and
isolated pawns, open/semi-open files, absolute pins, geometric attacks near kings,
attacked undefended pieces and bishops blocked by adjacent own pawns. These are
facts to investigate, not automatic strategic judgments.

`get_leadup(ply, lookback_plies=8)` retrieves 1–16 earlier half-moves without engine
searches or flag calculation. Its window excludes the target move; `ply=N+1` gives
the lead-up to the final board. Call it again at `from_ply` to walk farther back.
Existing comparison and branch tools work at any earlier ply, preserving full
history for Maia. The report instructions require the coach to connect earlier
decisions, the opponent's setup and the critical moment, and verify claimed earlier
improvements through likely human replies and tactical checks. The validator checks coverage, not the
truth of causal prose; no extra heading or model retry is forced by keyword checks.

## Chess-tool interface

The nine functions in [tools_schema.py](tools_schema.py) use the local `@chess_tool` decorator to build JSON schemas and validate arguments. They bind once to a `ChessTools` session in [tools_chess.py](tools_chess.py); the Codex adapter and independent-batch validator use the same definitions. Declared numeric bounds are checked before a batch performs any work. A position is `{ply, line}`: one-based ply **before** the move, plus an optional legal UCI branch; `ply=N+1` selects the final board. Full history is reconstructed locally. Tool arguments use UCI; reports use verified numbered SAN.

| Tool | Purpose and limits |
|---|---|
| `get_game_analysis()` | Read the compact overview already supplied initially. |
| `get_position(ply, line)` | Inspect and render an actual or hypothetical position without engine searches. |
| `get_leadup(ply, lookback_plies=8)` | Inspect 1–16 earlier half-moves and factual board changes; excludes the target move. |
| `maia_analyze(ply, player_elo, opponent_elo, line, topk=3)` | Predict 1–5 choices with explicit mover/opponent ratings, 600–3000. |
| `maia_compare(ply, ratings, line, topk=3)` | Compare up to six rating levels and 1–5 choices per level. |
| `stockfish_analyze(ply, movetime_ms, line, multipv=2, root_moves, pv_plies=8)` | Check 1–5 lines, optionally restricted to legal candidates; return 1–16 continuation plies. |
| `explore_candidate(ply, candidate, line, stockfish_ms, maia_elos, max_plies=8)` | Compare immediate human replies with strongest resistance; 2–16 continuation plies. Defaults to supplied Elo and +200/+400/+600. |
| `compare_played_vs_candidate(ply, candidate)` | Investigate both actual and alternative choices, with human replies, engine checks and diagrams. |
| `investigate_batch(requests)` | Execute 1–6 independent investigations; no nested batches. |

`investigate_batch(requests=[{"tool": ..., "arguments": ...}, ...])` accepts up to
six independent checks in one model response. Children execute in order against
the persistent engines; each counts toward the tool budget. Invalid schemas or
oversized batches are rejected before execution, and an illegal branch returns
an item error while other independent items complete. Native multiple tool calls
remain supported. Follow-up calls should clarify specific missing facts or ideas,
not rediscover evidence already supplied.

## Coaching report contract

The report contains a Performance snapshot, analysis of the stages actually
reached, a main pattern, a prioritized improvement plan and exercises. Equivalent
headings are accepted. Best/Worst decisions are optional when unsupported. The
snapshot includes the PGN Link/Site and both fitted played levels alongside the
selected player's actual Elo. Include ranges only when the saved method supplies
them; point-only estimates must not acquire invented confidence ranges. Tiny
samples and rating-boundary fits require appropriate qualification.

Survey the overview and give important decisions and continuations adequate
explanation across the game. Balance means avoiding disproportionate focus and
repeated discussion, not reducing useful diagrams or explanations. Show the idea,
likely replies, resulting positions and practical lesson with checked notation,
SVGs and prose together. There is no production report word quota, per-moment
paragraph limit or diagram cap. Plans/exercises can refer back to earlier analysis
without retelling it. These are editorial guides, not retry-triggering checks.

Coaching judgments start with human play: likely replies near the opponent's
actual/fitted level, higher-rating trends, practical counterplay, and how easy it
is for the player to find and execute the continuation. Apply this to praise and
criticism equally. Small engine losses in a still-winning position, longer forced
mates, or a higher-rated preference alone do not establish an error. Check a trap's
sound defenses and their likelihood too; do not assume the opponent cooperates.
Stockfish supplies tactical verification and strongest resistance, not an assumed
opponent or an automatic ranking of practical choices.

Each main lesson explains a decision or idea, including tactics, plans, strategic
choices and positional judgments over several moves. Learning tips must connect
supported higher-Elo Maia choices to their purpose after likely replies, with
sound-defense limitations distinguished, then
give a practical thinking cue or exercise. Checked variations identify the exact
Maia rating(s) supporting particular choices and distinguish those choices from
Stockfish verification. An engine PV must not be described as an entire line
played by a higher-rated human. Missing support is stated instead of invented;
existing rating curves and branch evidence are reused before requesting more.

`Best decisions` and/or `Worst decisions` appear when supported by the game;
equivalent headings such as `Strengths` are accepted. These sections assess ideas,
tactics, plans, strategy and positional choices, not only individual moves.
Include either, both or neither as suitable; avoid duplicated stage analysis or
turning one example into a recurring habit. Apply the higher-Elo lessons in the
plan and exercises. Neither section is mandatory when unsupported. The validator
does not infer their suitability from error flags or impose keyword checks on
chess explanations.
Missing mandatory sections are reported together, with one repair attempt. There is no
extra unverified fallback generation at the model-response limit.

### Diagrams and continuations

Every comparison now supplies the starting `comparison_diagram` and an
`after_defense_diagram` for both played and alternative branches. These SVGs replay
the already checked histories, highlight the last move and require no additional
engine searches. `get_position(ply,line)` can render an earlier setup, later line
endpoint or exercise board locally. Boards use the selected player's orientation;
played-move arrows are orange and alternative arrows green.

SVGs are an essential part of the report: show critical positions and important
steps in continuations alongside numbered SAN and explanations. Use multiple
boards to follow the buildup, decision, likely reply and resulting position, with
comparisons where helpful. Captions distinguish the actual game from hypothetical
branches and explain what to notice. Do not reduce diagrams or explanatory detail
to meet brevity or balance targets. There is no rigid image quota or cap.
Initial baselines use candidate probability curves without duplicating the Maia
top-move tables. The model may finish immediately when the evidence suffices.

### Saved hints and performance snapshot

Every move has `flags: []` or a short list of calculated labels. They are attention
hints, not explanations or proof of intent. The 22 configured hint labels are supported.
Inaccuracy/Mistake/Blunder follow Lichess `Advice.scala`: losses of 0.1/0.2/0.3
on its -1..1 winning-chances scale (5/10/15 percentage points), with its separate
mate-created/mate-lost rules. These labels and the performance counts use the same
sequence of post-move evaluations. Only-move checks require all legal roots; material
sacrifice hints require a sound near-best move and a legal capture of the offered
piece, even if declined. Acceptance must lose at least two material points and
leave at least one point lost after any immediate recovery (so a bishop for two
pawns can qualify). Actual captures of another sacrificed piece also count;
unrelated moves do not inherit a previously hanging piece's hint. Flags are calculated
inside `analyze_game`, while all legal-root scores are available, and saved with
each move in `analysis.json`. Summaries and coaching tools only read saved labels.
When an older saved analysis lacks current performance statistics, the coaching
entry point upgrades its quality labels, `natural_but_bad`, phases and statistics
locally; other tactical hints are preserved. No new engine or model calls are
needed for this upgrade. To regenerate other missing hints, use `--analysis-only`.

`analysis.json.performance` stores both players' error counts, average centipawn
loss, game accuracy and phase accuracies. Each move also has `accuracy` and
`centipawn_loss`. `analysis/lichess_accuracy.py` ports the local Lila checkout's
`AccuracyPercent.scala`, `AccuracyCP.scala` and `Advice.scala`, together with
scalachess 17.16.2's winning-chances and phase-divider rules and scalalib 11.10.12's
means. Game accuracy combines the volatility-weighted and harmonic means; it is
not an arithmetic mean of move accuracies. Centipawn losses are clamped at zero
after limiting evaluations to ±1000 CP. Phase calculations restart at +15 CP as
the referenced Lichess implementation does. Unavailable results display `—`.

The evaluation after a move comes from the next position's unrestricted search,
or the final played-move evaluation at the end of the game. Legal terminal results
take precedence. Standard games begin at Lichess's +15 CP; custom FEN positions
use their saved initial evaluation so an existing disadvantage is not charged to
the first move. Statistics may differ from the website when engine evaluations
differ. The report renderer inserts the exact two-player table into Performance
snapshot locally, without asking Codex to calculate or repeat it. The prompt's
`<!-- performance-statistics -->` placeholder is filled at validation/publication.
The saved Lichess game2 evaluation fixture reproduces the reference 83%/86% game
accuracy, all phase percentages, error counts and 46/33 average centipawn loss.

## Common runner and budgets

Production limits come from root `config.yaml`, including time, cumulative tokens,
model responses and report drafts. The chess-call allowance follows the configured
response limit. Test-specific limits
are configured only in `tests/coach/config.yaml`.

`run_coach(..., request=CoachingRequest(...))`, from `coach.agent_runner`, is the common execution interface.
A caller can supply instructions, task preparation, a report validator, output
name, tool access and execution limits. The production defaults produce a full
report; test scenarios supply their own request through the same function.
Production modules have no test-mode flags and do not import test modules.

Codex supplies cumulative usage events. The application interrupts on observed
limits; one in-flight response can exceed a threshold before usage arrives.
Internal Codex transport retries are runtime-managed. The whole-run timer and
tool-call limit provide additional bounds. Estimates are not exact tokenization
or a billing guarantee.

`COACH.MAX_MODEL_RESPONSES` (24) controls the **coaching agent's model-response budget**.
It does not limit Stockfish depth, positions analyzed, or Maia inference. The
initial whole-game pass still processes every move. Model responses are counted
from new cumulative usage events; when a turn reports no usage, it counts as one
estimated response. The same setting also gives the normal run a default allowance
of `4 × COACH.MAX_MODEL_RESPONSES` (96) chess-tool investigations, including locally prepared
investigations. A tool investigation may perform several Stockfish searches and
Maia predictions; 96 is not a limit of 96 engine searches. `--max-model-responses` changes
this agent budget for one run. Token and elapsed-time limits apply independently through
`COACH.TOTAL_TOKEN_BUDGET` and `COACH.RUN_TIMEOUT_SECONDS`.
`COACH.OUTPUT_TOKEN_ESTIMATE` is only a fallback accounting
estimate when usage is missing, not a cap on response length. Override it with
`--unreported-output-token-estimate` (the old `--max-tokens` spelling remains an alias).
`COACH.REPORT_ATTEMPTS` counts the initial draft and any corrections.

### Progress output

Coaching prints timestamped progress as it runs: preparation, the decision being
explored, checked branches and Maia ratings, report validation and completion.
Codex's public commentary messages explain the current focus and findings.
Raw reasoning events, planning text, opaque provider state and the final report
are not dumped into progress output. Commentary is kept separate from the saved
Markdown report. The Codex event distinction follows the
[official App Server documentation](https://learn.chatgpt.com/docs/app-server#items).

During quiet waits, a status line appears every `COACH.PROGRESS_INTERVAL_SECONDS` seconds
(15 by default, configured in root `config.yaml`). These updates add no model
requests or chess investigations. Brief model-authored commentary uses the normal
response budget. Programmatic callers can pass `progress=callback` to receive the
same messages or `progress=None` to disable them. The progress worker stops when
the run completes, fails or is interrupted.

## Outputs

Default full-run folder: `<PGN parent>/output/<PGN stem>-full/`. For example,
`games/game2.pgn` writes to `games/output/game2-full/`, and
`D:/Chess/rapid.pgn` writes to `D:/Chess/output/rapid-full/`. This is derived from
the input path, not the repository or current working directory.
`--analysis-only` and `--coach-only` use the same default. `--output-dir` overrides
the complete destination. No output directory is configured in YAML, and existing
reports are not moved. Test output remains under `tests/coach/output/`.

```text
analysis.json             game, player estimates and compact per-move evidence
player-rating/fit.json     numeric rating result and figure data
player-rating/analysis.svg accuracy curve beside method-specific points or posteriors
player-rating/prior.svg    rating-prior weights before normalization, 0–1 scale
initial_evidence.json     the prepared evidence package supplied before generation
investigations.json       detailed evidence from the latest agent investigation
agent_run.json            provider, model, status, investigated plies and token usage
coaching.md               accepted full report
coaching.draft.md         latest full draft, possibly rejected
agent_trace.jsonl         this run's tool arguments, timing, result IDs and status
usage.jsonl               append-only accounting across coaching runs
response_usage.jsonl      this run's per-response token deltas and context sizes
game.pgn                  original PGN
positions/*.svg           chess diagrams
```

Accepted reports replace previous accepted files only after validation. Detailed investigations are saved separately in investigations.json. Model reasoning and raw responses are not
saved; drafts contain only the proposed user-facing Markdown report. Requested
evidence goes to Codex using the existing ChatGPT sign-in.

Rating figures are generated locally for analysis-only runs and refreshed saved
analysis, using the same fitted distributions stored in `analysis.json`. They
are regenerated even when the saved rating signature is unchanged and replace
`player-rating/analysis.svg` and `player-rating/prior.svg` in the normal output
folder. The accuracy graph displays the retained-probability cutoff, effective
sigma and sigma scale. All rating axes display 200–3000; the numerical fitting
grid remains 0–3200. The standalone prior has the same canvas size as one
shared-accuracy panel. This does not require a coaching model call.

`response_usage.jsonl` records input, output, cached and uncached input, cumulative
totals, elapsed time and tool-response character growth for each observed response.
Codex entries use cumulative-usage deltas and preserve the provider's `last` usage;
duplicate/stale usage notifications are ignored. Cached input is a subset of input,
never added twice. Prompt/task/schema character sizes help identify overhead but
do not measure hidden runtime instructions. Missing/interrupted usage is explicitly
estimated, not invented as exact tokens. No raw model messages or reasoning are saved.

Full reports require three investigated selected-player decisions (or all if
fewer), coverage of actual game stages, and valid diagrams. Smoke reports require
one investigated decision and a diagram. These checks establish evidence coverage,
not a guarantee of correct chess interpretation.

## Engines and analysis

Reusable game analysis lives in [analysis/](../analysis/README.md); the coach
supplies the agent workflow and investigation tools. Shared `engine/` adapters
provide Maia and Stockfish. Current YAML defaults use four Stockfish workers
with four threads each and 512 MB total pool hash. `--threads-per-worker`
overrides the per-process thread count for a run.

The initial pass shares [analysis/stockfish_search.py](../analysis/stockfish_search.py) with the
Maia web app's **Analyze Entire Game**. It screens legal moves, searches the best
line, prioritizes the played move and Maia candidates, and refines alternatives.
The default bounded strategy shares a deadline across these phases and stops
at its depth ceiling or time budget. All search parameters come from the same
`ANALYSIS` configuration used by the web app; there are no duplicate coach search
settings. Initial analysis and focused checks both use
`ANALYSIS.STOCKFISH_EVALUATION`: currently a depth ceiling of 28, a default
10-second allowance, and a maximum requested allowance of 30 seconds. Actual
depths and phase timings stay in local cached evidence. The frontend's preset
scales do not apply to coach analysis.

Web continuation previews use `ANALYSIS.STOCKFISH_EXPLORATION`. YAML durations are seconds; existing
`--verify-ms`, `--max-ms`, and tool `movetime_ms` arguments remain milliseconds.
`--depth` sets the common depth ceiling; `--verify-ms` also sets the initial
bounded position allowance. There are no separate initial-analysis depth or
time-scale overrides.
These are per-search budgets, not whole-game timeouts. `MAIA.DEVICE: auto`
selects CUDA when available; history-aware inference is batched across positions.

The [test guide](../tests/README.md) contains offline checks and local profiling
commands. All test scenarios remain in `tests/coach`; normal production functions
do not accept test-only mode flags.

## Player-rating methods

The coach and web use the same player-rating method. The default is the current-game shared-curve affine fit:

```yaml
ANALYSIS:
  PLAYER_RATING:
    METHOD: shared_curve_affine
```

It uses only this game's Stockfish move qualities, complete equal-rating Maia policies at 600–2600 and supplied account ratings. It excludes single-legal-move positions, compares observed arithmetic accuracy with the game's shared accuracy curve, and uses the conditional variance of alternative human choices. The mean available account rating translates a common prior for both players. There is no population curve, calibration asset or access to another benchmark game's evidence. The [analysis guide](../analysis/README.md#configuration-and-ratings) gives the affine formula and its assumptions.

The default returns points only: `central_interval`, player `interval` and player `uncertainty` are null. Coaching must not invent a range. Its figures show the current-game curve, prior level, accuracy-based adjustments and final points; a curve intersection is not the final estimate. Account inputs can change both players' estimates through the common prior, without a fixed small sensitivity bound.

The current method is derived in the [Shared-Curve Affine Estimation paper](../docs/shared_curve_affine.md). The population-based `hierarchical_affine` and `uncertainty_ensemble` implementations and their papers have been removed, and `arithmetic_coverage` remains disabled. Previously generated reports and affected comparisons are historical; they must not be described as results from the new method. Compatible cached engine evidence can be refitted without reanalysis, but old rating conclusions require review after refresh.

`METHOD: bayesian_shared_curve` remains available. It uses only current-game arithmetic evidence and the fixed fourth-power prior, without account ratings, and returns a central 20% conditional posterior interval. That interval is not a calibrated error bound. There is no global YAML interval setting.

`analysis.json` records `played_elo`, `played_elo_method`, `played_elo_prior`,
`played_elo_central_interval`, `played_elo_rating_range`, `played_elo_interval_scope`, and estimator/evidence
metadata under `rating_fit`. The common `played_elo_name`,
`played_elo_description`, and `played_elo_diagnostics` fields describe the selected
estimator without coach-specific algorithm branches. Optional input-use metadata
is kept separate from point estimates and intervals. Numeric rating evidence stays in the configured cache outside the
agent's compact summary. `--coach-only` can refresh ratings from compatible saved
evidence. Legacy account-conditioned evidence cannot be relabeled as shared-curve
evidence: run `--analysis-only` once to rebuild it. Existing compatible engine
caches can be reused; no cache directories are moved.

Both active estimators use the same interface. The old supervised Bayesian/minimax
models and joint Maia probability fitter have been removed.
The runtime uses the [common rating interface](../analysis/README.md#player-rating-estimator-interface)
for evidence, method selection, result validation, and cache signatures. A new
method file defines a concrete `Rating(PlayerRating)` class and is selected
by its filename in YAML, without editing a registry or coach-specific code.

Neither method resolves all serial dependence, engine uncertainty, Maia error,
or calibration/generalization error. Their within-game ordering follows observed
arithmetic accuracy; this constraint is not proof that the estimates adjust fully
for different positional difficulty. Small samples and unusually easy or hard
positions should remain part of the coaching context.

The saved move evidence retains equal-rating policies, top-five choices, and probabilities
for the candidate union, always including played and Stockfish-best moves.
Stockfish screens every legal move internally. Positive evaluations favor White;
loss/gain is mover-relative and mate distances remain separate. Detailed numeric
evidence remains local rather than entering LLM context.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py"
```

Tests use fake model responses and engines without external LLM calls. References:
[Codex Python SDK](https://learn.chatgpt.com/docs/codex-sdk),
[dynamic tools](https://learn.chatgpt.com/docs/app-server#dynamic-tool-calls-experimental).

## Development rules

- Validate provider changes with offline tests first, then at most a small live `python tests/coach/smoke_test.py path/to/analysis.json` report (one position, 100–180 words).
- Do not generate a full live coaching report during development unless the user explicitly requests that full run. Do not repeatedly retry failed live tests.
- Use only Codex with the user's existing ChatGPT sign-in. Do not introduce direct LLM API providers or make API-key-backed calls.
- Keep application configuration in root `config.yaml`; no `getenv` overrides. Component-local `settings.py` modules may read YAML directly. Coach uses `coach/settings.py`, not `engine/settings.py`. Do not add a global `project_config.py` or shared configuration loader.
- Reuse the shared `ANALYSIS`, `MAIA` and `STOCKFISH` mappings; do not duplicate engine-analysis settings under `COACH`. Test-specific configuration stays in `tests/coach/config.yaml`.
- Close all engines and owned Codex processes after tests. Never leave a server running.

This README is the component's documentation. [prompt.txt](prompt.txt) is the runtime instruction resource; the isolated test prompt is [tests/coach/prompt-smoke.txt](../tests/coach/prompt-smoke.txt). Their `.txt` extension distinguishes executable prompt resources from documentation. Production and test scenarios use the same runner without test-only arguments in production code.
