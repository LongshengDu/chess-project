---
name: project-development
description: Apply the user's durable architecture, configuration, data-separation, documentation, and workflow instructions when working in this local Maia chess project. Use with the relevant component skill for implementation, audits, refactors, or experiments.
---

# Chess project conventions

These instructions distill recurring project requirements. Apply them to the current request; they do not authorize rerunning past experiments, generating reports, starting services, or changing unrelated components. A later explicit user instruction takes precedence. Read the current configuration and relevant component README for implementation details rather than treating historical settings as fixed requirements.

## Architecture and Python layout

- Keep the project minimal, modular, and object-oriented where state or resource ownership benefits from it. Each Python file represents one coherent responsibility; avoid duplicate implementations, overlapping wrappers, and a central module that owns unrelated components.
- Keep engine operations in `engine/`, chess analysis in `analysis/`, HTTP and persistence in `backend/`, coaching orchestration in `coach/`, and the sole frontend in `web/`.
- The web and coach must use the same full-game analysis pipeline, including player-rating fitting, even when a frontend does not yet display every result. Extend the shared implementation instead of making app-specific copies.
- Give related modules a consistent group-first name, such as `assets_maia.py` and `assets_stockfish.py`. Once a group has its own package, use short role names inside it: `analysis/player_rating/interface.py`, not a repeated `player_rating_` prefix. Preserve existing unambiguous names.
- Use `profiler` for profiling functionality and `runtime profiler` when specifically referring to timing; do not mechanically rename every occurrence to the longer term.
- During a structural move or rename, update all affected imports, tests, commands, documentation, and links. A restriction on an earlier cleanup task is not a permanent prohibition on updating those references.
- Keep Python dependency declarations, lockfiles, and shared Python tooling at the repository root. Components share the project environment; native frontend and dependency manifests retain their appropriate locations.

## Configuration ownership

- Use root `config.yaml` for application configuration, with `MAIA`, `STOCKFISH`, `ANALYSIS`, `COACH`, `SERVER`, and `FRONTEND` sections. Do not introduce application `getenv` overrides or another competing configuration source.
- Components read the YAML they need, directly or through their own `settings.py`. Do not add a global `project_config.py` or make `engine/settings.py` the configuration gateway for other components.
- `STOCKFISH` owns basic engine setup; `ANALYSIS` owns Stockfish search budgets, scheduling, workers, per-worker threads/hash, and shared analysis settings. Coach reuses those settings instead of creating duplicate coach analysis options. Frontend presets belong to `FRONTEND`.
- Use the shortest unambiguous names. Preserve clear names such as `HOST`, `PORT`, `MODEL`, and `DEVICE`; qualify genuinely ambiguous values and state units. Use consistent time units and distinguish per-worker resources from totals.
- Keep estimator-specific mathematical arguments in the estimator's Python code rather than promoting every experimental setting to global YAML. Read current model names, defaults, budgets, and paths from their owner instead of freezing old values in skills.

## Data, artifacts, and cleanup

- **Games in the evaluation collection are tests only.** Never use their labels or unlabeled positions, policies, accuracies, or other derived statistics to train, calibrate, normalize, set population priors, or construct reusable fitting assets for other games. Removing labels or excluding the current target does not make that acceptable.
- Each game's own cached numeric engine evidence may be reused to estimate that same game. Commercial PGN estimates are comparison targets, not estimator inputs or parameter-fitting data. Do not turn a benchmark into a tuning loop that merely reproduces its answers.
- Keep reusable caches at the configured cache location. Selecting a report, game-output, or profiler directory must not silently redirect caches there.
- Preserve user-written scratch scripts, notebooks, and generated chess SVGs. When reorganizing them, place them under `tests/` and update references; do not classify them as disposable because they look experimental.
- Remove stale code and generated artifacts only within the current cleanup scope. Historical experiments do not become active algorithms simply because their files remain available.

## Documentation, testing, and runs

- Root and runtime component guides are each a single `README.md`. Give each non-`__init__.py` runtime Python module its own responsibility-table row. Technical papers belong in `docs/`; skill instructions belong here in `.agents/skills/`.
- Put tests and development experiments in `tests/<component>/`, with one shared `tests/README.md`. Do not add READMEs under individual test component directories or inventory every test file.
- Normal and test runs share common interfaces. Keep smoke-only arguments, prompts, and scenarios inside `tests/`; do not add production `smoke=True` branches.
- For coaching changes, begin with local checks and a very small live report when needed. Do not generate an expensive full coaching report as a routine test. Record usage and expose useful progress for agent runs.
- Stop servers and engine processes started for the work when finished; leave normal server startup to the user unless the current request says otherwise.

## Component skills

Read only the skills needed for the task; do not load or execute every component workflow.

| Area | Skill | Focus |
| --- | --- | --- |
| Analysis | [chess-analysis](../chess-analysis/SKILL.md) | Shared evidence, metrics, rating algorithms, profiling, figures |
| Player-rating methods | [chess-player-rating](../chess-player-rating/SKILL.md) | New fitting models, drop-in interface, mathematical validation, evaluation separation |
| Backend | [chess-backend](../chess-backend/SKILL.md) | Local API, storage, shared services, lifecycle |
| Coach | [chess-coach](../chess-coach/SKILL.md) | Codex investigations, human-centered advice, report quality, token costs |
| Docs | [chess-docs](../chess-docs/SKILL.md) | Component guides and standalone technical papers |
| Engine | [chess-engine](../chess-engine/SKILL.md) | Maia and Stockfish operations, batching, resource budgets |
| Games | [chess-games](../chess-games/SKILL.md) | PGN inputs, rating scales, per-game outputs, evaluation isolation |
| Tests | [chess-tests](../chess-tests/SKILL.md) | Test layout, reusable interfaces, bounded experiments |
| Web | [chess-web](../chess-web/SKILL.md) | Maia frontend fidelity, local play and analysis integration |

Use the [project README](../../../README.md) for current setup and the [configuration](../../../config.yaml) for active values. Component skills add their own constraints without duplicating the project rules.
