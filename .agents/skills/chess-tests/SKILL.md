---
name: chess-tests
description: Add or run regression tests, small coaching checks, experiments, and runtime benchmarks for this chess repository. Use for development validation and test organization, with strict separation from production code and benchmark-game training.
---

# Chess tests

Apply the [project guidance](../project-development/SKILL.md). [tests/README.md](../../../tests/README.md) is the shared guide for commands, fixtures, and current runners.

## Location and interfaces

- Keep first-party tests, experiment scripts, benchmarks, and fixtures under `tests/`, grouped by component such as `tests/analysis/`, `tests/backend/`, `tests/coach/`, `tests/engine/`, and `tests/web/`.
- Keep only the shared `tests/README.md`; do not create READMEs inside component test directories or document every individual test case.
- Dependency-owned tests remain with their dependency. Do not reorganize vendored projects merely to enforce first-party layout.
- Put smoke prompts, test configuration, smoke scripts, and smoke-only arguments under `tests/coach/`, not `coach/`.
- Production functions must expose common interfaces usable by both normal and test callers. Do not add `smoke=True`, test-specific branches, or runtime imports from tests.
- Keep Python project/dependency configuration at the repository root. Tests are not an independent Python project.
- Preserve user scratch scripts, notebooks, and generated chess SVGs. They are not disposable clutter because they are outside automatic test discovery.
- A file move or rename requires corresponding import, fixture-path, command, and documentation updates, including tests affected by runtime restructuring.

## Test-only game boundary

- Apply the project's test-only boundary to every experiment and fixture. Unlabeled contexts and leave-one-out assets remain prohibited; estimating each game from its own evidence is allowed.
- Attach commercial reference values after prediction. Do not pass them through inference, select coefficients by an automatic label-fitting objective, or disguise them as preprocessing data.
- Prefer synthetic inputs for mathematical and interface invariants. Frozen real-game evidence is an evaluation fixture, never a runtime dependency.
- Do not rebuild deleted population assets or re-enable withdrawn methods to make old tests pass. Keep obsolete expectations clearly distinguished from active coverage.
- Do not present repeated selection on this collection as independent validation. Report filtered cohorts and withdrawn cases honestly.

## Regression checks

- Use the established root environment and commands in the test README. Scope checks to the changed behavior and its shared callers before broadening them.
- Verify meaningful behavior and invariants rather than mirroring implementation details or checking prose verbatim without a contractual reason.
- Use fake engines, scripted model responses, in-process HTTP clients, and temporary files for ordinary regressions. Routine discovery should not spend model tokens, download assets, or start listening servers.
- Check shared web/coach behavior at the common pipeline boundary rather than implementing separate fixtures that mask divergent calculations.
- Rating changes need appropriate coverage for scale conversion, ordering, missing evidence, cache reuse, method replacement, and output semantics.
- A method that returns a point estimate must not pass by borrowing another method's interval or posterior drawing.
- Keep fixtures and test settings separate from production defaults. An experiment's chosen sigma, model, budget, or probability cutoff is not a project policy.
- Preserve accepted full reports and source analyses when testing renderers or coach changes; use a separate test output location.

## Small live coaching checks

- Validate offline first. When a live agent check is needed during development, produce a very small report before considering a full run.
- A full live coaching report requires a user request for that work; ordinary testing should not silently expand into one.
- Use the common Codex runner with a small test request and isolated test limits. Keep live test details in `tests/coach/`, not production functions.
- Reuse saved analysis when it is compatible. Do not redo full-game analysis just to check a report prompt.
- Record usage per response and total usage so repeated context or tool work is measurable.
- Do not repeatedly retry an expensive failed test without diagnosing it. Keep retries, tool investigations, response count, elapsed time, and tokens bounded by the test configuration.
- Close model and engine processes owned by the test. Do not leave server instances running when finished.

## Runtime measurements and experiments

- Keep benchmarks opt-in and separate from normal test discovery. Run the engine or full-game workload only when it answers the requested performance question.
- Distinguish cold runs from cache-warm runs and retain the settings and hardware used with their results.
- Record wall time, component work, cache hits, model batches, and concurrency where relevant. Summed concurrent work is not elapsed time.
- Do not attribute a batch's timing to separately measured individual positions; identify it as shared batch work.
- Compare analysis quality and output differences alongside speed. A faster run or changed score alone does not establish better chess accuracy.
- Use SVG for new rating comparison figures. Keep prior research and user artifacts separate from newly generated outputs.
- Experiment runners should leave production configuration, game inputs, and accepted outputs intact unless updating those outputs is the requested task.
- Do not hardcode a historical list of games or experimental values into general instructions. Discover inputs and expose task-appropriate experiment arguments.
- Report what was actually run, what was cached or mocked, and any skipped or withdrawn checks. Never count withdrawn cases as passing tests.

For component-specific changes, consult the corresponding skill and current README; this skill does not authorize unrelated cleanup or a new fitting method.
