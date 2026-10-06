---
name: chess-backend
description: Maintain this chess project's local Python HTTP backend, saved studies, play sessions, and analysis transport. Use for backend changes and backend integration; keep chess calculations in analysis and engine operations in engine.
---

# Chess backend

Apply the [project instructions](../project-development/SKILL.md) and read the [backend contract](../../../backend/README.md) for the affected service. These instructions capture enduring project choices; a later explicit user request can change product behavior.

## Responsibilities

- Keep HTTP validation and response mapping in route modules, request/job and play lifetimes in services, and persistence in the repository.
- Keep reusable chess calculations, full-game orchestration, accuracy, move hints, rating fitting, and runtime-profiler calculations in `analysis/`.
- Keep model inference, native processes, and engine adapters in `engine/`. Backend modules adapt and coordinate those components rather than duplicating them.
- Use the common full-game pipeline for both the web application and coach. Persist complete evidence, hints, accuracy, and both rating fits even when the current web UI does not display every field.
- Distinguish backend analysis job/transport modules from calculation modules. Profiler routes expose controls; they do not implement timing calculations or report generation.

## Integration and persistence

- Reuse the configured Maia adapter and Stockfish pool across play and analysis within an application, with explicit resource ownership and concurrency controls.
- Consult the backend README for the current history, persistence, cancellation, and cache-compatibility contracts when changing those features. These are implementation contracts, not additional user preferences frozen by this skill.
- Keep engine assets, reusable analysis caches, database storage, and generated output separate. Changing an output destination does not relocate caches.
- The current **Analyze Entire Game** contract saves full analysis; it does not automatically invoke a coaching agent or create a coaching export. Future web coaching should consume the same saved analysis through an explicit workflow.

## Configuration and structure

- Backend-local settings read `SERVER` for server/storage options and the appropriate shared sections as needed, following the project's YAML ownership rules.
- Expose resolved configuration to the frontend rather than copying defaults into request adapters. Do not create separate coach/web analysis constants.
- Apply the common naming, refactor, and module-documentation conventions to backend changes.

## Verification and lifecycle

- Put meaningful backend tests in `tests/backend/`, using the Flask test client and fake engines where possible. Cover the behavior changed, especially persistence, cancellation, and resource cleanup.
- Test integration through the common analysis interface rather than reproducing analysis mathematics inside backend tests.
- Do not leave servers or engine processes started for the task running after completion. Let the user start the application themselves unless they explicitly request otherwise.
- When performance is at issue, use the shared runtime profiler and separate elapsed wall time from overlapping engine work. Do not infer timing savings from cache hits alone.
