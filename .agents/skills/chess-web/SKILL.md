---
name: chess-web
description: Maintain this chess project's sole local Maia frontend in web, including upstream analysis/play page integration, local Python adapters, navigation, build tooling, and browser behavior.
---

# Local Maia web application

Apply the [project instructions](../project-development/SKILL.md) and read the [web contract](../../../web/README.md). Use the [backend contract](../../../backend/README.md) for API behavior and [analysis contract](../../../analysis/README.md) for shared results and search settings.

## Product and upstream integration

- `web/` is the single frontend. Keep analysis as the landing page and **Play Maia** as a direct navigation entry alongside **ANALYSIS**; a single play option does not need a dropdown.
- Reuse the real pages and components from `deps/maia-platform-frontend`. Preserve the upstream analysis and play layout and functionality rather than substituting a simplified imitation.
- Preserve the analysis board, move navigation, Moves by Rating, analysis panels, options, custom PGN/FEN import, and responsive behavior when changing local adapters.
- Preserve the upstream Play Against Maia setup and play flow while using the local Python backend and shared Maia settings.
- Keep unrelated hosted platform navigation and service dependencies out of this local application unless the user explicitly asks to add them.
- Keep local adaptations compatible with the upstream frontend. Read the web README for current adapter, patch, and build mechanisms rather than treating a particular mechanism as a permanent user requirement.
- Keep user-facing flows about chess. Show implementation details only when they help the user make a meaningful decision; do not expose internal module/configuration terminology as product copy.

## Shared analysis and saved state

- **Analyze Entire Game** uses the same full-game Python pipeline as the coach, including move evidence, hints, accuracy, and rating fitting for both players.
- Save complete results even where the UI currently hides a field, so future rating panels and coaching can reuse the canonical result.
- The current analysis button caches analysis and does not create a coaching report, expose a **For your coaching AI** export, or call an LLM. Add future web coaching as an explicit workflow when requested.
- Keep progress, cancellation, saved-result restoration, and repeated analysis coherent. Avoid duplicate interactive searches or autosaves overwriting an active full-game result.
- Respect backend cache compatibility and completion status. Display actual achieved depth/time-limit status rather than implying every search reached its requested depth.
- Preserve PGN/FEN import and local play-to-analysis handoff when adapting upstream data structures. Do not invent a separate frontend rating calculation or evidence schema.

## Settings and structure

- Web-local Python build settings read `FRONTEND`, following the common YAML ownership rules.
- Get resolved engine settings and analysis presets from the backend instead of duplicating defaults in browser code.
- Frontend depth presets scale the shared evaluation time limits; they do not redefine coach analysis or continuation exploration budgets.
- Apply the common module naming, structural-migration, and README inventory rules to adapters and Python build modules.

## Verification and lifecycle

- Place browser/integration tests and benchmarks under `tests/web/`; preserve existing scratch and generated chess graphics instead of deleting them as cleanup.
- Verify changed behaviors and upstream compatibility, including navigation, save/restore, cancellation, rating charts, and legal play where affected. Build/type-check when changes affect the frontend bundle.
- Use the shared runtime profiler for performance work. Do not confuse browser rendering, network transport, cached replay, and engine search time.
- Stop any development or backend server started for verification when finished. The user should be able to start the application themselves.
- Keep durable component documentation in `web/README.md`; current configuration and source are authoritative for numeric defaults and commands.
