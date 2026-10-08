---
name: project-development
description: Apply shared architecture, configuration, data-preservation and workflow conventions when developing this chess project.
---

# Project development

Apply these conventions within the current request. Read the relevant component README and [config.yaml](../../../config.yaml) for current contracts and settings; skills do not authorize additional runs or cleanup.

- Keep code minimal, modular and object-oriented where state or resource ownership benefits. Give each Python module one coherent responsibility.
- Keep chess calculations in `analysis`, engine operations in `engine`, HTTP/persistence in `backend`, coaching in `coach`, and the single frontend in `web`. Web and coach share the analysis pipeline and reusable evidence.
- Use descriptive, consistent module-group names. Preserve unambiguous names; update all callers, tests and documentation when moving or renaming code.
- Use root `config.yaml` for application settings. Components read their own sections directly or through local settings modules. Do not introduce environment overrides, a global configuration gateway or duplicate analysis settings.
- Keep Python dependencies and tooling at the project root in one shared environment. Use concise configuration names with clear meanings and units.
- Separate engine assets, analysis caches, durable game history and generated reports. Output locations must not silently relocate caches. Share raw evidence without duplicating it in consumer-specific stores.
- Preserve user PGNs, annotations, scratch scripts, notebooks and chess diagrams. Limit cleanup to the requested scope. Support current interfaces without adding speculative or retired-format compatibility.
- Treat evaluation games as tests only. Neither labels nor unlabeled game data may train, calibrate, normalize or supply reusable population assets. A game's own evidence may analyze that game.
- Keep one `README.md` per code component and at the root. Inventory each runtime Python module except `__init__.py`; put standalone papers in `docs`.
- Keep tests, experiments and smoke-specific code under `tests/<component>`, with only the shared `tests/README.md`. Production and tests use common interfaces.
- Validate proportionately with local checks. Use a small live coaching test only when needed; a full coaching run requires a request. Report actual verification and close processes started for the work.

Use the relevant component skill rather than loading every workflow. For game runs and cache behavior, follow [chess-analysis](../chess-analysis/SKILL.md).
