---
name: chess-web
description: Develop the project's sole Maia frontend, upstream page integration, local backend adapters, and browser behavior.
---

# Local Maia web application

Apply the [project instructions](../project-development/SKILL.md). Read the [web README](../../../web/README.md) for current product, integration and build contracts.

- Keep web as the sole frontend. Reuse the upstream Maia pages and components while connecting them to the local Python backend.
- Preserve the agreed page layouts and functionality when adapting upstream code. Avoid replacing established chess interactions with simplified imitations or unrelated hosted features.
- Consume the shared analysis pipeline and canonical evidence. Obtain resolved engine settings from the backend instead of duplicating calculations, schemas or defaults in browser code.
- Keep saved-state restoration, progress and cancellation coherent. Display measured search status accurately and preserve user games through interrupted work.
- Keep product language focused on chess; expose implementation details only when they help users make decisions.
- Verify affected browser behavior and upstream compatibility, including builds when needed. Keep tests in the shared test tree, update component documentation, and stop servers started for verification.
