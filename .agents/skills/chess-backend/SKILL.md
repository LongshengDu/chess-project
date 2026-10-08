---
name: chess-backend
description: Develop the local Python backend, game persistence, play sessions, and integration with shared chess analysis.
---

# Chess backend

Apply the [project instructions](../project-development/SKILL.md). Read the [backend README](../../../backend/README.md) for current interfaces and behavior.

- Own HTTP validation, transport, job lifetimes and persistence. Keep chess calculations in analysis and native engine operations in engine.
- Reuse the common analysis pipeline and evidence across web and coach. Adapt shared results rather than implementing separate calculations or caches.
- Preserve durable game history and user data. Keep reusable evidence, downloaded assets and generated output separate; cache operations must not erase history.
- Make resource ownership explicit. Coordinate concurrent requests and cancellation without closing engines borrowed from another owner.
- Read component configuration from project YAML and expose resolved settings to clients instead of duplicating defaults. Keep service, transport and persistence responsibilities modular.
- Verify affected behavior with focused backend tests and fake engines where practical. Keep tests in the shared test tree, update component documentation, and stop processes started for verification.
