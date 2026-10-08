---
name: chess-tests
description: Develop regression tests, experiments, coaching checks, and benchmarks with isolated fixtures, bounded live runs, and evaluation-only game data.
---

# Chess tests

Apply the [project guidance](../project-development/SKILL.md). Use [tests/README.md](../../../tests/README.md) for the shared environment, commands, and runners.

- Keep first-party tests, fixtures, experiments, and benchmarks under component directories in `tests/`, with only the shared README. Preserve dependency-owned tests and user scratch scripts, notebooks, and diagrams.
- Exercise common production interfaces; keep test-only options and prompts in tests. Do not add production smoke branches or runtime dependencies on test code.
- Test games and reference labels are evaluation-only, including unlabeled contexts. Never derive reusable training, calibration, normalization, or population assets from them; prefer synthetic inputs for invariants.
- Verify meaningful behavior with temporary storage, fake engines, and scripted model responses. Run focused offline checks first; ordinary tests must not spend model tokens, download assets, or start servers.
- Use bounded live checks only when needed for the requested work; full coaching reports require an explicit request. Begin with a small report, diagnose failures before retrying, and close owned processes.
- Isolate experiment outputs from production configuration and accepted results. Report mocked versus live work, cache state, settings, and limitations; compare output quality alongside performance.
