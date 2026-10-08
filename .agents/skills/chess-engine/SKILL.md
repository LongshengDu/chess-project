---
name: chess-engine
description: Develop Maia inference, Stockfish adapters, engine assets, concurrency, and native process lifecycle.
---

# Chess engine operations

Apply the [project instructions](../project-development/SKILL.md). Read the [engine README](../../../engine/README.md) for current interfaces and lifecycle contracts.

- Own model loading, inference, engine adapters, downloaded assets and native resources. Keep analysis policy, chess metrics and reusable result caching in analysis.
- Provide shared adapters for all callers. Make ownership explicit so releasing borrowed resources cannot stop another application's engines.
- Batch independent Maia inference while preserving game history and input alignment. Use configured GPU support with a working CPU path; Stockfish parallelism uses CPU resources.
- Reuse workers and account for aggregate processor and memory demand. Measure startup, inference, search and elapsed time separately before choosing optimizations.
- Respect configured search limits and cancellation. Report achieved depth and completion accurately, rather than treating a requested target as an observed result.
- Validate changed behavior with focused tests in the shared test tree, using fakes where practical. Keep component documentation current and close resources started for verification.
