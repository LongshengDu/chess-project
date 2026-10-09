---
name: chess-games
description: Work with PGNs and saved game analysis while preserving user data, rating context, cache reuse, and the games' evaluation-only role.
---

# Chess games

Apply the [project guidance](../project-development/SKILL.md), [analysis contract](../../../analysis/README.md), and [test guide](../../../tests/README.md).

- Preserve PGN moves, headers, and annotations unless editing them is requested. Reanalysis does not authorize rewriting source games; preserve unrelated research, notebooks, and diagrams.
- Treat supplied games as evaluation data only. Neither reference labels nor unlabeled positions may become training, calibration, normalization, or population assets. Each game's own evidence may support its analysis.
- Keep original metadata separate from effective runtime overrides. Respect declared site and time-control rating scales; require missing context rather than guessing. Commercial estimates are comparison data, never analysis or coaching inputs.
- Follow the [analysis reuse policy](../chess-analysis/SKILL.md). Check full game history and context; honor `--rebuild-from-cache` and explicit refresh requests without silently starting new searches.
- Match the requested scope: analysis, coaching, rendering, comparison, or cleanup. Discover requested inputs dynamically and isolate experiment outputs under tests. Do not launch collection-wide or expensive live runs unless requested.
- Report actual work, cache reuse, and unavailable evidence honestly. Distinguish accuracy metrics and rating scales; avoid unsupported conclusions about player ability from cross-game comparisons.
