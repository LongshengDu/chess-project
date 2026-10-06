---
name: chess-games
description: Work with the PGN collection and saved per-game results in this chess repository, including rating comparisons, metadata corrections, and requested reruns. Enforces the collection's test-only role and preserves user game data.
---

# Chess games

Apply the [project guidance](../project-development/SKILL.md). Use [analysis/README.md](../../../analysis/README.md) for saved analysis, rating scales, and fitting contracts, and [tests/README.md](../../../tests/README.md) for evaluation runners.

## Role of the collection

- `games/` contains user PGNs and evaluation cases. Apply the project's test-only boundary, including the prohibition on reusable assets made from unlabeled contexts or other games in a leave-one-out run.
- Current-game inference is allowed: use that game's own evidence and permitted actual-rating inputs to estimate that game.
- Keep actual player ratings separate from commercial played-strength estimates. Actual ratings are declared inputs where the selected method uses them; estimate headers remain testing labels.
- Attach reference headers only to completed predictions for comparison, never to fitting.
- Do not promote historical results involving cross-game fitting assets as clean test-only evidence.

## PGN integrity and scale

- Preserve source PGNs, moves, headers, and user annotations unless their modification is part of the request. Reanalyzing a game does not authorize rewriting it.
- Read fresh PGN headers when comparing results: references and actual ratings can be revised after an earlier saved run.
- Infer the declared rating scale from `Site` and `TimeControl` through the shared converter. Maia's internal scale is Lichess Blitz; do not compare numbers from different scales without conversion.
- Keep output ratings, commercial comparison labels, axes, and table descriptions in the same declared scale.
- Check game identity, starting position, move sequence, and evidence compatibility before reusing a saved analysis. A matching output folder name alone is insufficient.
- Retain scale provenance and make missing or unsupported metadata explicit according to the shared interface.
- A standalone paper's example should use the PGN and relevant actual-rating inputs, not unrelated commercial-estimate headers.

## Requested runs

- Discover the applicable PGNs from the requested directory or explicit list; do not freeze an old game count or filename range into a general runner.
- Distinguish a full engine analysis from a rating-only refit and from a figures-only refresh. Execute the requested level of work and describe it accurately.
- Reuse compatible cached same-game engine evidence for a rating-only change. Do not rerun expensive searches unnecessarily.
- Normal per-game output defaults to `<PGN parent>/output/<PGN stem>-full/`. An output override does not implicitly relocate the shared analysis cache.
- On a fitting run, refresh the fit and rating SVGs even if compatible evidence is cached. Replace only known generated artifacts and preserve unrelated files.
- When only fitting is requested, preserve all non-rating analysis evidence and source PGNs.
- When using saved data, identify the method and configuration that actually produced a result. Old cached estimates must not be relabeled as new calculations.
- Do not launch full collection runs, live coaching, or a server just because this skill is loaded. Follow the scope of the user's request.

## Comparisons

- Report White and Black consistently, with clear method names and a stated rating scale. Join commercial references only after independent per-game inference.
- Mark cases whose observed accuracy does not intersect the measured Maia shared curve when that distinction matters to the comparison. Do not hide edge cases in aggregate statistics.
- Keep comparisons of within-game White/Black ordering separate from above/below-actual performance and numerical error; these answer different questions.
- If the user requests ordering evaluation, distinguish sign agreement from the requested tie tolerance. A reference tie requires a sufficiently small fitted gap; a small fitted gap does not require the reference itself to be tied, provided a nonzero reference ordering is respected. Use the task's evaluation tolerance rather than freezing an old number here.
- Do not impose a historical MAE target or parameter grid on a new task. State which games qualified for a filtered comparison and why.
- Repeated testing and method selection against the collection are not evidence of independent generalization. Describe results as results on these test games.
- Do not silently compare freshly calculated results with withdrawn corpus-based numbers as though both were valid current methods.

## Preservation and organization

- Keep development runners and comparison artifacts under the relevant `tests/` component rather than adding scripts to `games/` or runtime packages.
- Apply the project preservation rules to scratch scripts, notebooks, and generated chess diagrams.
- Preserve historical research records unless deletion is explicitly requested; keep withdrawal status visible where applicable.
- Refreshing renderer-owned figures is different from deleting user diagrams or all older research output.
- Follow [chess-player-rating](../chess-player-rating/SKILL.md) for estimator design, the [analysis skill](../chess-analysis/SKILL.md) for pipeline changes, and the [tests skill](../chess-tests/SKILL.md) for experiments. Do not add a training stage to an evaluation runner.
