---
name: chess-analysis
description: Run or develop the shared chess analysis pipeline, accuracy measurements, analysis cache and runtime profiler.
---

# Chess analysis

Apply [project conventions](../project-development/SKILL.md). Read [analysis/README.md](../../../analysis/README.md) for current commands, measurement definitions and data contracts.

## Analysis runs

- **Do not refresh the analysis cache unless the user explicitly requests it.** Reuse existing compatible evidence by default. Repeating analysis, regenerating outputs or changing documentation does not authorize cache bypass or deletion.
- Use `--refresh-cache` only for an explicit refresh request. Preserve unrelated evidence and user data.
- When the user requires existing evidence only or forbids new searches, use `--analysis-only --rebuild-from-cache` (batch: `--rebuild-from-cache`). Missing evidence must stop the run rather than start engines. Ordinary analysis may calculate missing or incompatible requests; distinguish this from refreshing cached results.
- Run only the requested games and workflow. Analysis does not implicitly authorize coaching or a collection-wide rerun.

## Development

- Keep one analysis implementation shared by web and coach. Analysis owns calculations, search orchestration and result caching; engine adapters own inference and native processes.
- Reuse complete measurements with compatible history, engine and search context. Avoid duplicate payloads; preserve saved evidence and concurrent writes.
- Separate raw measurements, prepared analysis and derived figures. Render figures from saved data without engine work.
- Preserve legal-move alignment, rating scales and metric definitions. Distinguish descriptive spread from uncertainty and measured evidence from interpretation; do not fabricate missing values.
- Test mathematical and integration changes with synthetic evidence or fake engines. Use configured batching/concurrency and measured timing when optimizing performance.
