---
name: chess-docs
description: Maintain this chess project's component READMEs and standalone technical papers, with accurate module inventories, publishable mathematics, and evidence provenance.
---

# Chess project documentation

Use this skill when editing repository documentation or technical papers. Read the [project instructions](../project-development/SKILL.md) first, then the relevant component README and implementation. Documentation must describe current behavior and supported claims, not repeat withdrawn conclusions.

## Component documentation

- Keep one documentation Markdown file named `README.md` at the project root and in each code component: `analysis`, `backend`, `coach`, `engine`, `tests`, and `web`.
- Consolidate component documentation there instead of adding extra component Markdown documents or nested component READMEs. The requested `.agents/skills/` tree and standalone papers in `docs/` serve separate purposes.
- Give each non-`__init__.py` Python module its own table row with a clear responsibility. Use package-relative paths when modules are grouped in subdirectories.
- Explain boundaries where names could suggest duplication: engine operation versus analysis, HTTP adaptation versus shared analysis, coaching versus evidence generation, and runtime profiling versus analysis results.
- Do not inventory individual test files or `__init__.py` files. Document test suites and useful development workflows in `tests/README.md` only; do not add READMEs under `tests/<component>/`.
- When moving or renaming code, update documentation paths, imports, commands, and references that depend on it. A previous cleanup exclusion does not justify leaving stale paths after a requested structural change.
- Use the project's root Python environment and dependency installation in run instructions. Do not invent separate component dependency installations.
- Keep current schema, configuration, and algorithm descriptions consistent with their implementation. Link to another component's maintained contract instead of copying a second divergent version.

## Standalone technical papers

- Papers in `docs/` must stand on their own after publication. Introduce the problem, inputs, notation, design reasoning, method principles, calculations, assumptions, and limitations needed to understand the result.
- Explain why each important mathematical choice is made and what it implies. Derive or substantiate claims such as optimality, ordering preservation, prior effects, and uncertainty instead of using labels as justification.
- Distinguish declared assumptions from measured evidence, conditional variability from calibrated error, and extrapolated tails from measured rating anchors.
- Keep mathematical notation internally consistent and define units, perspective, rating scale, aggregation, and boundary behavior when they matter.
- Use concise, informative prose. Keep each paragraph on one source line; use normal blank lines between paragraphs, not forced line breaks inside prose.
- Do not use repository source links or local implementation paths as explanatory support. A reader must understand the paper without access to this checkout.
- Do not turn the paper into instructions for running the application. Omit installation steps, CLI walkthroughs, and a Reproducibility section.
- Do not include contextless commercial PGN-estimate comparisons as paper evidence. Those belong to explicit evaluation artifacts with their provenance and limitations.
- Use a self-contained numerical or short-game example when helpful. For a game example, include the relevant moves and actual Elo/scale context; omit unrelated PGN headers and external estimated-rating labels.
- Add explanatory SVG figures when they clarify the curve, prior, uncertainty, calculation, or resulting estimate. Use readable axes, units, legends, and captions; figures must agree with the equations.
- Keep figure references portable and bundle the referenced assets with the paper. Relative links to publication figures are appropriate; private source links are not.

## Evidence and maintenance

- Verify the algorithm and values described against the current implementation before revising a paper or README. Do not preserve an obsolete claim merely because an older document contains it.
- Never describe benchmark-derived assets or development comparisons as independent test evidence. Benchmark games cannot supply training, calibration, normalization, prior, or population assets, even without reference labels.
- Synthetic illustrations and explicitly identified current-game calculations can explain a method without claiming external validation. Label the source of examples honestly.
- When an invalid method is replaced, remove or replace its problematic paper and dedicated figures when that cleanup is requested. Preserve unrelated user research and generated artifacts according to the project instructions.
- Keep active papers separate from withdrawn historical experiments. Do not silently reuse old comparison figures under a new method name.
- For generated figure updates, replace the corresponding old outputs rather than leaving several apparently current versions. Do not remove unrelated user diagrams.
- Check links, equations, figure labels, and rendering appropriate to the edit. Do not run expensive analysis, coaching, or parameter experiments merely to update prose unless the requested documentation needs that calculation.
