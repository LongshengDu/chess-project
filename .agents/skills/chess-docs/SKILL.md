---
name: chess-docs
description: Maintain component READMEs and standalone technical papers with accurate module descriptions, clear mathematics, and supported claims.
---

# Chess documentation

Apply the [project guidance](../project-development/SKILL.md). Consult the [project README](../../../README.md), relevant component README, and current implementation; standalone papers belong in [docs](../../../docs/).

- Keep one `README.md` per code component. Give each runtime Python module a responsibility table row, excluding `__init__.py`; describe test suites in the shared tests README instead of inventorying individual tests.
- Document current behavior and component boundaries. Update affected paths and commands after structural changes; link maintained contracts instead of duplicating them.
- Make papers self-contained: explain the problem, notation, principles, calculations, assumptions, and limitations. Exclude internal source links, installation instructions, and application walkthroughs.
- Anonymize paper examples and figures as White/Black without altering source games. Use supported calculations, portable figures, readable labels, consistent side colors, and selectable SVG text.
- Keep prose concise, with one source line per paragraph. Distinguish assumptions, measurements, and illustrative examples; never present test-derived assets as independent evidence.
- Verify links, mathematics, and rendering proportionately. Preserve unrelated user research and figures; do not start expensive analysis or coaching runs merely to revise documentation.
