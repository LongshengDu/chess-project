---
name: chess-coach
description: Develop the Codex chess coach, its tools, prompts, and reports.
---

# Chess coaching

Apply [project guidance](../project-development/SKILL.md). Read [coach/README.md](../../../coach/README.md) for current contracts and report requirements.

- Keep Codex as the sole coaching agent. Reuse shared analysis and compatible cached evidence instead of creating a separate coaching analysis pipeline.
- Base both praise and criticism on likely human choices, Maia support at the declared level, practical counterplay, and ease of execution.
- Use Stockfish to verify tactical soundness and stronger resistance, without assuming the opponent finds every engine defense.
- Investigate the buildup to critical moments, including earlier positional choices. Explain the tactical and strategic ideas behind likely continuations.
- Attribute higher-rated Maia support to the checked choices it supports. Distinguish human probabilities from engine verification and acknowledge unclear explanations without fabrication.
- Use chess tools when they clarify evidence or ideas. Batch independent investigations, reuse results, provide progress, and record usage per response.
- Use SVG boards alongside checked notation and explanations for critical positions and continuations. Diagrams should clarify decisions without replacing explanations.
- Cover the game proportionately. Include Best decisions or Worst decisions when supported; discuss ideas and decisions, avoid repeated examples, and derive actionable learning tips.
- Write reports for chess players. Keep developer instructions, operational details and generic methodology explanations in documentation or logs, outside the report.
- Validate changes offline first, then with a small live report when necessary. Generate a full live report only when requested; diagnose failures before retrying.
