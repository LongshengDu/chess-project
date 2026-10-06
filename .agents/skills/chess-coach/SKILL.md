---
name: chess-coach
description: Develop the chess project's Codex coaching agent, chess tools, prompts, and report generation while preserving human-centered analysis, bounded development runs, and reusable local evidence.
---

# Chess coaching

Use this skill for work in `coach/` and its integration with shared analysis. Read the [project instructions](../project-development/SKILL.md) first and the relevant contracts in [coach/README.md](../../../coach/README.md). Loading this skill does not request or authorize a coaching run.

## Architecture and execution

- Keep Codex as the only coaching agent, using the user's existing ChatGPT sign-in. Do not restore Gemini, OpenRouter, smolagents, or direct LLM API providers unless the user explicitly changes that requirement.
- Read current model, reasoning effort, budgets, and output settings from root YAML through the component's settings module. Do not freeze past model choices or budget numbers into new code or this skill.
- Reuse the same full-game analysis and rating-fitting pipeline as the web application. Coach consumes its results; it must not maintain a second analysis implementation.
- Reuse shared `ANALYSIS`, `MAIA`, and `STOCKFISH` settings instead of duplicating them under `COACH`. Agent response limits and investigation allowances are agent settings, distinct from engine search limits.
- Keep normal and test runs on the same common interface. Smoke arguments, prompts, fixtures, limits, and runners belong in `tests/coach/`, not production `coach/` code.
- For changed agent behavior, validate offline first, then use a very small live report when live validation is needed. Generate a full live report only when requested; repeated failures are a reason to fix the integration, not spend more tokens on full retries.
- Default generated reports to the PGN parent's `output/<game-name>-full` directory. Honor explicit output overrides and keep reusable engine caches in their configured location.
- Close owned Codex processes and engines after success, failure, or interruption. Do not leave development servers running.

## Evidence and token use

- Run local chess analysis before requesting coaching. Keep the complete `analysis.json` and detailed fitting evidence local; send a compact summary and selected evidence initially.
- Prepare a few useful critical-moment investigations locally and supply them together. Cover the stages actually reached, including useful quiet decisions, without hardcoding named moves or sample games.
- Allow multiple independent chess investigations in one agent response through the common tool interface. Use follow-up calls for missing facts, lines, or ideas that need clarification.
- Reuse saved evidence and previous tool results. Avoid repeatedly sending full game histories, duplicate distributions, or facts already supplied.
- Calculate short per-move hint arrays during game analysis and save them alongside each move in `analysis.json`. Coaching reads these flags; it does not duplicate or independently recalculate them in summaries or prompts.
- Treat hints as attention cues, not proof of intent, explanations, or automatic coaching judgments. Unflagged moves can still carry positional lessons.
- Record token usage per model response, together with aggregate usage and tool accounting, so growth and retries can be inspected.
- Print concise progress describing the agent's current investigation and checked findings while coaching proceeds. Do not present fabricated thinking or unsupported conclusions.
- When coaching generation is requested again, refresh the report even if the underlying compatible analysis is cached. This does not enable coaching implicitly from the web's analysis-only action.

## Human play is the foundation

- Base both praise and criticism on likely human choices near the opponent's actual and fitted level, Maia rating trends, counterplay, and how practical the player's continuation is to find and execute.
- Use Stockfish to check tactical validity and strongest resistance. Do not assume the opponent plays its best defense or use engine ranking alone as the coaching verdict.
- Distinguish objective soundness, practical merit, and the result in the actual game. A small evaluation loss can accompany a useful practical decision; a trap also needs its sound defenses and their likelihood examined.
- Compare played and alternative choices through their likely replies and subsequent ideas. A higher-rated first move without its continuation is not a complete lesson.
- Investigate what happened before a critical moment: earlier choices, developing positional weaknesses, pawn structure, piece placement, and the opponent's setup. Connect those facts to the later decision instead of only examining its aftermath.
- Check proposed earlier improvements with the same human-reply and tactical standards as the critical move itself.
- Teach decisions and ideas: tactics, plans, strategic choices, and positional judgments across moves. Do not reduce coaching to a list of evaluation changes.
- Use balanced coverage across the game. Do not special-case a memorable move or repeatedly expand one example into a supposed recurring weakness.
- Verify claims through legal chess lines and valid chess knowledge. If the idea or causal explanation remains unclear, say so rather than inventing it.

## Ratings and attribution

- The coaching request must identify the player being coached and their actual Elo; do not invent missing identity or rating, or add a report instruction asking a later reader to fill it in.
- Compare actual Elo with fitted played strength. Neither value automatically overrides the other; the difference is useful evidence about this performance.
- Respect the declared site/time-control scale. Convert account and displayed ratings before comparing them with Maia's native rating anchors or passing them to chess tools.
- Praise supported above-level decisions and constructively criticize avoidable below-level decisions using the same practical standard.
- State the exact higher Maia rating supporting a choice beside its checked variation. Separate Maia-supported choices from Stockfish verification; an engine PV is not a complete line played by a higher-rated human.
- Do not invent probabilities, higher-rated support, confidence ranges, or interval interpretations. Use the selected estimator's saved contract, including point-only estimates when applicable.

## Reports that teach chess

- Produce a readable coaching report, not a technical execution report. Keep provider logs, token accounting, fit diagnostics, and search internals in their appropriate local records.
- Include a Performance snapshot with actual/fitted ratings and the shared Lichess-style accuracy, average centipawn loss, error counts, and available stage accuracies. Missing values are not zero.
- Preserve the intended optional headings:
  - `## Best decisions (only when supported)`
  - `## Worst decisions (only when supported)`
- Apply the user's wording: “Best/Worst concern decisions and ideas, not evaluation rankings. Include either/both/neither as suitable; avoid duplicated stage analysis or turning one example into a recurring habit. Apply the higher-Elo lessons in the plan and exercises.”
- Explain the concrete reasons for important mistakes and the skills to improve. Turn higher-rated ideas into useful thinking cues, learning tips, a prioritized plan, and exercises supported by the investigation.
- SVG chessboards are important teaching content. Show critical positions and meaningful steps in continuations alongside checked numbered notation and explanations.
- Use enough boards to make the buildup, decision, likely reply, and resulting idea understandable. Balance does not mean fewer diagrams, less useful explanation, or an arbitrary image cap.
- Caption actual versus hypothetical positions and clarify what the reader should notice. Render boards from verified histories and use real generated paths.
- Keep validation focused on the report contract and verified evidence. Do not force unsupported Best/Worst sections or repeatedly retry because stylistic keywords are missing.
