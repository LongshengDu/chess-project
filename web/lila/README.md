# Lila UI integration boundary

The frontend imports reusable leaf modules from the pinned
`deps/lichess-lila` source tree. Vite resolves Lila's `@/` aliases into
`ui/lib/src`, while this package supplies the exact dependency versions used by
the pinned revision.

## Direct source reuse

| Used by Maia frontend | Pinned Lila source |
| --- | --- |
| Loose Snabbdom `hl` and `onInsert` helpers | `ui/lib/src/view/snabbdom.ts` |
| Wheel replay accumulation | `ui/lib/src/view/stepwiseScroll.ts` |
| Pointer-safe click/hold handling | `ui/lib/src/pointer.ts` |
| Accelerating hold repetition | `ui/lib/src/common.ts` (`repeater`) |
| Promotion state, geometry, rendering, and board mutation | `ui/lib/src/game/promotion.ts` (`PromotionCtrl`) |
| Captured-material calculation and VNodes | `ui/lib/src/game/view/material.ts` (`renderMaterialDiffs`) |
| Analysis-node completion and canonical IDs | `ui/lib/src/tree/node.ts` (`completeNode('standard')`) |
| Chess piece and mono material images | `public/piece/cburnett`, `public/piece/mono` |
| Standard move/capture/check/checkmate sound mapping | `public/sound/standard/Move.mp3`, `Capture.mp3`, `public/sound/Silence.mp3`, `standard/GenericNotify.mp3` |

`web/view.ts` adds only the integration behavior that this app already needs
around those modules. In particular, the promotion VNode receives the classes
required by this sibling-to-Chessground mount and retains keyboard/ARIA
activation. Material VNodes retain the prior IDs and accessible descriptions.
Replay buttons retain native keyboard activation in addition to Lila's pointer
and hold behavior.

## Necessary local adapters

| Local module | Why it remains local |
| --- | --- |
| `gameSetup.ts` | Lila's setup components require its global translations, mutable props, variants, lobby controller, board editor, and random-color flow. This app submits White/Black plus optional FEN to Flask. |
| `analyse/eval.ts` | Lila's tiny formatter shares a module with site dialogs and engine-runtime imports, so it is not independently consumable. |
| `analyse/moveTable.ts` | Maia probabilities and Python Stockfish results do not implement Lila's `AnalyseCtrl`, `OpeningData`, or browser-ceval contracts. |
| `analyse/tree.ts` | Flask identifies a branch by imported base ply plus variation UCIs; the wrapper stores that API metadata around direct Lila nodes. |
| `analyse/treeView.ts` | The upstream tree views require the full analyse controller, context menus, comments, glyphs, studies, translations, and preferences. |

These adapters preserve Lila-compatible selectors and interaction patterns, but
they are application code rather than copies intended to track upstream line
for line.

## Reuse ceiling

Importing `ui/round` would require Lichess sockets, accounts, clocks,
tournaments, translations, and server-rendered round data. Importing
`ui/botPlay` would instead make chessops' in-browser `Game` authoritative and
route bot moves through Lila's local bridge. Either choice would replace the
required Python/python-chess/Maia/Stockfish architecture.

The maintainable maximum is therefore direct reuse of the independent Lila
leaf modules above, with a small local controller and API-specific views. When
the submodule revision changes, run the strict frontend build and the browser
checks in `LICHESS_UI_AUDIT.md` before accepting the update.
