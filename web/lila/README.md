# Lila UI compatibility slice

These modules keep the local frontend close to the pinned Lila round UI without
loading Lila's site runtime. Refresh them deliberately when the
`deps/lichess-lila` submodule changes.

| Local module | Pinned Lila source |
| --- | --- |
| `snabbdom.ts` | `ui/lib/src/view/snabbdom.ts` |
| `stepwiseScroll.ts` | `ui/lib/src/view/stepwiseScroll.ts` |
| `repeater.ts` | `ui/round/src/view/replay.ts` (`repeater` behavior) |
| `promotion.ts` | `ui/lib/src/game/promotion.ts` |
| `gameSetup.ts` | `ui/lib/src/setup/view/color.ts`, `ui/lobby/src/view/setup/components/colorButtons.ts`, and `fenInput.ts` |
| `analyse/eval.ts` | `ui/lib/src/ceval/util.ts` (`renderEval`) |
| `analyse/moveTable.ts` | `ui/analyse/src/view/tools.ts`, `explorer/explorerView.ts`, `explorer/explorerUtil.ts`, and `ui/lib/src/ceval/view/main.ts` |
| `analyse/tree.ts` | `ui/lib/src/tree`, `ui/analyse/src/ctrl.ts` (root/node/path model) |
| `analyse/treeView.ts` | `ui/analyse/src/treeView/columnView.ts`, `inlineView.ts`, and `ui/lib/css/tree/_tree.scss` |

`web/view.ts` also preserves Lila's round selectors (`round__app`, `aPp`,
`qZM`, `Z7yx`, `i5d`, and `bo3`) so behavior and SCSS can be compared or
refreshed component by component. `LocalRoundController` replaces Lila's site,
socket, account, tournament, translation, and clock dependencies with the
local Flask API surface. The PGN surface follows Lila's tools order and uses
its `analyse__tools` / `ceval` / `analyse__moves` / `explorer-box` /
`table.moves` composition plus its explorer-row-to-`paleBlue`-Chessground-arrow
interaction. Candidate clicks and legal Chessground moves enter the local Lila-style
tree; Python validates and returns every new variation node.

Player-side behavior follows `ui/round/src/ground.ts` and `view/main.ts`: the
player color is the default board orientation, a separate flip flag reverses
it, and the opposite color is rendered above the board. The local code imports
Chessground's shared `Color` type and `opposite` helper instead of duplicating
them. This supports games as White or Black without loading Lila's socket- and
account-dependent round controller.
The New Game dialog combines Lila's setup color-card structure with its
from-position FEN input. Random color and the full clock/rating/variant lobby
form are intentionally omitted from this local standard-chess surface.
The color-dependent player and material siblings are keyed so Snabbdom moves
the existing board host instead of replacing it. The mount hook also destroys
and recreates Chessground if a future view change replaces that host, avoiding
a controller that targets detached DOM.

Live-game sound timing follows `ui/round/src/ctrl.ts`: the local move sound is
played before the later opponent move sound. Unlike Lila's socket-backed game,
the local app gets both sound classifications from two small Python responses,
so captures and checks remain authoritative without importing Lila's site-wide
sound runtime or duplicating chess rules in TypeScript.
Each ply follows Lila's base-plus-modifier sequence: move or capture first, then
check or checkmate 100 ms later. The standard check is intentionally silent;
checkmate uses the standard notification sound. A checking capture therefore
still plays its capture sound instead of being collapsed into silence.
