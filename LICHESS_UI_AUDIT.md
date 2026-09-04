# Lichess round UI audit

This app is compared against the official `lichess-org/lila` source pinned as
the Git submodule `deps/lichess-lila` at commit
`f4da67dc45bba6769600db33af925c05ae21a8d0`.

## Verdict

The frontend previously used Lila directly only for Chessground and assets;
several small helpers were local copies or reimplementations. It now imports
the independently reusable Lila source for Snabbdom helpers, wheel navigation,
pointer/hold handling, promotion, material rendering, and standard analysis
node completion. The four obsolete local helper copies were removed.

The remaining local code is not a disguised copy of the full Lila app. It is
the integration boundary for the existing Flask/python-chess/Maia/Stockfish
API and for Maia-specific analysis. Importing `ui/round` would pull in Lichess
sockets and server state; importing `ui/botPlay` would move game authority to
an in-browser chessops `Game`. Neither is compatible with the required backend
design. Direct leaf-module reuse plus local API adapters is the practical
maximum that preserves current and future functionality.

## Feature and source audit

| Lichess play feature | Upstream source | Maia implementation | Status |
| --- | --- | --- | --- |
| Loose Snabbdom VNodes and insert hooks | `ui/lib/src/view/snabbdom.ts` | Direct `hl` and `onInsert` imports in all local views | Direct reuse |
| Responsive board/table/player layout | `ui/round/src/view/main.ts`, `ui/round/css/_app-layout.scss`, `_layout.scss` | Snabbdom view in `web/view.ts`, responsive desktop/mobile/zen rules in `web/style.scss` | Implemented |
| Drag and click moves, coordinates, legal destinations, animation, last move, check | `ui/round/src/ground.ts` | Chessground config in `web/controller.ts`; FEN/dests/check/last move supplied only by `backend/game.py` | Implemented |
| Right-drag arrows, right-click circles, modifier colors, context-menu suppression | `ui/round/src/ground.ts` | Chessground drawable configuration in `web/controller.ts` plus Snabbdom help view | Implemented |
| SAN move table and active move | `ui/round/src/view/replay.ts`, `ui/round/css/_moves-col2.scss` | Python SAN/history; Snabbdom-rendered Lila `aPp`/`qZM`/`Z7yx` notation | Implemented |
| First/previous/next/latest replay controls | `ui/round/src/view/replay.ts` | Six-cell replay bar with live-position glow | Implemented |
| Hold previous/next to repeat | `ui/lib/src/common.ts`, `ui/lib/src/pointer.ts`, `ui/round/src/view/replay.ts` | Direct `repeater` and `addPointerListeners` imports; local callback targets the Python-backed replay state | Direct reuse + adapter |
| Replay by mouse wheel over board | `ui/lib/src/view/stepwiseScroll.ts`, `ui/round/src/view/main.ts` | Direct `stepwiseScroll` import | Direct reuse |
| Replay keyboard bindings | `ui/round/src/keyboard.ts` | Controller bindings for Left/K, Right/J, Up/0/Home, Down/$/End | Implemented |
| Flip, zen, board menu, help shortcuts | `ui/round/src/keyboard.ts`, `view/boardMenu.ts` | F, Z, H, ?, Escape plus menu UI | Implemented |
| Exact promotion geometry and order | `ui/lib/src/game/promotion.ts`, `ui/lib/css/chess/_promotion.scss` | Direct `PromotionCtrl`; thin mount/accessibility adapter preserves this app's board placement and keyboard behavior | Direct reuse + adapter |
| Relative captured pieces and `+N` value | `ui/lib/src/game/material.ts`, `ui/lib/src/game/view/material.ts`, `ui/round/css/_material.scss` | Direct `renderMaterialDiffs` from the Python-supplied FEN and pinned mono piece assets; existing Python metadata supplies the accessible description | Direct reuse + adapter |
| Player color, orientation, rows, and turn indication | `ui/round/src/ground.ts`, `view/main.ts`, `view/replay.ts`, `view/table.ts`, `view/clock.ts` | The API exposes `humanColor`; the selected player color is at the bottom by default, Chessground's `opposite` helper derives the top color, and the active untimed `∞` clock follows the live turn | Implemented |
| Move/capture/check/checkmate sounds and sound toggle | `ui/round/src/ctrl.ts`, `ui/site/src/sound.ts`, `public/sound/standard` | Official pinned Lichess MP3 assets; Python emits Lila's base move/capture plus optional check/mate sequence, while separate human and Maia responses preserve move-by-move timing | Implemented |
| Board preferences | `ui/round/src/view/boardMenu.ts` | Flip, zen, blindfold, coordinates, sound, PGN copy, persistent local preferences | Implemented |
| Takeback, claim draw, resign confirmation | `ui/round/src/view/table.ts`, `view/button.ts` | Local action route; Python pops a human/Maia pair, validates draw claims, or records resignation | Implemented |
| Result and termination text in notation | `ui/round/src/view/replay.ts` | Result plus checkmate/stalemate/material/repetition/rule/resignation reason | Implemented |
| New game, side selection, and FEN start | `ui/round/src/view/button.ts`, `ui/lib/src/setup/view/color.ts`, `ui/lobby/src/view/setup/components/colorButtons.ts`, `fenInput.ts` | New Game opens a focused setup dialog for the standard board or a validated FEN and White/Black color cards; Maia moves automatically when the FEN's turn belongs to the bot | Implemented |
| Compact single-column move strip | `ui/round/css/_moves-col1.scss` | Horizontal, autoscrolling notation and stacked board/panel below 850 px | Implemented |
| Analysis tools composition and engine status | `ui/analyse/src/view/tools.ts`, `ui/lib/src/ceval/view/main.ts`, `ui/lib/css/ceval/_ctrl.scss` | Local `analyse__tools` and `ceval` adapter renders current-position Stockfish depth, loading bar, and Lila-formatted White-positive evaluation | Implemented |
| Explorer-style move probability table | `ui/analyse/src/explorer/explorerView.ts`, `ui/analyse/css/explorer/_explorer.scss` | Lila `explorer-box` / `table.moves` structure adapted to ranked Maia moves, Stockfish scores, percentage bars, PGN markers, and remaining mass | Implemented |
| Candidate hover arrows | `ui/analyse/src/explorer/explorerUtil.ts`, `ui/analyse/src/autoShape.ts` | Native delegated row hover/focus sets a `paleBlue` Chessground auto-shape | Implemented |
| Clickable analysis variation tree | `ui/analyse/src/treeView/columnView.ts`, `inlineView.ts`, `ui/lib/src/tree`, `ui/lib/css/tree/_tree.scss` | Direct `completeNode('standard')` supplies canonical nodes/IDs; the local Flask-path wrapper and `tview2` view keep imported mainline and Maia sidelines clickable and keyboard focusable | Direct reuse + adapter |
| Play moves into analysis | `ui/analyse/src/ctrl.ts`, `ground.ts` | Candidate clicks or moves by either color on Chessground select an existing child or add a Python-validated variation node, including promotions | Implemented |

## Deliberately not applicable

- Real chess clocks, berserk, expiration, and correspondence clocks: this fixed
  first version is explicitly untimed, so both player rows show Lichess-style
  active/inactive infinity clocks.
- Premoves and predrops: the product requirement explicitly locks movement while
  Maia calculates; predrops and variant pockets do not apply to standard chess.
- Remote draw/takeback offers, opponent-gone timers, abort, rematch negotiation,
  lag/online signals, rating changes, spectators, chat, tournaments, Swiss,
  simul, forecasts, and “new opponent”: these require another user or the
  Lichess server. Local takeback, draw claim, resignation, side-selectable new games, and PGN copy
  cover the corresponding single-player actions.
- Remote Lichess analysis links, voice move input, account preference pages, haptic
  account settings, and Lichess's alternate nonvisual UI are separate optional
  site tools/preferences rather than the active standard round surface. The
  local app still supplies semantic labels, keyboard replay, status live
  regions, and a keyboard/drawing help dialog.
- Lichess variants and three-check material icons: this app is intentionally a
  standard starting-position game against Maia.

## Verification checklist

- TypeScript strict check and Vite production build pass.
- Direct-import browser checks cover pointer and keyboard replay, canonical
  analysis paths, mainline/branch selection, material VNodes, and absence of
  runtime errors under real pointer input.
- Python contract checks cover legal moves, SAN/replay snapshots, material and
  score, takeback, resignation, draw claim, PGN result, Elo 1500, and cleanup.
- White- and Black-side game flows verify turn ownership, Maia's automatic move,
  player-oriented board rendering, standard/FEN roots, PGN colors and setup
  headers, invalid-FEN rejection, and resignation results.
- A real cached Maia3 UCI process was exercised as White and returned `1. e4`
  before the Black-side player's first turn.
- A real Maia3 UCI process was kept alive while `e4 e5 d4 exd4` was played; the
  rendered UI showed Black's captured pawn and `+1` and rolled both back on
  takeback.
- Rendered browser checks cover clickable notation, keyboard and wheel replay,
  menu/settings, flip/zen/blindfold/coordinates, result/resign confirmation,
  New Game, and desktop/mobile layouts.
- PGN analysis checks cover the Lila tools/ceval/explorer selectors, Maia and
  Stockfish table values, PGN highlighting, and a pale-blue board arrow while
  hovering or focusing a candidate row.
- Variation checks click `d4` from the Maia table, play `...e5` directly on
  Chessground, verify both nodes appear in the `tview2` branch, then jump back
  to imported `e4`; the 390 px layout keeps both tree and explorer visible.
- The direct Lila promotion overlay was exercised in live play and PGN
  analysis. All four piece images and accessible selectors loaded, Escape
  restored the pawn and controls, and knight underpromotion returned legal
  `g8=N` in both modes.
