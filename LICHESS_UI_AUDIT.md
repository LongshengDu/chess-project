# Lichess round UI audit

This app is compared against the official `lichess-org/lila` source pinned as
the Git submodule `deps/lichess-lila` at commit
`f4da67dc45bba6769600db33af925c05ae21a8d0`.

The upstream round controller cannot run independently: it expects Lichess
socket messages, account preferences, translations, tournaments, clocks, and
server-rendered data. The app therefore reuses Chessground and the official
standard sound files directly, while adapting the locally applicable round UI
behavior to the small Flask/python-chess state contract.

| Lichess play feature | Upstream source | Maia implementation | Status |
| --- | --- | --- | --- |
| Responsive board/table/player layout | `ui/round/src/view/main.ts`, `ui/round/css/_app-layout.scss`, `_layout.scss` | `web/index.html`, responsive desktop/mobile/zen rules in `web/style.css` | Implemented |
| Drag and click moves, coordinates, legal destinations, animation, last move, check | `ui/round/src/ground.ts` | Chessground config in `web/main.ts`; FEN/dests/check/last move supplied only by `app.py` | Implemented |
| Right-drag arrows, right-click circles, modifier colors, context-menu suppression | `ui/round/src/ground.ts` | Chessground drawable configuration plus help dialog | Implemented |
| SAN move table and active move | `ui/round/src/view/replay.ts`, `ui/round/css/_moves-col2.scss` | Python SAN/history; clickable three-column notation sheet | Implemented |
| First/previous/next/latest replay controls | `ui/round/src/view/replay.ts` | Six-cell replay bar with live-position glow | Implemented |
| Hold previous/next to repeat | `ui/round/src/view/replay.ts` (`repeater`) | Pointer-hold repeater in `web/main.ts` | Implemented |
| Replay by mouse wheel over board | `ui/round/src/view/main.ts` | Throttled board-frame wheel navigation | Implemented |
| Replay keyboard bindings | `ui/round/src/keyboard.ts` | Left/K, Right/J, Up/0/Home, Down/$/End | Implemented |
| Flip, zen, board menu, help shortcuts | `ui/round/src/keyboard.ts`, `view/boardMenu.ts` | F, Z, H, ?, Escape plus menu UI | Implemented |
| Exact promotion geometry and order | `ui/lib/src/game/promotion.ts`, `ui/lib/css/chess/_promotion.scss` | Full-board overlay; queen, knight, rook, bishop on the promotion file; backdrop cancel | Implemented |
| Relative captured pieces and `+N` value | `ui/lib/src/game/material.ts`, `ui/lib/src/game/view/material.ts`, `ui/round/css/_material.scss` | Python computes role-count differences and 9/5/3/3/1 score for every replay ply; Cburnett mini-pieces render above/below board | Implemented |
| Player rows and turn indication | `ui/round/src/view/user.ts`, `view/table.ts`, `view/clock.ts` | Maia 1500 / Maia3-5M and You / White rows; active untimed `∞` clock follows the live turn | Implemented |
| Move/capture/check/checkmate sounds and sound toggle | `ui/round/src/ctrl.ts`, `public/sound/standard` | Official pinned Lichess MP3 assets imported by Vite; Python labels move events | Implemented |
| Board preferences | `ui/round/src/view/boardMenu.ts` | Flip, zen, blindfold, coordinates, sound, PGN copy, persistent local preferences | Implemented |
| Takeback, claim draw, resign confirmation | `ui/round/src/view/table.ts`, `view/button.ts` | Local action route; Python pops a human/Maia pair, validates draw claims, or records resignation | Implemented |
| Result and termination text in notation | `ui/round/src/view/replay.ts` | Result plus checkmate/stalemate/material/repetition/rule/resignation reason | Implemented |
| New game/rematch behavior | `ui/round/src/view/button.ts` | Immediate New Game resets the standard position without restarting Maia | Implemented |
| Compact single-column move strip | `ui/round/css/_moves-col1.scss` | Horizontal, autoscrolling notation and stacked board/panel below 850 px | Implemented |

## Deliberately not applicable

- Real chess clocks, berserk, expiration, and correspondence clocks: this fixed
  first version is explicitly untimed, so both player rows show Lichess-style
  active/inactive infinity clocks.
- Premoves and predrops: the product requirement explicitly locks movement while
  Maia calculates; predrops and variant pockets do not apply to standard chess.
- Remote draw/takeback offers, opponent-gone timers, abort, rematch negotiation,
  lag/online signals, rating changes, spectators, chat, tournaments, Swiss,
  simul, forecasts, and “new opponent”: these require another user or the
  Lichess server. Local takeback, draw claim, resignation, New Game, and PGN copy
  cover the corresponding single-player actions.
- Analysis-board links, voice move input, account preference pages, haptic
  account settings, and Lichess's alternate nonvisual UI are separate optional
  site tools/preferences rather than the active standard round surface. The
  local app still supplies semantic labels, keyboard replay, status live
  regions, and a keyboard/drawing help dialog.
- Lichess variants and three-check material icons: this app is intentionally a
  standard starting-position game against Maia.

## Verification checklist

- TypeScript strict check and Vite production build pass.
- Python contract checks cover legal moves, SAN/replay snapshots, material and
  score, takeback, resignation, draw claim, PGN result, Elo 1500, and cleanup.
- A real Maia3 UCI process was kept alive while `e4 e5 d4 exd4` was played; the
  rendered UI showed Black's captured pawn and `+1` and rolled both back on
  takeback.
- Rendered browser checks cover clickable notation, keyboard and wheel replay,
  menu/settings, flip/zen/blindfold/coordinates, result/resign confirmation,
  New Game, and desktop/mobile layouts.
- The promotion overlay was exercised from a promotion-ready Python position;
  its four selectors loaded and queen promotion returned legal SAN.
