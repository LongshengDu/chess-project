# Maia Local Web Chess

A small local web chess game with a Lichess-inspired play layout. You play
White against Maia3-5M at Elo 1500. A small Snabbdom controller and Lichess
round-style views render the interface around Chessground, while `python-chess`
owns the game state, notation, and rules.

## Repository layout

```text
backend/app.py         Flask application and chess API
backend/analysis.py    PGN parsing, Maia policy, and Stockfish scoring
pyproject.toml         Python/uv project configuration
uv.lock                Locked Python dependencies
web/                   Frontend source and all pnpm/TypeScript configuration
deps/lichess-lila/     pinned Lichess source submodule (UI reference/assets)
deps/maia/             pinned Maia3 source submodule
deps/stockfish/        vendored Stockfish source and Windows executable
```

The notebook experiments and their supporting files also remain at the root.

## Install and run

Requirements: Python 3.10+, Git, Node.js 22+, `pnpm`, `uv`, and a web browser. From the
repository root in PowerShell:

```powershell
git submodule update --init --depth 1
uv sync
uv run maia3-cache --model maia3-5m --cache-dir .\backend\.cache\maia3
cd web
pnpm install
pnpm run build
cd ..
uv run python .\backend\app.py
```

Open <http://127.0.0.1:5000>. Stop the server with `Ctrl+C`.

`maia3-cache` downloads the 5M checkpoint into the project-local ignored
`backend/.cache/maia3` directory. Server startup checks that cache before
launching Maia, and the UCI process runs in local-files-only mode so model
downloads cannot consume its initialization timeout. After the initial setup,
rebuild the frontend only when `web/` changes; the normal run command is:

```powershell
uv run python .\backend\app.py
```

## Architecture

```text
Snabbdom round view + tiny LocalRoundController
    |                        |
    |                        +--> Chessground (display and input only)
    | local JSON routes
    v
Flask + ChessApi + python-chess (position, legality, outcome, history)
    | python-chess UCI support
    v
Maia3 (one persistent process)
```

The controller sends a UCI move such as `e2e4` to `POST /api/move`. Python
validates and pushes it, asks Maia for a `nodes=1` reply, pushes that reply, and
returns a complete display state. The controller converts the returned
destination object to Chessground's required `Map`; it does not calculate
moves. Promotion choices are also supplied by Python. Stopping the server runs
a `finally` block that sends the engine its UCI quit command.

The frontend follows Lila's controller → view → Snabbdom patch loop and reuses
adapted Lila helpers for VNodes, wheel replay, held replay buttons, promotion
geometry, round selectors, and responsive styling. Chessground is mounted from
a Snabbdom insertion hook and remains stable across redraws. Site-wide Lila
sockets, accounts, translations, chat, tournaments, and clocks are not loaded.

Python returns a display snapshot for every ply, including relative captured
material and material score. The move sheet, navigation buttons, mouse wheel,
and Lichess keyboard shortcuts use those snapshots to replay the game without
implementing chess rules in TypeScript. Chessground provides right-drag arrows
and circles, and promotion uses an in-board queen/knight/rook/bishop chooser.

The controls provide local takeback, draw claim, resign confirmation, and New
Game. Automatic draws and claimable threefold/fifty-move draws follow
`python-chess`. Board-menu preferences cover flip, zen, blindfold, coordinates,
sound, PGN copy, and help. Move, capture, check, and checkmate audio comes from
the pinned Lichess source in `deps/lichess-lila`.

## PGN analysis

Choose **Analyse PGN**, paste one game's Portable Game Notation, and select a
move in the move sheet. The board shows the position before that move. Maia3-5M
returns the exact legal-move policy conditioned on 1500 Elo for both players;
the panel displays the ten most likely human moves, the imported PGN move when
it falls outside that top ten, and the probability mass of the remaining legal
moves. The percentages therefore always total 100%.

The analysis surface is a focused Lila adaptation rather than a separate card
UI: it uses the upstream `analyse__tools` ordering, `analyse__moves` replay,
ceval engine header, explorer `table.moves` structure, evaluation formatter,
loading treatment, and explorer-row hover arrows on Chessground. The only table-specific additions are
the Maia probability visualization and the imported-PGN marker. Source mappings
are recorded in `web/lila/README.md`.

Stockfish searches the displayed Maia candidates and the played PGN move in one
depth-12 MultiPV search. Evaluations are shown from the side-to-move's point of
view. Maia is loaded lazily from `backend/.cache/maia3`, and the vendored
`deps/stockfish/stockfish-windows-x86-64-avx2.exe` starts only when analysis is
first requested. Imported analysis is separate from the live game state.

Click any Maia candidate to follow it. If it is not the imported continuation,
it is added as a Lila-style variation; click any move in the analysis tree to
jump between the PGN and its branches. You can also move either color directly
on Chessground to extend the selected line. Python reconstructs the branch from
its imported ply, validates every move and promotion, and supplies the next
position's legal destinations before Maia and Stockfish run again.

See [LICHESS_UI_AUDIT.md](LICHESS_UI_AUDIT.md) for the source-by-source feature
comparison. Lichess's round controller is intentionally not imported because
it depends on Lichess sockets, accounts, tournaments, translations, and server
state. The separable behavior and assets are adapted to the local Flask API.

## Development

Run `pnpm run build` from `web/` after editing the frontend. The generated
`web/dist/` directory contains no hand-written application logic. Python and uv
configuration remains at the repository root, while all Node, pnpm, Vite, and
TypeScript configuration lives under `web/`. Dependency sources remain separate
under `deps/`; update their pinned revisions deliberately and repeat the UI audit
when refreshing Lichess.
