# Maia Local Web Chess

A small local web chess game with a Lichess-inspired play layout. You play
White against Maia3-5M at Elo 1500. Chessground renders the board, while
`python-chess` owns the game state, notation, and rules.

## Repository layout

```text
app.py                 Flask application and chess API
web/                   TypeScript, HTML, and CSS frontend
deps/lichess-lila/     pinned Lichess source submodule (UI reference/assets)
deps/maia/             pinned Maia3 source submodule
deps/stockfish/        vendored Stockfish source and Windows executable
```

The notebook experiments and their supporting files also remain at the root.

## Install and run

Requirements: Python 3.10+, Git, Node.js 22+, `uv`, and a web browser. From the
repository root in PowerShell:

```powershell
git submodule update --init --depth 1
uv sync
uv run maia3-cache --model maia3-5m
npm install
npm run build
uv run python .\app.py
```

Open <http://127.0.0.1:5000>. Stop the server with `Ctrl+C`.

`maia3-cache` downloads the 5M checkpoint once, so the first move does not wait
for a model download. After the initial setup, rebuild the frontend only when
`web/` changes; the normal run command is:

```powershell
uv run python .\app.py
```

## Architecture

```text
Chessground (display and input only)
    | four local JSON routes
    v
Flask + ChessApi + python-chess (position, legality, outcome, history)
    | python-chess UCI support
    v
Maia3 (one persistent process)
```

The frontend sends a UCI move such as `e2e4` to `POST /api/move`. Python
validates and pushes it, asks Maia for a `nodes=1` reply, pushes that reply, and
returns a complete display state. The frontend converts the returned
destination object to Chessground's required `Map`; it does not calculate
moves. Promotion choices are also supplied by Python. Stopping the server runs
a `finally` block that sends the engine its UCI quit command.

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

See [LICHESS_UI_AUDIT.md](LICHESS_UI_AUDIT.md) for the source-by-source feature
comparison. Lichess's round controller is intentionally not imported because
it depends on Lichess sockets, accounts, tournaments, translations, and server
state. The separable behavior and assets are adapted to the local Flask API.

## Development

Run `npm run build` after editing `web/`. The generated `web/dist/` directory
contains no hand-written application logic. Dependency sources live under
`deps/`; update their pinned revisions deliberately and repeat the UI audit when
refreshing Lichess.
