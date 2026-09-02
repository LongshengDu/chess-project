# Maia Local Web Chess

A deliberately small local web chess game with a Lichess-inspired play layout: you play White against Maia3-5M at Elo 1500. Chessground renders the board, while `python-chess` owns all game state, notation, and rules.

## Install and run

Requirements: Python 3.10+, Git, Node.js 22+, and a web browser. Initialize the
pinned Lichess source submodule once after cloning:

```powershell
git submodule update --init --depth 1
```

Python dependencies and the virtual environment are managed by the root
`pyproject.toml` and `uv.lock`. From the repository root in PowerShell:

```powershell
uv sync
uv run maia3-cache --model maia3-5m
Set-Location .\maia
npm install
npm run build
Set-Location ..
uv run python .\maia\app.py
```

Open <http://127.0.0.1:5000> in your browser. Stop the server with `Ctrl+C`.

`maia3-cache` downloads the 5M checkpoint once so the first move does not wait for a model download. After installation and the frontend build, the only run command is:

```powershell
uv run python .\maia\app.py
```

The same `uv sync` and `uv run` commands work on macOS and Linux; use the
platform's normal shell commands when changing directories.

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

The frontend sends one UCI move such as `e2e4` to `POST /api/move`. Python validates and pushes it, asks Maia for a `nodes=1` reply, pushes that reply, and returns a complete display state. The frontend converts the returned destination object to Chessground's required `Map`; it does not calculate moves. Promotion choices are also supplied by Python. Stopping the server runs a `finally` block that sends the engine its UCI quit command.

Python also returns a display snapshot for every ply, including the relative
captured material and material score. The move sheet, navigation buttons,
mouse wheel, and Lichess keyboard shortcuts use those snapshots to replay the
game without implementing chess rules in TypeScript. Chessground provides
right-drag arrows and circles, and promotion uses Lichess's in-board
queen/knight/rook/bishop chooser.

The game controls provide local takeback, draw claim, resign confirmation, and
New Game. Automatic draws and claimable threefold/fifty-move draws follow
`python-chess`. Board-menu preferences cover flip, zen, blindfold, coordinates,
sound, PGN copy, and help. The move/capture/check/checkmate audio files come
directly from the pinned Lichess source at `vendor/lila`.

See [LICHESS_UI_AUDIT.md](LICHESS_UI_AUDIT.md) for the source-by-source feature
comparison. Lichess's round controller itself is intentionally not imported:
it depends on Lichess sockets, accounts, tournaments, translations, and server
state. The separable round behavior and assets are adapted to the local
Flask/python-chess API instead.

## Development

Rebuild after editing `web/`:

```powershell
npm run build
```

The generated `web/dist/` directory and `package-lock.json` contain no
hand-written application logic. To refresh the pinned upstream reference later,
update the `maia/vendor/lila` submodule deliberately and repeat the UI audit.
