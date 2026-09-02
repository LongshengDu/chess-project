# Chess project

The repository is one `uv` project. Its Python version, dependencies, lockfile,
and virtual environment are managed from the repository root:

```powershell
uv sync
```

Run the Flask game documented in [game/README.md](game/README.md):

```powershell
uv run python .\game\app.py
```

Run the Maia local web game documented in [maia/README.md](maia/README.md), then open <http://127.0.0.1:5000>:

```powershell
uv run python .\maia\app.py
```

The notebook experiments remain at the repository root. `pyproject.toml`,
`uv.lock`, `.python-version`, and `.venv` also live only at the root so both
applications use the same locked Python environment.
