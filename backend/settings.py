from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIST_DIR = PROJECT_ROOT / "web" / "dist"

MAIA_ELO = 1500
MAIA_MODEL_NAME = "maia3-5m"
MAIA_CACHE_DIR = PROJECT_ROOT / "backend" / ".cache" / "maia3"
MAIA_ENGINE_COMMAND = [
    "maia3-uci",
    "--model",
    MAIA_MODEL_NAME,
    "--cache-dir",
    str(MAIA_CACHE_DIR),
    "--local-files-only",
    "--use-uci-history",
    "--elo",
    str(MAIA_ELO),
    "--temperature",
    "0.3",
]
MAIA_CACHE_COMMAND = [
    "maia3-cache",
    "--model",
    MAIA_MODEL_NAME,
    "--cache-dir",
    str(MAIA_CACHE_DIR),
]

TOP_HUMAN_MOVES = 10
STOCKFISH_DEPTH = 12
STOCKFISH_THREADS = 8
STOCKFISH_HASH_MB = 512
STOCKFISH_CACHE_DIR = PROJECT_ROOT / "backend" / ".cache" / "stockfish"
STOCKFISH_RELEASE_API = (
    "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
)
STOCKFISH_ASSETS = {
    ("windows", "x86-64"): "stockfish-windows-x86-64-avx2.zip",
    ("windows", "arm64"): "stockfish-windows-armv8.zip",
    ("linux", "x86-64"): "stockfish-ubuntu-x86-64-avx2.tar",
    ("darwin", "x86-64"): "stockfish-macos-x86-64-avx2.tar",
    ("darwin", "arm64"): "stockfish-macos-m1-apple-silicon.tar",
}
