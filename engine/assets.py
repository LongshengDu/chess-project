"""Prepare both engine asset caches; run with ``python -m engine.assets``."""
from pathlib import Path

from .assets_maia import ensure_model_cached
from .assets_stockfish import ensure_stockfish_cached
from .settings import CONFIG


def ensure_runtime_assets() -> Path:
    ensure_model_cached()
    return ensure_stockfish_cached()


def main() -> None:
    stockfish_path = ensure_runtime_assets()
    print(f"Maia cache: {CONFIG['MAIA']['CACHE_DIR']}")
    print(f"Stockfish: {stockfish_path}")


if __name__ == "__main__":
    main()
