"""Resolve a configured Maia checkpoint or populate its official model cache."""
import subprocess

from maia3.model_registry import resolve_model_spec

from .settings import CONFIG


def ensure_model_cached() -> None:
    if CONFIG['MAIA']['CHECKPOINT'] is not None:
        if not CONFIG['MAIA']['CHECKPOINT'].is_file():
            raise FileNotFoundError(f"MAIA.CHECKPOINT in config.yaml does not exist: {CONFIG['MAIA']['CHECKPOINT']}")
        return
    CONFIG['MAIA']['CACHE_DIR'].mkdir(parents=True, exist_ok=True)
    model = resolve_model_spec(CONFIG['MAIA']['MODEL'])
    if model.checkpoint_filename and any(
        path.is_file() and path.stat().st_size > 0
        for path in CONFIG['MAIA']['CACHE_DIR'].rglob(model.checkpoint_filename)
    ):
        return
    subprocess.run([CONFIG['MAIA']['CACHE_EXECUTABLE'], '--model', CONFIG['MAIA']['MODEL'],
                    '--cache-dir', str(CONFIG['MAIA']['CACHE_DIR'])], check=True)


