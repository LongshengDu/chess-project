"""Content-addressed JSON evidence with atomic, concurrent-safe writes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import time


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Publish complete JSON without sharing temporary names between writers."""
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        # Windows can briefly deny replacement while another writer publishes.
        for attempt in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class JsonCache:
    def __init__(self, directory):
        self.directory = Path(directory)

    def get(self, key):
        path = self.directory / (identity(key) + '.json')
        if path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                pass
        return None

    def put(self, key, value):
        write_json(self.directory / (identity(key) + '.json'), value)
