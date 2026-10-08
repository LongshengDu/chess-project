"""Content identities for local engine assets without loading or starting them."""
from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=32)
def _digest(path, size, modified, changed):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            digest.update(block)
    return 'sha256:' + digest.hexdigest()


def asset_identity(path):
    """Moving/touching identical bytes retains identity; changed bytes invalidate it."""
    path = Path(path).resolve()
    before = path.stat()
    value = _digest(str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError(f'Engine asset changed while identifying it: {path.name}')
    return value
