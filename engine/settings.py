"""Engine configuration, read directly from the project's config.yaml."""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.yaml'
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8-sig'))

if not isinstance(CONFIG, dict) or set(CONFIG) != {'MAIA', 'STOCKFISH', 'ANALYSIS', 'COACH', 'SERVER', 'FRONTEND'}:
    raise ValueError('config.yaml requires exactly six sections: MAIA, STOCKFISH, ANALYSIS, COACH, SERVER, FRONTEND.')
if any(not isinstance(section, dict) for section in CONFIG.values()):
    raise ValueError('Each config.yaml section must be a mapping.')

for _path in (('MAIA', 'CACHE_DIR'), ('MAIA', 'CHECKPOINT'), ('STOCKFISH', 'CACHE_DIR')):
    _section = CONFIG
    for _name in _path[:-1]:
        _section = _section[_name]
    _key = _path[-1]
    if _section[_key] is not None:
        _section[_key] = (CONFIG_PATH.parent / Path(_section[_key]).expanduser()).resolve()
for _path in (('MAIA', 'CACHE_EXECUTABLE'), ('STOCKFISH', 'EXECUTABLE')):
    _section = CONFIG
    for _name in _path[:-1]:
        _section = _section[_name]
    _key = _path[-1]
    if _section[_key] and (any(c in _section[_key] for c in ('/', '\\')) or _section[_key].startswith('.')):
        _section[_key] = str((CONFIG_PATH.parent / Path(_section[_key]).expanduser()).resolve())
