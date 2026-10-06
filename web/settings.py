"""Frontend build configuration, read directly from the project's YAML."""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.yaml'
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8-sig'))

if not isinstance(CONFIG, dict) or set(CONFIG) != {'MAIA', 'STOCKFISH', 'ANALYSIS', 'COACH', 'SERVER', 'FRONTEND'}:
    raise ValueError('config.yaml requires exactly six sections: MAIA, STOCKFISH, ANALYSIS, COACH, SERVER, FRONTEND.')
if any(not isinstance(section, dict) for section in CONFIG.values()):
    raise ValueError('Each config.yaml section must be a mapping.')

_build = CONFIG['FRONTEND']['BUILD']
if _build['TOOLS_DIR'] is not None:
    _build['TOOLS_DIR'] = (CONFIG_PATH.parent / Path(_build['TOOLS_DIR']).expanduser()).resolve()
_node = _build['NODE_EXECUTABLE']
if _node and (any(separator in _node for separator in ('/', '\\')) or _node.startswith('.')):
    _build['NODE_EXECUTABLE'] = str((CONFIG_PATH.parent / Path(_node).expanduser()).resolve())

FRONTEND = Path(__file__).resolve().parent
