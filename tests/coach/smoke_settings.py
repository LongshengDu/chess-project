"""Controls for the isolated coach tests, read from their own YAML file."""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).with_name('config.yaml')
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8-sig'))
CONFIG['OUTPUT_DIR'] = (CONFIG_PATH.parent / Path(CONFIG['OUTPUT_DIR']).expanduser()).resolve()
globals().update(CONFIG)
