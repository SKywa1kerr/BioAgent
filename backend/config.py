import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config = None

def get_config() -> dict:
    global _config
    if _config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config
