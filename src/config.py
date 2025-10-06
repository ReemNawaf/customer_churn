from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "settings.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()
