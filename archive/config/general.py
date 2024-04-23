"""General utils."""
from typing import Any
import yaml


def _load_config_raw(path: str) -> Any:
    """
    Loads the raw data from the config file at `path`.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        return data
