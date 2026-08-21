import json
from pathlib import Path
from typing import Any

from paths import CONFIG_PATH


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Read the JSON config file.

    Called once at import time, so every module sees the same values for the
    whole run even if the file changes on disk.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Config file to read. Defaults to the path resolved by `paths`.

    Returns
    -------
    dict
        The parsed config, with its "model", "training" and "generation"
        sections.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()

MODEL_CONFIG = CONFIG["model"]
TRAINING_CONFIG = CONFIG["training"]
GENERATION_CONFIG = CONFIG["generation"]
