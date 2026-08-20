import json

from paths import CONFIG_PATH


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()

MODEL_CONFIG = CONFIG["model"]
TRAINING_CONFIG = CONFIG["training"]
GENERATION_CONFIG = CONFIG["generation"]
