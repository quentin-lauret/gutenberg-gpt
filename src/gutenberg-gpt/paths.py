from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", ROOT / "config.json"))
ARTIFACTS_DIRECTORY = Path(os.getenv("ARTIFACTS_DIRECTORY", ROOT / "artifacts"))
DATA_DIRECTORY = Path(os.getenv("DATA_DIRECTORY", ROOT / "data" / "books_clean"))
BIN_DATA_DIRECTORY = Path(os.getenv("BIN_DATA_DIRECTORY", ROOT / "data" / "bin"))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", ARTIFACTS_DIRECTORY / "gutenberg-fr-tokenizer.json"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACTS_DIRECTORY / "gutenberg-gpt.pt"))