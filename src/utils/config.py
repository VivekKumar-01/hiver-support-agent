"""
Small helper to load config/config.yaml from anywhere in the project.

Beginner note: we centralize config loading in one place so every script
reads the SAME thresholds/seeds instead of copy-pasting magic numbers.
That's what "reproducibility" means in practice.
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative_path: str) -> Path:
    """Turn a path from config.yaml (relative to project root) into an absolute Path."""
    return PROJECT_ROOT / relative_path
