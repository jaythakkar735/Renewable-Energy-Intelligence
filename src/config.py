"""Configuration loader and schema validator for Renewable Energy Intelligence."""
import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Loads and validates the project YAML configuration.

    Args:
        config_path: Path to configuration YAML. If None, defaults to config/config.yaml.

    Returns:
        Dictionary containing configuration parameters.
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ensure required sections exist
    required_sections = ["paths", "data", "splits", "forecasting", "site", "tariffs", "battery"]
    for sec in required_sections:
        if sec not in cfg:
            raise KeyError(f"Missing required configuration section: '{sec}'")

    return cfg
