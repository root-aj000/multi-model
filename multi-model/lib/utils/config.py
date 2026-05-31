"""
Configuration utilities.

BUG-22 FIX: get_dataset_paths now anchors paths to the project root
(the directory containing this file's grandparent), so the code works
regardless of the current working directory.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Project root is three levels up from this file:
#   lib/utils/config.py  →  lib/utils/  →  lib/  →  project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file and return it as a dictionary.

    Args:
        config_path: Path to the JSON configuration file.
                     Relative paths are resolved from the project root.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError:        If the JSON cannot be parsed.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = _PROJECT_ROOT / config_file

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as fh:
        try:
            config = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file '{config_file}': {exc}") from exc

    logger.info("Loaded config from %s", config_file)
    return config


def get_dataset_paths(split: str, dataset_root: Optional[str] = None) -> Dict[str, Path]:
    """
    Get file paths for a dataset split, anchored to the project root or a
    custom dataset root.

    BUG-22 FIX: Previously used a bare relative path ("dataset/split"),
    which broke whenever the script was run from a directory other than
    the project root. Now always resolves relative to the project root.

    Args:
        split: Dataset split name — "train", "val", or "test".
        dataset_root: Optional base dataset path. If provided and not
            absolute, it is resolved relative to the project root.
            Supports either:
              - base dataset root containing split subdirectories
                (e.g. /path/to/dataset/)
              - direct split folder path
                (e.g. /path/to/dataset/train)

    Returns:
        Dict with keys "csv" and "images" pointing to absolute Paths.
    """
    if dataset_root:
        root = Path(dataset_root)
        if not root.is_absolute():
            root = _PROJECT_ROOT / root
        if (root / split).exists():
            root = root / split
    else:
        root = _PROJECT_ROOT / "dataset" / split

    return {
        "csv": root / f"{split}.csv",
        "images": root / "images",
    }


def get_label_maps(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Build label maps from the configuration dictionary.

    Supports two config formats:
      1. Flat  : {"label_maps": {"theme": ["Food", ...]}}
      2. Nested: {"ATTRIBUTES": {"theme": {"num_classes": 9, "labels": [...]}}}

    Args:
        cfg: The configuration dictionary.

    Returns:
        Dict mapping attribute names to ordered lists of label strings.
    """
    result: Dict[str, List[str]] = {}

    # Format 1: flat label_maps key
    for key, value in cfg.get("label_maps", {}).items():
        if isinstance(value, list):
            result[key] = value

    # Format 2: nested ATTRIBUTES with labels
    for attr_name, attr_cfg in cfg.get("ATTRIBUTES", {}).items():
        if isinstance(attr_cfg, dict) and isinstance(attr_cfg.get("labels"), list):
            result[attr_name] = attr_cfg["labels"]

    return result
