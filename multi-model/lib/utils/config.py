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


# def get_dataset_paths(split: str, dataset_root: Optional[str] = None) -> Dict[str, Path]:
#     """
#     Get file paths for a dataset split, anchored to the project root or a
#     custom dataset root.

#     BUG-22 FIX: Previously used a bare relative path ("dataset/split"),
#     which broke whenever the script was run from a directory other than
#     the project root. Now always resolves relative to the project root.

#     Args:
#         split: Dataset split name — "train", "val", or "test".
#         dataset_root: Optional base dataset path. If provided and not
#             absolute, it is resolved relative to the project root.
#             Supports either:
#               - base dataset root containing split subdirectories
#                 (e.g. /path/to/dataset/)
#               - direct split folder path
#                 (e.g. /path/to/dataset/train)

#     Returns:
#         Dict with keys "csv" and "images" pointing to absolute Paths.
#     """
#     if dataset_root:
#         root = Path(dataset_root)
#         if not root.is_absolute():
#             root = _PROJECT_ROOT / root
#         if (root / split).exists():
#             root = root / split
#     else:
#         root = _PROJECT_ROOT / "dataset" / split

#     return {
#         "csv": root / f"{split}.csv",
#         "images": root / "images",
#     }


import os
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # adjust depth as needed

# print(f"DEBUG: Resolved project root: {_PROJECT_ROOT}")
def get_dataset_paths(split: str, dataset_root: Optional[str] = None) -> Dict[str, Path]:
    """
    Get file paths for a dataset split, anchored to the project root or a
    custom dataset root.

    BUG-22 FIX: Previously used a bare relative path ("dataset/split"),
    which broke whenever the script was run from a directory other than
    the project root. Now always resolves relative to the project root.

    BUG-23 FIX: Split subdirectory existence check was silently falling
    through when the directory didn't exist at resolution time (e.g. on
    Kaggle where the input path is mounted read-only and the split folder
    IS the root). Now applies a more robust heuristic:
      1. If root already ends with the split name → use as-is.
      2. Else if root/split exists as a directory → descend into it.
      3. Else stay at root and warn (allows caller to surface a clear error).

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

    Raises:
        FileNotFoundError: If the resolved root directory does not exist.
    """
    # ------------------------------------------------------------------ #
    # 1. Resolve the base root                                             #
    # ------------------------------------------------------------------ #
    if dataset_root:
        root = Path(dataset_root)
        if not root.is_absolute():
            root = _PROJECT_ROOT / root
    else:
        root = _PROJECT_ROOT / "dataset"

    root = root.resolve()

    # ------------------------------------------------------------------ #
    # 2. Descend into the split subdirectory — with explicit fallbacks     #
    # ------------------------------------------------------------------ #
    if root.name == split:
        # Caller already passed the split folder directly — use as-is.
        logger.debug(
            "dataset_root already points to the split folder: %s", root
        )

    elif (root / split).is_dir():
        # Standard layout: <dataset_root>/<split>/
        root = root / split
        logger.debug("Descending into split subdirectory: %s", root)

    else:
        # Neither condition met — stay at root and warn.
        # load_dataset will surface a clear FileNotFoundError if the
        # CSV is truly absent.
        logger.warning(
            "Split subdirectory '%s' not found under '%s'. "
            "Treating '%s' as the split root. "
            "Expected layout: <dataset_root>/%s/%s.csv",
            split, root, root, split, split,
        )

    # ------------------------------------------------------------------ #
    # 3. Sanity-check the resolved root exists                             #
    # ------------------------------------------------------------------ #
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root directory does not exist: {root}\n"
            f"  split        : {split}\n"
            f"  dataset_root : {dataset_root!r}\n"
            "Check that the Kaggle dataset is attached and the path is correct."
        )

    # ------------------------------------------------------------------ #
    # 4. Build and return the final paths                                  #
    # ------------------------------------------------------------------ #
    paths = {
        "csv":    root / f"{split}.csv",
        "images": root / "images",
    }

    logger.debug("Resolved dataset paths for split '%s': %s", split, paths)
    return paths











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
