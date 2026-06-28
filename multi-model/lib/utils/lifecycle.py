import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def setup_directories(dirs: Optional[List[Path]] = None) -> None:
    """
    Ensure all required directories exist.

    Args:
        dirs: List of directory paths to create.  Defaults to standard set.
    """
    if dirs is None:
        dirs = [
            Path("uploads"),
            Path("saved_models"),
            Path("local/models"),
            Path("local/ocr"),
            Path("local/cache"),
        ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def cleanup_upload_directory(upload_dir: Path) -> None:
    """
    Remove all files from the upload directory.

    Args:
        upload_dir: Path to the upload directory.
    """
    if not upload_dir.exists():
        return
    resolved_root = upload_dir.resolve()
    for item in upload_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            # Prevent symlink traversal outside upload_dir
            if item.resolve().is_relative_to(resolved_root):
                shutil.rmtree(item)
            else:
                logger.warning("Skipping symlink to outside upload dir: %s", item)
