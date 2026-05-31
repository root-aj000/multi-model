import os
import shutil
from pathlib import Path
from typing import List, Optional


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
    for item in upload_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
