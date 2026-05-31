import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_model_cache_directory() -> Path:
    """Return the directory for cached visual models."""
    return Path("local/models")


def list_cached_models() -> List[str]:
    """List the names of cached visual models."""
    cache_dir = get_model_cache_directory()
    if not cache_dir.exists():
        return []
    return [d.name for d in cache_dir.iterdir() if d.is_dir()]


def get_cache_size() -> float:
    """Return the total size in MB of the visual model cache."""
    cache_dir = get_model_cache_directory()
    if not cache_dir.exists():
        return 0.0
    total_size = sum(
        f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
    )
    return total_size / (1024 * 1024)


def clear_model_cache(confirm: bool = False) -> bool:
    """Clear the visual model cache. Returns True if performed."""
    if not confirm:
        return False
    cache_dir = get_model_cache_directory()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    return True


def get_text_model_cache_directory() -> Path:
    """Return the directory for cached text models."""
    return Path("local/tokenizer")


def list_cached_text_models() -> List[str]:
    """List the names of cached text models."""
    cache_dir = get_text_model_cache_directory()
    if not cache_dir.exists():
        return []
    return [d.name for d in cache_dir.iterdir() if d.is_dir()]


def get_text_cache_size() -> float:
    """Return the total size in MB of the text model cache."""
    cache_dir = get_text_model_cache_directory()
    if not cache_dir.exists():
        return 0.0
    total_size = sum(
        f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
    )
    return total_size / (1024 * 1024)


def clear_text_model_cache(confirm: bool = False) -> bool:
    """Clear the text model cache. Returns True if performed."""
    if not confirm:
        return False
    cache_dir = get_text_model_cache_directory()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    return True


def get_dir_size(path: Path) -> float:
    """
    Compute the total size of a directory in bytes.

    Uses a generator to avoid loading all file entries into memory
    at once, which is important for large directory trees.

    Args:
        path: Path to the directory.

    Returns:
        Total size in bytes.
    """
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def get_cache_info() -> Dict[str, Any]:
    """
    Return a dictionary with cache information for all model types.

    Returns:
        Dictionary with cache size info.
    """
    return {
        "visual_cache_size_mb": get_cache_size(),
        "text_cache_size_mb": get_text_cache_size(),
    }


def clear_cache(engine: Optional[str] = None, confirm: bool = False) -> bool:
    """
    Clear model caches.

    Args:
        engine: Optional engine name to clear specific caches.
        confirm: Must be True to actually clear.

    Returns:
        True if any cache was cleared.
    """
    if not confirm:
        return False
    cleared = False
    if engine in ("visual", None):
        cleared |= clear_model_cache(confirm=True)
    if engine in ("text", None):
        cleared |= clear_text_model_cache(confirm=True)
    return cleared


def count_files(path: Path) -> int:
    """Count the number of files in a directory."""
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file())


class CacheManager:
    """
    Manages multiple cache directories.
    """

    def __init__(self, cache_dirs: Dict[str, Path]) -> None:
        """
        Initialize the CacheManager.

        Args:
            cache_dirs: Mapping of cache names to their directory paths.
        """
        self.cache_dirs = cache_dirs

    def info(self) -> Dict[str, Any]:
        """Return information about all managed caches."""
        return {
            name: get_dir_size(path)
            for name, path in self.cache_dirs.items()
            if path.exists()
        }
