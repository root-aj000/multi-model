from pathlib import Path
from lib.utils.cache import CacheManager, get_dir_size, count_files, get_cache_info


def test_get_dir_size_nonexistent():
    """get_dir_size returns 0 for non-existent directory."""
    result = get_dir_size(Path("/nonexistent"))
    assert result == 0


def test_count_files_nonexistent():
    """count_files returns 0 for non-existent directory."""
    result = count_files(Path("/nonexistent"))
    assert result == 0


def test_cache_manager_info():
    """CacheManager.info returns a dict."""
    cm = CacheManager({"test": Path("/nonexistent")})
    info = cm.info()
    assert isinstance(info, dict)
