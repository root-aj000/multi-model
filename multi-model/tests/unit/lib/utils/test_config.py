import pytest
from lib.utils.config import load_config, get_dataset_paths, get_label_maps


def test_load_config_file_not_found():
    """load_config raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.json")


def test_get_dataset_paths_returns_dict():
    """get_dataset_paths returns a dict with 'csv' and 'images' keys."""
    paths = get_dataset_paths("train")
    assert "csv" in paths
    assert "images" in paths


def test_get_label_maps_empty():
    """get_label_maps returns empty dict for config without label_maps."""
    result = get_label_maps({})
    assert result == {}
