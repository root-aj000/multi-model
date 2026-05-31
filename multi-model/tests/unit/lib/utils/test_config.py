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


def test_get_dataset_paths_with_custom_root(tmp_path):
    """get_dataset_paths resolves a custom dataset root to CSV and images."""
    custom_root = tmp_path / "ads_dataset"
    split_dir = custom_root / "train"
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    (split_dir / "train.csv").write_text("image_path,label\n")

    paths = get_dataset_paths("train", str(custom_root))
    assert paths["csv"] == split_dir / "train.csv"
    assert paths["images"] == images_dir

    paths_direct = get_dataset_paths("train", str(split_dir))
    assert paths_direct["csv"] == split_dir / "train.csv"
    assert paths_direct["images"] == images_dir


def test_get_label_maps_empty():
    """get_label_maps returns empty dict for config without label_maps."""
    result = get_label_maps({})
    assert result == {}
