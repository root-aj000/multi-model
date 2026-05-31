"""
Tests for the dataset module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from lib.preprocessing.dataset import CustomDataset, IMAGE_FILENAME_COLUMN, load_dataset


class TestCustomDataset:
    """Tests for the CustomDataset class."""

    def test_init_raises_file_not_found_for_missing_csv(self, tmp_path):
        """CustomDataset raises FileNotFoundError when CSV does not exist."""
        csv_path = tmp_path / "nonexistent.csv"
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            CustomDataset(csv_path, image_dir)

    def test_init_raises_value_error_for_missing_image_filename_column(self, tmp_path):
        """CustomDataset raises ValueError when CSV is missing the image_filename column."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1,col2\n1,2\n")
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        with pytest.raises(ValueError, match=f"must contain an '{IMAGE_FILENAME_COLUMN}'"):
            CustomDataset(csv_path, image_dir)

    def test_len_returns_row_count(self, tmp_path):
        """__len__ returns the number of rows in the CSV."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("image_filename,label\nimg1.png,A\nimg2.png,B\nimg3.png,C\n")
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        ds = CustomDataset(csv_path, image_dir)
        assert len(ds) == 3

    @patch("lib.preprocessing.dataset.augment_image")
    def test_getitem_applies_augment_when_flag_true(self, mock_augment, tmp_path):
        """When augment=True, __getitem__ calls augment_image."""
        import numpy as np
        mock_augment.side_effect = lambda img: img

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("image_filename,label\nimg1.png,A\n")
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        # Create a real image file
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        img.save(image_dir / "img1.png")

        ds = CustomDataset(csv_path, image_dir, augment=True)
        ds[0]
        mock_augment.assert_called_once()

    def test_getitem_skips_augment_when_flag_false(self, tmp_path):
        """When augment=False, __getitem__ does not call augment_image."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("image_filename,label\nimg1.png,A\n")
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        img.save(image_dir / "img1.png")

        ds = CustomDataset(csv_path, image_dir, augment=False)
        image, labels = ds[0]
        assert isinstance(image, np.ndarray)
        assert labels == {"label": "A"}

    def test_getitem_raises_file_not_found_for_missing_image(self, tmp_path):
        """__getitem__ raises FileNotFoundError when image file is missing."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("image_filename,label\nmissing.png,A\n")
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        ds = CustomDataset(csv_path, image_dir)
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            ds[0]


class TestLoadDataset:
    """Tests for the load_dataset helper."""

    @patch("lib.preprocessing.dataset.get_dataset_paths")
    def test_load_dataset_returns_custom_dataset(self, mock_paths, tmp_path):
        """load_dataset returns a CustomDataset instance."""
        csv_path = tmp_path / "test_split" / "test_split.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("image_filename,label\nimg1.png,A\n")
        image_dir = tmp_path / "test_split" / "images"
        image_dir.mkdir(parents=True)

        mock_paths.return_value = {"csv": csv_path, "images": image_dir}
        ds = load_dataset("test_split")
        assert isinstance(ds, CustomDataset)
        assert len(ds) == 1

    def test_load_dataset_passes_augment_to_dataset(self, tmp_path):
        """load_dataset forwards the augment flag to CustomDataset."""
        csv_path = tmp_path / "train" / "train.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("image_filename,label\nimg1.png,A\n")
        image_dir = tmp_path / "train" / "images"
        image_dir.mkdir(parents=True)

        with patch("lib.preprocessing.dataset.get_dataset_paths") as mock_paths:
            mock_paths.return_value = {"csv": csv_path, "images": image_dir}
            ds = load_dataset("train", augment=True)
            assert ds.augment is True

            ds2 = load_dataset("val", augment=False)
            assert ds2.augment is False
