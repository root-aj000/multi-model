"""
Dataset utilities for loading and processing image-text datasets.

Provides a custom PyTorch Dataset that reads image paths and labels from
a CSV file, and helper functions to build datasets from configuration.

Label encoding is handled internally: string label values are converted
to integer class indices using the label_maps derived from the model config.
Columns that are not in label_maps (e.g. text, keywords, monetary_mention,
call_to_action, object_detected) are excluded from the returned label dict
so the training loop only receives integer-encoded targets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Column name in the CSV that holds the image file path
IMAGE_FILENAME_COLUMN = "image_path"

# Columns that are never used as classification targets
NON_LABEL_COLUMNS = {
    IMAGE_FILENAME_COLUMN,
    "index",
    "text",
    "keywords",
    "call_to_action",
    "monetary_mention",
    "object_detected",
}

# Handcrafted auxiliary feature categories (from the CSV metadata columns)
AUX_KEYWORDS = ["sale", "discount", "offer", "deal", "free", "new", "limited", "exclusive", "special", "buy"]
AUX_CTAS = ["buy now", "click here", "learn more", "shop now", "get yours"]
AUX_OBJECTS = ["phone", "laptop", "car", "dress", "shoes", "watch"]

AUX_FEATURE_COLUMNS = ["keywords", "call_to_action", "monetary_mention", "object_detected"]

AUX_FEATURE_DIM = len(AUX_KEYWORDS) + len(AUX_CTAS) + 1 + len(AUX_OBJECTS)


def _csv_value_to_str(val) -> str:
    """Return a lowercased, stripped string from a CSV cell (handles NaN/float)."""
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def encode_aux_features(row: pd.Series) -> torch.Tensor:
    """
    Encode handcrafted auxiliary features from a CSV row into a fixed-size
    binary vector.

    Feature layout:
      [0:10]   keywords (sale, discount, offer, deal, free, new, limited, exclusive, special, buy)
      [10:15]  call_to_action (buy now, click here, learn more, shop now, get yours)
      [15]     monetary_mention (any price mention)
      [16:22]  object_detected (phone, laptop, car, dress, shoes, watch)

    Returns:
        (AUX_FEATURE_DIM,) float32 tensor.
    """
    features = []

    # Keywords (10 binary flags) — comma-separated list
    kw_str = _csv_value_to_str(row.get("keywords"))
    kw_tokens = {word.strip() for phrase in kw_str.split(",") for word in phrase.split()}
    features.extend([1.0 if kw in kw_tokens else 0.0 for kw in AUX_KEYWORDS])

    # Call-to-action (5 binary flags) — exact match against known phrases
    cta_str = _csv_value_to_str(row.get("call_to_action"))
    features.extend([1.0 if cta == cta_str else 0.0 for cta in AUX_CTAS])

    # Monetary mention (1 binary flag)
    monetary_str = _csv_value_to_str(row.get("monetary_mention"))
    features.append(1.0 if monetary_str else 0.0)

    # Object detected (6 binary flags) — comma-separated list
    obj_str = _csv_value_to_str(row.get("object_detected"))
    obj_tokens = {o.strip().lower() for o in obj_str.split(",") if o.strip()}
    features.extend([1.0 if obj in obj_tokens else 0.0 for obj in AUX_OBJECTS])

    return torch.tensor(features, dtype=torch.float32)


class CustomDataset(Dataset):
    """
    Custom dataset that loads image paths and labels from a CSV file.

    Each row in the CSV is expected to have an 'image_path' column and
    one or more label columns whose string values are encoded to integer
    class indices using the provided label_maps.
    """

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        label_maps: Dict[str, List[str]],
        image_pipeline=None,
        text_pipeline=None,
        aux_feature_columns: Optional[List[str]] = None,
        return_index: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            csv_path: Path to the CSV file with image paths and labels.
            image_dir: Directory where images are stored.
            label_maps: Mapping of attribute name -> ordered list of label
                strings. Used to encode string labels to integer indices.
            image_pipeline: Callable that transforms a numpy image array to
                a tensor. If None, the raw numpy array is returned.
            text_pipeline: Callable that tokenizes a text string and returns
                a dict with 'input_ids' and 'attention_mask' tensors.
                If None, text features are not returned.
            aux_feature_columns: Optional list of CSV column names from which
                handcrafted auxiliary features are extracted. These are encoded
                into a fixed-size binary vector and returned alongside the
                standard outputs. See encode_aux_features().
            return_index: If True, the sample index is returned as the last
                element of the tuple. Used by ELR (Early Learning Regularization)
                to track per-sample prediction EMA targets.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV is missing the image_path column.
        """
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found at path: {csv_path}"
            )

        self.csv_path = csv_path
        self.image_dir = image_dir
        self.label_maps = label_maps
        self.image_pipeline = image_pipeline
        self.text_pipeline = text_pipeline
        self.aux_feature_columns = aux_feature_columns
        self.return_index = return_index
        self.data = pd.read_csv(csv_path)

        if IMAGE_FILENAME_COLUMN not in self.data.columns:
            raise ValueError(
                f"CSV file must contain an '{IMAGE_FILENAME_COLUMN}' column. "
                f"Found columns: {list(self.data.columns)}"
            )

        # Determine which columns are label targets (present in label_maps)
        self.label_columns: List[str] = [
            col for col in self.data.columns
            if col in self.label_maps
        ]

        # Precompute label→index mappings for O(1) lookup instead of O(n) .index()
        self._label_indices: Dict[str, Dict[str, int]] = {}
        for col in self.label_columns:
            self._label_indices[col] = {
                label: idx for idx, label in enumerate(self.label_maps[col])
            }

        # Cache whether text column exists
        self._has_text_column = "text" in self.data.columns

        logger.info(
            "Loaded dataset from %s with %d samples. Label columns: %s",
            csv_path, len(self.data), self.label_columns,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Load and return a single sample by index.

        Args:
            idx: Index of the sample to load.

        Returns:
            When text_pipeline is set:
                (image_tensor, label_dict, input_ids, attention_mask)
            Otherwise:
                (image_tensor, label_dict)

            label_dict maps attribute name -> integer class index (torch.long).

        Raises:
            FileNotFoundError: If the image file does not exist on disk.
            ValueError: If a label value is not found in its label map.
        """
        row = self.data.iloc[idx]
        image_path = self.image_dir / row[IMAGE_FILENAME_COLUMN]

        image = self._load_image(image_path)

        if self.image_pipeline is not None:
            image = self.image_pipeline(image)

        # Encode string labels to integer indices using precomputed mappings
        label_dict: Dict[str, torch.Tensor] = {}
        for col in self.label_columns:
            raw_value = row[col]
            label_index_map = self._label_indices[col]
            if raw_value not in label_index_map:
                raise ValueError(
                    f"Label value '{raw_value}' for attribute '{col}' not found "
                    f"in label map. Valid values: {list(label_index_map.keys())}"
                )
            label_dict[col] = torch.tensor(
                label_index_map[raw_value], dtype=torch.long
            )

        # Compute auxiliary handcrafted features if configured
        aux_features = None
        if self.aux_feature_columns is not None:
            aux_features = encode_aux_features(row)

        # Build return tuple
        result: Tuple = (image, label_dict)
        if self.text_pipeline is not None and self._has_text_column:
            raw_text = str(row["text"]) if pd.notna(row["text"]) else ""
            text_features = self.text_pipeline(raw_text)
            result = result + (text_features["input_ids"], text_features["attention_mask"])
        if aux_features is not None:
            result = result + (aux_features,)
        if self.return_index:
            result = result + (idx,)
        return result

    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load an image from disk as a NumPy array.

        Args:
            image_path: Path to the image file.

        Returns:
            Image as a NumPy array with shape (H, W, C) in RGB.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found at path: {image_path}"
            )

        pil_image = Image.open(image_path).convert("RGB")
        return np.array(pil_image)


from lib.utils.config import get_dataset_paths, get_label_maps  # noqa: E402


def load_dataset(
    split: str,
    label_maps: Dict[str, List[str]],
    image_pipeline=None,
    text_pipeline=None,
    dataset_root: Optional[str] = None,
    aux_feature_columns: Optional[List[str]] = None,
    return_index: bool = False,
) -> CustomDataset:
    """
    Load a dataset for the given split.

    Expects either the default directory structure:
        dataset/{split}/{split}.csv
        dataset/{split}/images/
    or a custom dataset root provided through config.

    Args:
        split: One of 'train', 'val', 'test'.
        label_maps: Mapping of attribute name -> ordered list of label strings,
            used to encode CSV string labels to integer class indices.
        image_pipeline: Optional image transform callable.
        text_pipeline: Optional text transform callable.
        dataset_root: Optional base dataset path. If provided and not absolute,
            it is resolved relative to the project root.
        aux_feature_columns: Optional list of CSV column names for handcrafted
            auxiliary feature extraction.
        return_index: If True, the dataset returns sample indices (for ELR).

    Returns:
        CustomDataset instance.

    Raises:
        FileNotFoundError: If CSV or image directory does not exist.
    """
    paths = get_dataset_paths(split, dataset_root)
    return CustomDataset(
        csv_path=paths["csv"],
        image_dir=paths["images"],
        label_maps=label_maps,
        image_pipeline=image_pipeline,
        text_pipeline=text_pipeline,
        aux_feature_columns=aux_feature_columns,
        return_index=return_index,
    )
