"""
Tests for the prediction pipeline builder.
"""

from unittest.mock import MagicMock, patch

import pytest

from use_cases.prediction.pipeline import build_prediction_pipeline


def test_build_prediction_pipeline_raises_on_missing_config():
    """build_prediction_pipeline raises FileNotFoundError for missing config."""
    with pytest.raises(FileNotFoundError):
        build_prediction_pipeline("/nonexistent/config.json")


def test_build_prediction_pipeline_returns_predictor():
    """build_prediction_pipeline returns a Predictor when config is valid."""
    mock_config = {
        "model": {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 32,
            "DROPOUT": 0.1,
            "FUSION_TYPE": "concat",
            "ATTRIBUTES": {
                "sentiment": {"num_classes": 3, "labels": ["pos", "neg", "neu"]},
            },
        },
        "ocr": {"engine": "easyocr", "model_dir": "local/ocr"},
    }

    with patch("use_cases.prediction.pipeline.load_config", return_value=mock_config), \
         patch("use_cases.prediction.pipeline.create_model") as mock_create, \
         patch("use_cases.prediction.pipeline.create_ocr_engine") as mock_ocr, \
         patch("use_cases.prediction.pipeline.get_label_maps", return_value={"sentiment": ["pos", "neg", "neu"]}):
        mock_create.return_value = MagicMock()
        mock_ocr.return_value = MagicMock()

        predictor = build_prediction_pipeline("fake_config.json")
        assert predictor is not None
