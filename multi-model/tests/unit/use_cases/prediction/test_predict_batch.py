"""
Tests for the batch prediction use case.
"""

from unittest.mock import MagicMock, patch

from lib.ocr.engine import OCREngine
from use_cases.prediction.predict_batch import predict_batch


def test_predict_batch_returns_list():
    """predict_batch returns a list of prediction dictionaries."""
    mock_model = MagicMock()
    mock_ocr = MagicMock(spec=OCREngine)
    mock_ocr.extract_text.return_value = ("test text", 0.9)
    label_maps = {"sentiment": ["positive", "neutral", "negative"]}

    with patch("use_cases.prediction.predict_batch.predict_image") as mock_predict:
        mock_predict.return_value = {"sentiment": "positive", "raw_text": "test text"}
        results = predict_batch(
            images=["img1", "img2"],
            model=mock_model,
            ocr_engine=mock_ocr,
            label_maps=label_maps,
        )
        assert isinstance(results, list)
        assert len(results) == 2


def test_predict_batch_handles_errors():
    """predict_batch catches RuntimeError and returns error dict."""
    mock_model = MagicMock()
    mock_ocr = MagicMock(spec=OCREngine)
    label_maps = {"sentiment": ["positive", "neutral", "negative"]}

    with patch("use_cases.prediction.predict_batch.predict_image") as mock_predict:
        mock_predict.side_effect = RuntimeError("inference failed")
        results = predict_batch(
            images=["img1"],
            model=mock_model,
            ocr_engine=mock_ocr,
            label_maps=label_maps,
        )
        assert len(results) == 1
        assert "error" in results[0]
