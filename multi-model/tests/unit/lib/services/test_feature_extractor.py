"""
Tests for the FeatureExtractor service.
"""

from unittest.mock import MagicMock

import pytest

from lib.services.feature_extractor import FeatureExtractor


def test_feature_extractor_init():
    """FeatureExtractor stores module references."""
    visual = MagicMock()
    text = MagicMock()
    ocr = MagicMock()
    extractor = FeatureExtractor(visual, text, ocr)
    assert extractor.visual_module is visual
    assert extractor.text_module is text
    assert extractor.ocr_engine is ocr


def test_feature_extractor_extract_calls_modules():
    """FeatureExtractor.extract calls visual module, OCR, and text module."""
    visual = MagicMock()
    visual.return_value = MagicMock()

    text = MagicMock()
    text.return_value = MagicMock()

    ocr = MagicMock()
    ocr.extract_text.return_value = ("hello world", 0.95)

    extractor = FeatureExtractor(visual, text, ocr)
    result = extractor.extract("fake_image")

    visual.assert_called_once_with("fake_image")
    ocr.extract_text.assert_called_once_with("fake_image")
    assert "visual_features" in result
    assert "raw_text" in result
    assert result["raw_text"] == "hello world"
    assert result["confidence"] == 0.95
