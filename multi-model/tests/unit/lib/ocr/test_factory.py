"""
Tests for the OCR engine factory.
"""

import pytest
from pathlib import Path

from lib.ocr.factory import create_ocr_engine, SUPPORTED_OCR_ENGINES


def test_create_ocr_engine_unsupported_raises():
    """create_ocr_engine raises ValueError for unsupported engine names."""
    with pytest.raises(ValueError, match="not supported"):
        create_ocr_engine("nonexistent_engine", Path("local/ocr"))


def test_supported_engines_contains_easyocr():
    """SUPPORTED_OCR_ENGINES contains 'easyocr'."""
    assert "easyocr" in SUPPORTED_OCR_ENGINES


def test_supported_engines_contains_paddleocr():
    """SUPPORTED_OCR_ENGINES contains 'paddleocr'."""
    assert "paddleocr" in SUPPORTED_OCR_ENGINES
