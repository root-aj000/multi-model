"""
Tests for the EasyOCR engine.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.ocr.easyocr import EasyOCREngine


def test_easyocr_engine_get_status():
    """EasyOCREngine.get_status returns expected keys."""
    with patch("lib.ocr.easyocr.easyocr", create=True) as mock_easyocr_module:
        mock_reader = MagicMock()
        mock_easyocr_module.Reader.return_value = mock_reader
        engine = EasyOCREngine(Path("local/ocr"), languages=["en"])
        status = engine.get_status()
        assert status["engine"] == "easyocr"
        assert "languages" in status
        assert "initialized" in status


def test_easyocr_engine_clear_cache_without_confirm():
    """EasyOCREngine.clear_cache returns False when confirm is False."""
    with patch.dict("sys.modules", {"easyocr": MagicMock()}):
        engine = EasyOCREngine(Path("local/ocr"), languages=["en"])
        assert engine.clear_cache(confirm=False) is False


def test_easyocr_engine_extract_text_no_results():
    """EasyOCREngine.extract_text returns empty when no text detected."""
    with patch.dict("sys.modules", {"easyocr": MagicMock()}) as mods:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mods["easyocr"].Reader.return_value = mock_reader
        engine = EasyOCREngine(Path("local/ocr"), languages=["en"])
        engine.reader = mock_reader
        text, confidence = engine.extract_text("fake_image")
        assert text == ""
        assert confidence == 0.0


def test_easyocr_engine_extract_text_with_results():
    """EasyOCREngine.extract_text returns combined text and average confidence."""
    with patch.dict("sys.modules", {"easyocr": MagicMock()}) as mods:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            [("bbox1",), "hello", 0.9],
            [("bbox2",), "world", 0.7],
        ]
        mods["easyocr"].Reader.return_value = mock_reader
        engine = EasyOCREngine(Path("local/ocr"), languages=["en"])
        engine.reader = mock_reader
        text, confidence = engine.extract_text("fake_image")
        assert "hello" in text
        assert "world" in text
        assert 0.7 < confidence < 0.9
