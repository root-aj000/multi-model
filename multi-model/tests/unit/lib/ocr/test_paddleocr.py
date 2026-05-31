"""
Tests for the PaddleOCR engine.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.ocr.paddleocr import PaddleOCREngine


def test_paddleocr_engine_get_status():
    """PaddleOCREngine.get_status returns expected keys."""
    with patch.dict("sys.modules", {"paddleocr": MagicMock()}):
        engine = PaddleOCREngine(Path("local/ocr"))
        status = engine.get_status()
        assert status["engine"] == "paddleocr"
        assert "initialized" in status


def test_paddleocr_engine_clear_cache_without_confirm():
    """PaddleOCREngine.clear_cache returns False when confirm is False."""
    with patch.dict("sys.modules", {"paddleocr": MagicMock()}):
        engine = PaddleOCREngine(Path("local/ocr"))
        assert engine.clear_cache(confirm=False) is False


def test_paddleocr_engine_extract_text_raises_when_not_available():
    """PaddleOCREngine.extract_text raises RuntimeError when not initialized."""
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine.model_dir = Path("local/ocr")
    engine.ocr = None
    with pytest.raises(RuntimeError, match="not available"):
        engine.extract_text("fake_image")
