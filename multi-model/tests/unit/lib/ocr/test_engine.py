import pytest
from lib.ocr.engine import OCREngine


def test_ocr_engine_is_abstract():
    """OCREngine cannot be instantiated directly."""
    with pytest.raises(TypeError):
        OCREngine()
