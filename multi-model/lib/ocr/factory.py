"""
OCR engine factory.

Creates OCR engine instances based on the engine name, providing a
centralized way to obtain the appropriate OCR implementation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ocr.easyocr import EasyOCREngine
from lib.ocr.engine import OCREngine
from lib.ocr.paddleocr import PaddleOCREngine

logger = logging.getLogger(__name__)

# Supported OCR engine names
SUPPORTED_OCR_ENGINES = ["easyocr", "paddleocr"]


def create_ocr_engine(
    engine_name: str,
    model_dir: Path,
    **kwargs: Any,
) -> OCREngine:
    """
    Create an OCR engine based on the engine name.

    Args:
        engine_name: The name of the OCR engine ('easyocr' or 'paddleocr').
        model_dir: Directory to store model files.
        **kwargs: Additional keyword arguments for engine initialization.
            For EasyOCR, 'languages' is accepted as a list of language codes.

    Returns:
        An instance of OCREngine.

    Raises:
        ValueError: If the engine_name is not supported.
    """
    if engine_name == "easyocr":
        languages = kwargs.get("languages")
        engine = EasyOCREngine(model_dir, languages=languages)
        logger.info("Created EasyOCR engine with model_dir=%s", model_dir)
        return engine

    if engine_name == "paddleocr":
        engine = PaddleOCREngine(model_dir)
        logger.info("Created PaddleOCR engine with model_dir=%s", model_dir)
        return engine

    raise ValueError(
        f"OCR engine '{engine_name}' is not supported. "
        f"Valid options are: {SUPPORTED_OCR_ENGINES}"
    )
