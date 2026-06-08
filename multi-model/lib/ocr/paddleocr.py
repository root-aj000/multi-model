"""
PaddleOCR engine implementation.

Provides an OCR engine backed by the PaddleOCR library for extracting
text from images with confidence scores. The OCR language is read from
model_config.json (ocr.language) rather than hardcoded.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.ocr.engine import OCREngine

logger = logging.getLogger(__name__)


class PaddleOCREngine(OCREngine):
    """
    OCR engine implementation using PaddleOCR.

    Wraps the PaddleOCR engine to extract text from images and compute
    average confidence scores across all detected text regions.
    """

    def __init__(self, model_dir: Path, language: str = "en") -> None:
        """
        Initialize the PaddleOCR engine.

        Args:
            model_dir: Directory to store PaddleOCR model files.
            language:  Language code for OCR recognition.
                       Read from config (ocr.language); defaults to 'en'.

        Raises:
            ImportError: If the paddleocr package is not installed.
        """
        self.model_dir = Path(model_dir)
        self.language = language
        self.ocr = None
        self._init_engine()

    def _init_engine(self) -> None:
        """
        Import and initialize the PaddleOCR engine.

        Sets self.ocr to None if PaddleOCR is not installed, allowing
        the system to gracefully handle the missing dependency.
        """
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                det_model_dir=str(self.model_dir),
            )
            logger.info("PaddleOCR engine initialized successfully")
        except ImportError as import_error:
            logger.error(
                "PaddleOCR package is not installed. "
                "Install it with: pip install paddleocr"
            )
            raise ImportError(
                "PaddleOCR is required but not installed. "
                "Install it with: pip install paddleocr"
            ) from import_error

    def extract_text(self, image: Any) -> Tuple[str, float]:
        """
        Extract text from an image using PaddleOCR.

        Args:
            image: Input image (PIL Image, numpy array, or file path).

        Returns:
            Tuple of (extracted_text, average_confidence).
            Returns ("", 0.0) if no text is detected.

        Raises:
            RuntimeError: If PaddleOCR is not available (not installed).
        """
        if self.ocr is None:
            raise RuntimeError(
                "PaddleOCR is not available. "
                "Ensure the paddleocr package is installed."
            )

        result = self.ocr.ocr(image, cls=True)

        if not result or not result[0]:
            logger.debug("No text detected in image")
            return "", 0.0

        texts = [line[1][0] for line in result[0]]
        confidences = [line[1][1] for line in result[0]]

        average_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        combined_text = " ".join(texts)
        logger.debug(
            "Extracted %d text regions, average confidence: %.3f",
            len(texts), average_confidence,
        )
        return combined_text, average_confidence

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current status of the PaddleOCR engine.

        Returns:
            Dictionary with engine name and initialization state.
        """
        return {
            "engine": "paddleocr",
            "initialized": self.ocr is not None,
            "model_dir": str(self.model_dir),
        }

    def clear_cache(self, confirm: bool = False) -> bool:
        """
        Clear the PaddleOCR model cache directory.

        Args:
            confirm: Must be True to actually clear the cache.

        Returns:
            True if the cache was cleared, False otherwise.
        """
        if not confirm:
            return False

        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
            logger.info("Cleared PaddleOCR model cache at: %s", self.model_dir)
        self.ocr = None
        return True
