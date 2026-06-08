"""
EasyOCR engine implementation.

Provides an OCR engine backed by the EasyOCR library for extracting
text from images with confidence scores. Language defaults to ['en']
and is set from model_config.json (ocr.language) by the caller.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.ocr.engine import OCREngine

logger = logging.getLogger(__name__)

# Minimum number of results to consider a valid extraction
MINIMUM_RESULTS_FOR_CONFIDENCE = 1


class EasyOCREngine(OCREngine):
    """
    OCR engine implementation using EasyOCR.

    Wraps the EasyOCR Reader to extract text from images and compute
    average confidence scores across all detected text regions.
    """

    def __init__(
        self,
        model_dir: Path,
        languages: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the EasyOCR engine.

        Args:
            model_dir: Directory to store EasyOCR model files.
            languages: List of language codes to load.
                       Read from config (ocr.language); defaults to ['en'].

        Raises:
            TypeError: If model_dir is not a Path or languages is not a list.
        """
        self.model_dir = Path(model_dir)
        self.languages = languages if languages is not None else ["en"]
        self.reader = None
        self._init_engine()

    def _init_engine(self) -> None:
        """
        Import and initialize the EasyOCR reader.

        Raises:
            ImportError: If the easyocr package is not installed.
            RuntimeError: If the EasyOCR reader fails to initialize.
        """
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages,
                model_storage_directory=str(self.model_dir),
            )
            logger.info(
                "EasyOCR engine initialized with languages: %s",
                self.languages,
            )
        except ImportError as import_error:
            logger.error(
                "EasyOCR package is not installed. Install it with: pip install easyocr"
            )
            raise ImportError(
                "EasyOCR is required but not installed. "
                "Install it with: pip install easyocr"
            ) from import_error
        except Exception as init_error:
            self.reader = None
            logger.error("EasyOCR reader failed to initialize: %s", init_error)
            raise RuntimeError(
                f"EasyOCR reader failed to initialize: {init_error}"
            ) from init_error

    def extract_text(self, image: Any) -> Tuple[str, float]:
        """
        Extract text from an image using EasyOCR.

        Args:
            image: Input image (PIL Image, numpy array, or file path).

        Returns:
            Tuple of (extracted_text, average_confidence).
            Returns ("", 0.0) if no text is detected.
        """
        if self.reader is None:
            raise RuntimeError(
                "EasyOCR reader is not initialized. "
                "The engine may have failed to initialize."
            )

        results = self.reader.readtext(image)

        if not results:
            logger.debug("No text detected in image")
            return "", 0.0

        texts = [result[1] for result in results]
        confidences = [result[2] for result in results]

        average_confidence = (
            sum(confidences) / len(confidences)
            if len(confidences) >= MINIMUM_RESULTS_FOR_CONFIDENCE
            else 0.0
        )

        combined_text = " ".join(texts)
        logger.debug(
            "Extracted %d text regions, average confidence: %.3f",
            len(texts), average_confidence,
        )
        return combined_text, average_confidence

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current status of the EasyOCR engine.

        Returns:
            Dictionary with engine name, languages, and initialization state.
        """
        return {
            "engine": "easyocr",
            "languages": self.languages,
            "initialized": self.reader is not None,
            "model_dir": str(self.model_dir),
        }

    def clear_cache(self, confirm: bool = False) -> bool:
        """
        Clear the EasyOCR model cache directory.

        Args:
            confirm: Must be True to actually clear the cache.

        Returns:
            True if the cache was cleared, False otherwise.
        """
        if not confirm:
            return False

        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
            logger.info("Cleared EasyOCR model cache at: %s", self.model_dir)
        self.reader = None
        return True
