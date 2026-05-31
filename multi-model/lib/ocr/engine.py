from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class OCREngine(ABC):
    """
    Abstract base class for OCR engines implementations.
    """

    @abstractmethod
    def _init_engine(self) -> None:
        """
        Initialize the underlying OCR engine.

        Subclasses must implement this to set up the engine-specific
        reader or client. Should raise ImportError if the backend is
        not available.
        """
        ...

    @abstractmethod
    def extract_text(self, image: Any) -> Tuple[str, float]:
        """
        Extract text from an image.

        Args:
            image: Input image (e.g., PIL Image or numpy array).

        Returns:
            A tuple of (extracted_text, confidence).
        """
        ...

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Return the status of the OCR engine.

        Returns:
            Dictionary containing engine status information.
        """
        ...

    @abstractmethod
    def clear_cache(self, confirm: bool = False) -> bool:
        """
        Clear the OCR engine's cache.

        Args:
            confirm: Must be True to actually clear the cache.

        Returns:
            True if the cache was cleared.
        """
        ...
