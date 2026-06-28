"""
Feature extraction service.

BUG-18 FIX: Replaced assert statements with explicit ValueError raises so
validation is not silently disabled when Python runs with -O (optimize flag).
"""

import logging
from typing import Any, Dict

import torch

from lib.models.text import TextModule
from lib.models.visual import VisualModule
from lib.ocr.engine import OCREngine
from lib.preprocessing.text.cleaner import clean_text
from lib.preprocessing.text.tokenizer import tokenize_text

logger = logging.getLogger(__name__)

# Default maximum token length — should match text_max_length in config
DEFAULT_MAX_TOKEN_LENGTH = 256


class FeatureExtractor:
    """
    Orchestrates visual and text feature extraction from an image.

    Uses an OCR engine to extract text, a cleaner to normalize it,
    a tokenizer to encode it, and the visual/text modules to produce
    feature tensors.
    """

    def __init__(
        self,
        visual_module: VisualModule,
        text_module: TextModule,
        ocr_engine: OCREngine,
        max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
    ) -> None:
        """
        Initialize the FeatureExtractor.

        Args:
            visual_module:    VisualModule instance.
            text_module:      TextModule instance.
            ocr_engine:       OCREngine instance.
            max_token_length: Max token sequence length. Must match
                              text_max_length in config.
        """
        self.visual_module = visual_module
        self.text_module = text_module
        self.ocr_engine = ocr_engine
        self.max_token_length = max_token_length

    def extract(self, image: Any) -> Dict[str, Any]:
        """
        Extract visual and text features from an image.

        Args:
            image: PIL Image or numpy array.

        Returns:
            Dict with keys:
                visual_features : Tensor from the visual module.
                text_features   : Tensor from the text module.
                raw_text        : Raw OCR output string.
                confidence      : Average OCR confidence score.
        """
        visual_features = self.visual_module(image)

        raw_text, confidence = self.ocr_engine.extract_text(image)
        cleaned_text = clean_text(raw_text)

        if cleaned_text:
            # tokenize_text now handles empty strings gracefully (BUG-08 FIX),
            # but we still guard here to avoid unnecessary tokenizer calls.
            text_encoding = tokenize_text(
                cleaned_text, max_length=self.max_token_length,
            )
            input_ids = text_encoding["input_ids"].unsqueeze(0)       # (1, seq_len)
            attention_mask = text_encoding["attention_mask"].unsqueeze(0)

            # BUG-18 FIX: use explicit ValueError instead of assert
            if input_ids.dim() != 2 or input_ids.size(0) != 1:
                raise ValueError(
                    f"Expected input_ids shape (1, seq_len), got {input_ids.shape}"
                )
            if attention_mask.dim() != 2 or attention_mask.size(0) != 1:
                raise ValueError(
                    f"Expected attention_mask shape (1, seq_len), got {attention_mask.shape}"
                )

            text_features = self.text_module(input_ids, attention_mask)
        else:
            logger.warning("No text extracted from image; using zero features")
            try:
                device = next(self.text_module.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            text_features = torch.zeros(1, self.text_module.out_features, device=device)

        return {
            "visual_features": visual_features,
            "text_features": text_features,
            "raw_text": raw_text,
            "confidence": confidence,
        }
