"""
Text tokenization utilities.

BUG-08 FIX: tokenize_text no longer raises ValueError on empty text.
Instead it returns a zero-padded encoding so DataLoader workers never
crash on rows with missing or empty text columns.
"""

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Default transformer model name for tokenization
DEFAULT_TOKENIZER_MODEL = "distilbert-base-uncased"

# Default maximum sequence length — must match text_max_length in config
DEFAULT_MAX_LENGTH = 256

# Hard cap on raw string length before tokenization to prevent OOM
MAX_RAW_TEXT_LENGTH = 100_000

# Module-level cached default tokenizer to avoid reloading on every call
_default_tokenizer: Optional[Any] = None


def load_tokenizer(model_name: str = DEFAULT_TOKENIZER_MODEL) -> Any:
    """
    Load and return a HuggingFace tokenizer for the given model name.

    The default tokenizer (distilbert-base-uncased) is cached at module
    level so repeated calls don't reload from disk.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A HuggingFace tokenizer instance.

    Raises:
        OSError: If the model is not found locally or on HuggingFace Hub.
    """
    global _default_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Loaded tokenizer for model: %s", model_name)
    if model_name == DEFAULT_TOKENIZER_MODEL and _default_tokenizer is None:
        _default_tokenizer = tokenizer
    return tokenizer


def tokenize_text(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenizer: Any = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize text using a BERT-based tokenizer.

    BUG-08 FIX: Empty or whitespace-only text no longer raises ValueError.
    A zero-padded encoding is returned instead, which is safe for the model
    (all-zero input_ids with all-zero attention_mask → no signal, no crash).

    Args:
        text:       The string to tokenize. Empty strings are handled gracefully.
        max_length: Maximum sequence length. Should match text_max_length in config.
        tokenizer:  Pre-loaded tokenizer. If None, uses the cached default.

    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors of shape (max_length,).
    """
    # BUG-08 FIX: return zero tensors instead of raising on empty text
    if not text or not text.strip():
        logger.debug("tokenize_text received empty text; returning zero-padded encoding")
        return {
            "input_ids": torch.zeros(max_length, dtype=torch.long),
            "attention_mask": torch.zeros(max_length, dtype=torch.long),
        }

    if len(text) > MAX_RAW_TEXT_LENGTH:
        logger.warning(
            "Input text length %d exceeds limit %d; truncating",
            len(text), MAX_RAW_TEXT_LENGTH,
        )
        text = text[:MAX_RAW_TEXT_LENGTH]

    global _default_tokenizer
    if tokenizer is None:
        if _default_tokenizer is None:
            _default_tokenizer = load_tokenizer()
        tokenizer = _default_tokenizer

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }
