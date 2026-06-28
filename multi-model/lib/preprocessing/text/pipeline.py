"""
Text processing pipeline for extracting structured information from OCR text.

Provides functions to extract keywords, monetary mentions, call-to-action
phrases, and product objects from cleaned text, as well as a pipeline
builder that composes a cleaner and tokenizer into a single callable.
"""

import logging
import re
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Common marketing keywords to detect in OCR text
COMMON_KEYWORDS = [
    "sale", "discount", "offer", "deal", "free",
    "new", "limited", "exclusive", "special", "buy",
]

# Common call-to-action phrases
CTA_PHRASES = [
    "buy now", "click here", "learn more", "shop now", "get yours",
]

# Common product category keywords
PRODUCT_KEYWORDS = [
    "phone", "laptop", "car", "dress", "shoes", "watch",
]

# Maximum input length for regex-based extraction to prevent excessive backtracking
_MAX_REGEX_INPUT_LENGTH = 10_000

# Regex pattern for monetary mentions ($X.XX or ₹X.XX)
_MONETARY_PATTERN = re.compile(
    r"\$[0-9,]+(?:\.[0-9]{2})?|₹[0-9,]+(?:\.[0-9]{2})?"
)


def extract_keywords(text: str) -> List[str]:
    """
    Extract marketing keywords from the provided text.

    Args:
        text: The cleaned text to analyze.

    Returns:
        List of identified keywords found in the text.
    """
    text_lower = text.lower()
    return [word for word in COMMON_KEYWORDS if word in text_lower]


def extract_monetary_mention(text: str) -> Optional[str]:
    """
    Extract the first monetary mention from text.

    Detects dollar and rupee amounts (e.g., $9.99, ₹1,500.00).

    Args:
        text: The cleaned text to analyze.

    Returns:
        The first found price or money mention, or None if not found.
    """
    if len(text) > _MAX_REGEX_INPUT_LENGTH:
        logger.warning(
            "Input text length %d exceeds limit %d; truncating for regex extraction",
            len(text), _MAX_REGEX_INPUT_LENGTH,
        )
        text = text[:_MAX_REGEX_INPUT_LENGTH]
    match = _MONETARY_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def extract_call_to_action(text: str) -> Optional[str]:
    """
    Detect a call-to-action phrase from the text.

    Args:
        text: The cleaned text to analyze.

    Returns:
        The identified call-to-action string, or None if not found.
    """
    text_lower = text.lower()
    for phrase in CTA_PHRASES:
        if phrase in text_lower:
            return phrase
    return None


def extract_objects_mentioned(text: str) -> List[str]:
    """
    Extract product category mentions from the text.

    Args:
        text: The cleaned text to analyze.

    Returns:
        List of identified product categories found in the text.
    """
    text_lower = text.lower()
    return [item for item in PRODUCT_KEYWORDS if item in text_lower]


def build_text_pipeline(
    cleaner: Callable[[str], str],
    tokenizer: Callable[[str, int], dict],
) -> Callable:
    """
    Build a text processing pipeline from a cleaner and a tokenizer.

    The returned pipeline first cleans raw text, then tokenizes the
    cleaned result.

    Args:
        cleaner: Function that cleans raw text and returns a string.
        tokenizer: Function that tokenizes cleaned text and returns a dict.

    Returns:
        A callable that processes raw text into tokenized output.
    """
    def pipeline(text: str, max_length: int = 256) -> dict:
        cleaned = cleaner(text)
        return tokenizer(cleaned, max_length)

    return pipeline
