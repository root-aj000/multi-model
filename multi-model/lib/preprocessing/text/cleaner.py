"""
Text cleaning utilities.

Two cleaners are provided:

  clean_text(text)     — aggressive OCR noise cleaner (original).
                         Use for raw OCR output with control chars and
                         garbage symbols.

  clean_adcopy(text)   — light cleaner for structured ad copy / CSV text.
                         Preserves apostrophes, mixed case (lowercased for
                         the uncased tokenizer), and standard punctuation.
                         Only removes genuine control characters and
                         non-printable bytes that would confuse the tokenizer.

BUG-20 FIX: Added a second strip() call after noise pattern removal so
the returned string never starts or ends with whitespace.
"""

import re
import unicodedata
from typing import Optional

# ── OCR noise cleaner (original) ─────────────────────────────────────────
# Preserve: letters, digits, whitespace, currency ($€£₹¥), percent (%),
# common punctuation (.,!?-&/), and the @ symbol.
# Remove: everything else (OCR noise, control chars, unusual symbols).
_NOISE_PATTERN = re.compile(r"[^a-z0-9\s$€£₹¥%.,!?\-&/@]")

# Collapse multiple consecutive whitespace characters into a single space
_MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+")

# ── Ad-copy cleaner ───────────────────────────────────────────────────────
# Only strip genuine control characters (ASCII 0x00–0x1F except \t and \n)
# that would produce garbage token IDs. Everything else is preserved.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def clean_text(text: Optional[str]) -> str:
    """
    Aggressive OCR noise cleaner.

    Strips everything except alphanumerics, whitespace, currency symbols,
    percent, and basic punctuation. Designed for raw OCR output that may
    contain garbage characters, control bytes, and encoding artifacts.

    BUG-20 FIX: Second strip() after noise removal prevents leading/trailing
    whitespace when a noise char was at the string boundary.

    Args:
        text: Raw OCR text, or None.

    Returns:
        Cleaned, lowercased text with normalized whitespace.

    Raises:
        TypeError: If text is not a string or None.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string or None, got {type(text).__name__}"
        )
    cleaned = text.lower().strip()
    cleaned = _NOISE_PATTERN.sub("", cleaned)
    cleaned = _MULTIPLE_WHITESPACE_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.strip()  # BUG-20 FIX
    return cleaned


def clean_adcopy(text: Optional[str]) -> str:
    """
    Light cleaner for structured ad copy or human-written CSV text.

    Unlike clean_text(), this preserves:
      - Apostrophes  ("don't" stays "don't", not "dont")
      - Standard punctuation (colons, brackets, quotes, hashtags, emojis)
      - Brand capitalisation (lowercased only, nothing stripped)

    Only removes genuine control characters (0x00-0x1F except \\t/\\n)
    that would produce garbage token IDs in the BERT tokenizer.

    Use this when the CSV 'text' column contains clean ad copy, headlines,
    slogans or CTAs — not raw OCR output.

    Args:
        text: Ad copy string, or None.

    Returns:
        Lowercased text with control characters removed and whitespace
        normalised. Safe to pass directly to any HuggingFace tokenizer.

    Raises:
        TypeError: If text is not a string or None.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string or None, got {type(text).__name__}"
        )
    # NFC normalisation: accented chars → single code points
    cleaned = unicodedata.normalize("NFC", text)
    # Lowercase for -uncased BERT models
    cleaned = cleaned.lower()
    # Remove only control characters — preserve punctuation, emoji, etc.
    cleaned = _CONTROL_CHAR_PATTERN.sub("", cleaned)
    # Normalise whitespace (tabs, newlines, multiple spaces → one space)
    cleaned = _MULTIPLE_WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()
