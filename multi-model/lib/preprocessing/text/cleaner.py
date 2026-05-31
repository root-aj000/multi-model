"""
Text cleaning utilities for OCR output.

BUG-20 FIX: Added a second strip() call after noise pattern removal so
the returned string never starts or ends with whitespace.
"""

import re
from typing import Optional

# Preserve: letters, digits, whitespace, currency ($€£₹¥), percent (%),
# common punctuation (.,!?-&/), and the @ symbol.
# Remove: everything else (OCR noise, control chars, unusual symbols).
_NOISE_PATTERN = re.compile(r"[^a-z0-9\s$€£₹¥%.,!?\-&/@]")

# Collapse multiple consecutive whitespace characters into one space
_MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """
    Clean OCR text by removing noise artifacts and normalizing whitespace.

    Processing order:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Remove noise characters (keep currency, punctuation, alphanumerics)
      4. Collapse multiple spaces into one
      5. Strip again — noise removal can expose leading/trailing spaces

    BUG-20 FIX: Step 5 (second strip) was missing. Without it, removing a
    noise character at the start or end of the string left a leading/trailing
    space in the output.

    Preserved characters beyond alphanumerics:
      Currency : $ € £ ₹ ¥
      Percent  : %
      Punctuation: . , ! ? - & / @

    Args:
        text: Raw OCR text, potentially with noise. None → empty string.

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
    cleaned = cleaned.strip()  # BUG-20 FIX: strip again after noise removal
    return cleaned
