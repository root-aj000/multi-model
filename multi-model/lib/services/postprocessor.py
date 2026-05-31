"""
Prediction postprocessing service.

BUG-15 FIX: format_prediction_result now accepts an include_confidence
parameter. When True, per-attribute confidence scores are kept in the
output instead of being stripped. Defaults to False for backward compat.
"""

from typing import Any, Dict

# Suffixes that mark internal score fields
_SCORE_SUFFIXES = ("_confidence", "_score", "_accuracy", "_f1")

# Exact field names that are always excluded (legacy internal fields)
_EXCLUDED_FIELDS = {"confidence_score", "predicted_label_num"}

# Suffix for numeric class index fields — always excluded from API output
_NUM_SUFFIX = "_predicted_label_num"

# Field rename: predicted_label_text → predicted_label
_RENAMED_FIELD = "predicted_label_text"
_RENAMED_DESTINATION = "predicted_label"


def format_prediction_result(
    raw_result: Dict[str, Any],
    include_confidence: bool = False,
) -> Dict[str, Any]:
    """
    Format the raw prediction result for API output.

    BUG-15 FIX: Added include_confidence parameter. Previously all
    confidence scores were unconditionally stripped, making it impossible
    for callers to access them. Now callers can opt in.

    Args:
        raw_result:         Raw result dict from the Predictor.
        include_confidence: If True, keep per-attribute confidence scores
                            (keys ending in '_confidence') in the output.
                            Defaults to False for backward compatibility.

    Returns:
        Cleaned result dict. Internal numeric index fields are always
        removed. Confidence scores are removed unless include_confidence=True.
    """
    cleaned: Dict[str, Any] = {}

    for key, value in raw_result.items():
        # Always exclude exact internal field names
        if key in _EXCLUDED_FIELDS:
            continue
        # Always exclude numeric label index fields
        if key.endswith(_NUM_SUFFIX):
            continue
        # Exclude score suffixes unless caller opts in to confidence
        if not include_confidence and any(key.endswith(s) for s in _SCORE_SUFFIXES):
            continue
        cleaned[key] = value

    # Rename predicted_label_text → predicted_label if present
    if _RENAMED_FIELD in cleaned:
        cleaned[_RENAMED_DESTINATION] = cleaned.pop(_RENAMED_FIELD)

    return cleaned
