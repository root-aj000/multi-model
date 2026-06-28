"""
Single image prediction use case.

Provides functions to extract text from an image via OCR and predict
multi-attribute labels using the FG_MFN model.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from lib.ocr.engine import OCREngine
from lib.preprocessing.image.transforms import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    resize_image,
)
from lib.preprocessing.text.cleaner import clean_text
from lib.preprocessing.text.pipeline import (
    extract_call_to_action,
    extract_keywords,
    extract_monetary_mention,
    extract_objects_mentioned,
)
from lib.preprocessing.text.tokenizer import tokenize_text
from lib.services.postprocessor import format_prediction_result
from lib.services.predictor import Predictor

logger = logging.getLogger(__name__)

# Fallback text max length — should always be overridden by config
_DEFAULT_TEXT_MAX_LENGTH = 256


def extract_text(image: Any, ocr_engine: OCREngine) -> Tuple[str, float]:
    """
    Extract text from an image using the given OCR engine.

    Converts PIL Images to numpy arrays before passing to the OCR engine,
    since most OCR backends (EasyOCR, PaddleOCR) expect numpy arrays.

    Args:
        image: The input image (PIL Image, numpy array, or file path).
        ocr_engine: An instance of the OCREngine.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    ocr_input = _to_numpy_array(image)
    return ocr_engine.extract_text(ocr_input)


def predict_image(
    image: Any,
    model: Any,
    ocr_engine: OCREngine,
    label_maps: Dict[str, Any],
    filename: str = "",
    config: Optional[Dict] = None,
    predictor: Optional[Predictor] = None,
) -> Dict[str, Any]:
    """
    Predict attributes for a single image.

    Extracts text via OCR, cleans and tokenizes it, runs the model
    forward pass, and post-processes the results. The output format
    matches the frontend's expected field names.

    Image size and text max_length are read from config so they always
    match the values used during training.

    Args:
        image: The input image (PIL Image, numpy array, or file path).
        model: The loaded FG_MFN model.
        ocr_engine: The OCR engine instance.
        label_maps: Dictionary mapping attributes to label names.
        filename: Original filename of the uploaded image.
        config: Configuration dictionary. Must contain 'image_size' and
            'text_max_length' to match training settings.
        predictor: Optional pre-created Predictor instance. If provided,
            avoids creating a new Predictor on every call, reducing overhead.

    Returns:
        Dictionary containing the prediction results with OCR metadata
        and extracted text features, using field names expected by the
        frontend (ocr_text, predicted_label, keywords, etc.).
    """
    cfg = config or {}

    if predictor is None:
        predictor = Predictor(model, label_maps)

    raw_text, confidence = extract_text(image, ocr_engine)
    cleaned_text = clean_text(raw_text)

    # Read image_size from config — must match training
    image_size = _get_image_size(cfg)
    image_tensor = _prepare_image_tensor(image, image_size=image_size)

    # Read text_max_length from config — must match training
    text_max_length = cfg.get("text_max_length", _DEFAULT_TEXT_MAX_LENGTH)
    input_ids, attention_mask = _prepare_text_tensors(
        cleaned_text, max_length=text_max_length,
    )

    result = predictor.predict_single(image_tensor, input_ids, attention_mask)

    clean_result = format_prediction_result(result)

    # Add OCR text using the frontend's expected field name
    clean_result["ocr_text"] = raw_text

    # Add filename for the frontend
    clean_result["filename"] = filename

    # Add primary label (first attribute's prediction)
    clean_result["predicted_label"] = _extract_primary_label(clean_result, label_maps)

    # Feature extraction uses raw_text (before cleaning) so that currency
    # symbols and punctuation are still present for regex-based extractors.
    # The model itself receives cleaned_text via the tokenizer.
    clean_result["keywords"] = _format_keywords(extract_keywords(raw_text))
    clean_result["monetary_mention"] = extract_monetary_mention(raw_text) or ""
    clean_result["call_to_action"] = extract_call_to_action(raw_text) or ""
    clean_result["object_detected"] = _format_objects(extract_objects_mentioned(raw_text))

    return clean_result


def _get_image_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    Read image_size from config and return as a (width, height) tuple.

    Args:
        config: Configuration dictionary.

    Returns:
        (width, height) tuple.
    """
    raw = config.get("image_size", DEFAULT_IMAGE_SIZE)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    if isinstance(raw, int):
        return (raw, raw)
    return DEFAULT_IMAGE_SIZE


def _extract_primary_label(
    result: Dict[str, Any],
    label_maps: Dict[str, Any],
) -> str:
    """
    Extract the primary label from prediction results.

    Uses the first attribute in label_maps as the primary label.

    Args:
        result: Post-processed prediction result dictionary.
        label_maps: Dictionary mapping attribute names to label lists.

    Returns:
        The primary predicted label string, or empty string if none found.
    """
    first_attr = next(iter(label_maps), None)
    if first_attr and first_attr in result:
        return str(result[first_attr])
    return ""


def _format_keywords(keywords: List[str]) -> str:
    """
    Format a list of keywords into a comma-separated string.

    Args:
        keywords: List of keyword strings.

    Returns:
        Comma-separated string of keywords, or empty string if none.
    """
    if not keywords:
        return ""
    return ", ".join(keywords)


def _format_objects(objects: List[str]) -> str:
    """
    Format a list of detected objects into a comma-separated string.

    Args:
        objects: List of object name strings.

    Returns:
        Comma-separated string of objects, or empty string if none.
    """
    if not objects:
        return ""
    return ", ".join(objects)


def _to_numpy_array(image: Any) -> np.ndarray:
    """
    Convert an image to a numpy array suitable for OCR engines.

    EasyOCR and PaddleOCR expect numpy arrays, not PIL Images.
    This function ensures the image is always converted to a
    numpy array in RGB format.

    Args:
        image: PIL Image, numpy array, or file path string.

    Returns:
        numpy array of the image in RGB format with dtype uint8.
    """
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
        return np.array(pil_image)
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        return image
    return np.array(image)


def _prepare_image_tensor(
    image: Any,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Any:
    """
    Convert an image to a batched tensor suitable for model input.

    Applies the same preprocessing used during training:
      1. Convert to RGB numpy array
      2. Handle grayscale / RGBA channels
      3. Resize to image_size (must match training)
      4. Scale to [0, 1]
      5. Apply ImageNet mean/std normalisation
      6. Transpose to (C, H, W) and add batch dimension

    Args:
        image: PIL Image, numpy array, or file path string.
        image_size: (width, height) to resize to. Must match training config.

    Returns:
        Batched image tensor of shape (1, C, H, W).
    """
    image_array = _to_numpy_array(image)

    # Handle grayscale: (H, W) -> (H, W, 3) or (H, W, 1) -> (H, W, 3)
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = np.repeat(image_array, 3, axis=2)
    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        # Drop alpha channel for RGBA images
        image_array = image_array[:, :, :3]

    # Resize to match training image size — critical for consistent features
    image_array = resize_image(image_array, image_size)

    # Scale to [0, 1]
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float32) / 255.0
    else:
        image_array = image_array.astype(np.float32)

    # Apply ImageNet normalisation (same as training transform)
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD

    image_tensor = torch.from_numpy(
        np.transpose(image_array, (2, 0, 1))
    ).unsqueeze(0)

    return image_tensor


def _prepare_text_tensors(
    text: str,
    max_length: int = _DEFAULT_TEXT_MAX_LENGTH,
) -> Tuple[Any, Any]:
    """
    Tokenize text and return batched input_ids and attention_mask tensors.

    max_length is read from config via the caller to ensure it always
    matches the value used during training.

    Args:
        text: Cleaned text string to tokenize.
        max_length: Maximum token sequence length. Must match training config.

    Returns:
        Tuple of (input_ids, attention_mask), each of shape (1, seq_len).
    """
    if not text or not text.strip():
        # Return a zero-padded encoding rather than passing a space hack
        input_ids = torch.zeros(1, max_length, dtype=torch.long)
        attention_mask = torch.zeros(1, max_length, dtype=torch.long)
        return input_ids, attention_mask

    encoding = tokenize_text(text, max_length=max_length)
    input_ids = encoding["input_ids"].unsqueeze(0)
    attention_mask = encoding["attention_mask"].unsqueeze(0)
    return input_ids, attention_mask
