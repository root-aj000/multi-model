"""
Batch image prediction use case.

BUG-19 FIX: Catches Exception (not just RuntimeError) so all error types
            are handled gracefully without aborting the entire batch.
BUG-27 FIX: A single Predictor instance is created once and reused for
            every image in the batch instead of creating a new one per call.
"""

import logging
from typing import Any, Dict, List, Optional

from lib.ocr.engine import OCREngine
from lib.services.predictor import Predictor
from use_cases.prediction.predict_image import predict_image

logger = logging.getLogger(__name__)


def predict_batch(
    images: List[Any],
    model: Any,
    ocr_engine: OCREngine,
    label_maps: Dict[str, Any],
    filenames: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    predictor: Optional[Predictor] = None,
) -> List[Dict[str, Any]]:
    """
    Predict attributes for a batch of images.

    BUG-27 FIX: Creates one Predictor instance and passes it to every
    predict_image call instead of letting predict_image create a new one
    on each iteration.

    BUG-19 FIX: Catches all Exception subclasses (not just RuntimeError)
    so errors like ValueError (empty text), FileNotFoundError (missing image),
    and OSError don't abort the entire batch.

    Args:
        images:    List of input images (PIL Images, numpy arrays, or paths).
        model:     Loaded FG_MFN model.
        ocr_engine: OCR engine instance.
        label_maps: Dict mapping attributes to label name lists.
        filenames: Optional list of filenames (one per image).
        config:    Optional configuration dict.
        predictor: Optional pre-created Predictor. Created once if not provided.

    Returns:
        List of prediction dicts, one per image. Failed images get
        {"error": "<message>"} instead of crashing the batch.
    """
    # BUG-27 FIX: create one Predictor and reuse it for all images
    if predictor is None:
        predictor = Predictor(model, label_maps)

    results = []
    for idx, image in enumerate(images):
        filename = filenames[idx] if filenames and idx < len(filenames) else ""
        try:
            result = predict_image(
                image, model, ocr_engine, label_maps,
                filename=filename,
                config=config,
                predictor=predictor,  # BUG-27 FIX: pass shared predictor
            )
            results.append(result)
        except Exception as err:  # BUG-19 FIX: catch all exceptions
            logger.error(
                "Prediction failed for image at index %d (%s): %s",
                idx, type(err).__name__, err,
            )
            results.append({"error": str(err), "filename": filename})

    return results
