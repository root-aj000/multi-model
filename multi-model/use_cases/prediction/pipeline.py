"""
Prediction pipeline builder.

BUG-14 FIX: build_prediction_pipeline now requires a checkpoint_path
argument and calls load_model() to load trained weights. Previously it
called create_model() which produced a randomly initialised model,
making every prediction random.

All defaults (OCR engine, model dir, language) are read from
model_config.json (ocr section) rather than hardcoded here.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import torch

from lib.models.factory import load_model
from lib.ocr.factory import create_ocr_engine
from lib.services.predictor import Predictor
from lib.utils.config import get_label_maps, load_config


def build_prediction_pipeline(
    config_path: str,
    checkpoint_path: str,
) -> Predictor:
    """
    Build and return a fully-configured prediction pipeline.

    BUG-14 FIX: checkpoint_path is now a required argument. The model is
    loaded from the checkpoint using load_model() (which also sets eval mode,
    fixing BUG-11) instead of create_model() which produced random weights.

    Args:
        config_path:     Path to the prediction configuration JSON file.
        checkpoint_path: Path to the trained model checkpoint (.pt / .pth).

    Returns:
        Predictor instance ready for inference.

    Raises:
        FileNotFoundError: If config or checkpoint file does not exist.
        KeyError:          If required configuration keys are missing.
    """
    config = load_config(config_path)
    label_maps = get_label_maps(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BUG-14 FIX: load trained weights; BUG-11 FIX: load_model sets eval()
    model_cfg = config.get("model", config)
    model = load_model(model_cfg, device, checkpoint_path)

    ocr_config = config.get("ocr", {})
    engine_name = ocr_config.get("engine", "easyocr")
    model_dir = Path(ocr_config.get("model_dir", "local/ocr"))
    language = ocr_config.get("language", "en")
    # EasyOCR expects a list; PaddleOCR expects a string
    ocr_engine = create_ocr_engine(
        engine_name, model_dir,
        language=language,
        languages=[language],
    )

    predictor = Predictor(model, label_maps)
    logger.info(
        "Prediction pipeline built: checkpoint=%s, device=%s",
        checkpoint_path, device,
    )
    return predictor
