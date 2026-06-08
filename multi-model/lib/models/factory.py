"""
Model factory for creating and loading FG_MFN models.

BUG-11 FIX: load_model now returns the model in eval() mode.
BUG-21 FIX: load_tokenizer renamed to load_model_tokenizer to avoid
            shadowing lib.preprocessing.text.tokenizer.load_tokenizer.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoTokenizer

from lib.models.fg_mfn import FG_MFN

logger = logging.getLogger(__name__)


def load_model_tokenizer(encoder_name: str) -> Any:
    """
    Load and return a HuggingFace tokenizer for the given encoder name.

    NOTE: Prefer lib.preprocessing.text.tokenizer.load_tokenizer for
    training/evaluation pipelines — it has module-level caching.
    This function is provided for one-off inference use.

    Args:
        encoder_name: Valid HuggingFace model identifier.

    Returns:
        Configured tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    logger.info("Loaded tokenizer for encoder: %s", encoder_name)
    return tokenizer


# Keep the old name as an alias so existing callers don't break immediately,
# but log a deprecation warning.
def load_tokenizer(encoder_name: str) -> Any:
    """Deprecated alias for load_model_tokenizer. Use load_model_tokenizer instead."""
    logger.warning(
        "factory.load_tokenizer is deprecated. "
        "Use lib.preprocessing.text.tokenizer.load_tokenizer or "
        "lib.models.factory.load_model_tokenizer instead."
    )
    return load_model_tokenizer(encoder_name)


def create_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Create a fresh FG_MFN model from a configuration dictionary.

    The model is placed on `device` and set to train() mode.
    If multiple CUDA GPUs are available, the model is wrapped with
    nn.DataParallel so all GPUs are used automatically.  Callers that
    need to access FG_MFN-specific attributes (e.g. for checkpoint saving)
    should access model.module when DataParallel is active.

    Use load_model() when loading for inference — it sets eval() mode.

    Args:
        cfg:    Model configuration dict (IMAGE_BACKBONE, TEXT_ENCODER, …).
        device: Target device (should be torch.device("cuda") or "cuda:0").

    Returns:
        FG_MFN (or DataParallel-wrapped FG_MFN) in train() mode on `device`.
    """
    model = FG_MFN(cfg)
    model = model.to(device)

    num_gpus = torch.cuda.device_count()
    if device.type == "cuda" and num_gpus > 1:
        logger.info(
            "DataParallel: spreading model across %d GPUs: %s",
            num_gpus,
            [torch.cuda.get_device_name(i) for i in range(num_gpus)],
        )
        model = torch.nn.DataParallel(model)

    model.train()
    logger.info("Created fresh FG_MFN model on %s (num_gpus=%d)", device, max(num_gpus, 1))
    return model


def load_model(
    cfg: Dict[str, Any],
    device: torch.device,
    checkpoint_path: str,
) -> FG_MFN:
    """
    Load an FG_MFN model from a checkpoint file.

    BUG-11 FIX: Returns the model in eval() mode so BatchNorm and Dropout
    behave correctly during inference without requiring the caller to
    remember to call model.eval().

    Args:
        cfg:             Model configuration dict.
        device:          Target device.
        checkpoint_path: Path to the .pt / .pth checkpoint file.

    Returns:
        FG_MFN model with loaded weights in eval() mode on `device`.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError:      If the checkpoint cannot be loaded.
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    # create_model sets train() mode; we switch to eval() after loading weights
    model = create_model(cfg, device)

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True,
        )
    except TypeError as load_error:
        # Older PyTorch versions don't support weights_only=True
        logger.warning(
            "weights_only=True failed for '%s', retrying without it: %s",
            checkpoint_path, load_error,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
    except Exception as load_error:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}': {load_error}"
        ) from load_error

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as state_error:
        raise RuntimeError(
            f"Checkpoint state_dict does not match model architecture: {state_error}"
        ) from state_error

    # BUG-11 FIX: always return in eval mode for inference
    model.eval()
    logger.info("Loaded model from checkpoint '%s' (eval mode)", checkpoint_path)
    return model
