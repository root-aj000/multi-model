"""
Image preprocessing pipeline builder.

BUG-09 FIX: DEFAULT_IMAGE_SIZE is now imported from transforms.py (512×512)
instead of being hardcoded here as (224×224), which mismatched the model config.
"""

from typing import Any, Callable, Dict

from lib.preprocessing.image.transforms import DEFAULT_IMAGE_SIZE, build_image_transform


def build_image_pipeline(config: Dict[str, Any]) -> Callable:
    """
    Build the complete image preprocessing pipeline from a config dict.

    BUG-09 FIX: The previous version had DEFAULT_IMAGE_SIZE = (224, 224)
    which silently produced wrong-sized images when image_size was absent
    from config. Now falls back to transforms.DEFAULT_IMAGE_SIZE (512, 512)
    which matches the model config default.

    Args:
        config: Dict with optional keys:
            image_size : (width, height) tuple or [w, h] list.
                         Defaults to DEFAULT_IMAGE_SIZE (512, 512).
            augment    : bool — enable augmentation. Defaults to False.
            augmentation: dict — augmentation parameters passed through.

    Returns:
        Callable that transforms a (H,W,C) uint8 numpy array to a
        (C,H,W) float32 tensor.
    """
    raw_size = config.get("image_size", DEFAULT_IMAGE_SIZE)

    # Normalise to a (width, height) tuple
    if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
        image_size = (int(raw_size[0]), int(raw_size[1]))
    elif isinstance(raw_size, int):
        image_size = (raw_size, raw_size)
    else:
        import logging
        logging.getLogger(__name__).warning(
            "Invalid image_size %r in config; using default %s", raw_size, DEFAULT_IMAGE_SIZE
        )
        image_size = DEFAULT_IMAGE_SIZE

    augment = config.get("augment", False)
    aug_cfg = config.get("augmentation", None)
    return build_image_transform(size=image_size, augment=augment, aug_cfg=aug_cfg)
