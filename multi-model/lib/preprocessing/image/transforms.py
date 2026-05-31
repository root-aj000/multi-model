"""
Image transformation functions for preprocessing.

BUG-13 FIX: Hue augmentation now performs a real HSV hue rotation instead
of the previous additive RGB shift (which was just brightness noise).

Augmentation is always applied BEFORE normalization so that color/brightness
operations work on raw [0, 255] uint8 pixel values.
"""

import colorsys
import random
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance

# Default image input size — must match image_size in model_config.json
DEFAULT_IMAGE_SIZE = (512, 512)

# ImageNet normalization parameters expected by pretrained backbones
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default augmentation parameters (used when no augmentation config is provided)
_DEFAULT_AUG: Dict[str, Any] = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.1,
    "rotation_degrees": 15,
    "color_jitter": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.2,
        # hue: fraction of the full hue circle [0, 0.5]
        # 0.05 means ±18° rotation (5% of 360°)
        "hue": 0.05,
    },
    "random_crop_scale": [0.8, 1.0],
    "random_crop_ratio": [0.9, 1.1],
}


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: int = Image.LANCZOS,
) -> np.ndarray:
    """
    Resize an image to the given (width, height).

    Args:
        image:         NumPy array of shape (H, W, C).
        size:          Target (width, height).
        interpolation: PIL interpolation method. LANCZOS for downscaling.

    Returns:
        Resized image array.
    """
    pil_image = Image.fromarray(image)
    resized = pil_image.resize(size, interpolation)
    return np.array(resized)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values using ImageNet mean/std.

    Args:
        image: uint8 or float array with values in [0, 255].

    Returns:
        Float32 array normalized by ImageNet mean/std.
    """
    normalized = image.astype(np.float32) / 255.0
    return (normalized - IMAGENET_MEAN) / IMAGENET_STD


def _hue_shift(pil_image: Image.Image, hue_factor: float) -> Image.Image:
    """
    Rotate the hue channel of an RGB image.

    BUG-13 FIX: The original code added a constant to all RGB channels,
    which changes brightness, not hue. This function converts to HSV,
    rotates the H channel, and converts back — a true hue shift.

    Args:
        pil_image:  Input PIL Image in RGB mode.
        hue_factor: Hue shift in [-0.5, 0.5] (fraction of full circle).
                    0.05 ≈ ±18° rotation.

    Returns:
        PIL Image with hue rotated.
    """
    arr = np.array(pil_image, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)

    shifted = np.empty_like(flat)
    for i, (r, g, b) in enumerate(flat):
        hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
        hue = (hue + hue_factor) % 1.0
        shifted[i] = colorsys.hsv_to_rgb(hue, sat, val)

    result = (shifted.reshape(h, w, 3) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


def augment_image(
    image: np.ndarray,
    aug_cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply random augmentation to a uint8 image array.

    Must be called BEFORE normalization so color/brightness operations
    work on raw [0, 255] values.

    Augmentations applied (each independently at random):
      - Horizontal flip
      - Vertical flip
      - Random rotation
      - Color jitter (brightness, contrast, saturation, hue)
      - Random resized crop

    Args:
        image:   (H, W, C) uint8 array.
        aug_cfg: Augmentation config dict. Falls back to _DEFAULT_AUG if None.

    Returns:
        Augmented (H, W, C) uint8 array.
    """
    cfg = aug_cfg or _DEFAULT_AUG
    pil_image = Image.fromarray(image.astype(np.uint8))

    # Horizontal flip
    if random.random() < cfg.get("horizontal_flip_prob", 0.5):
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Vertical flip
    if random.random() < cfg.get("vertical_flip_prob", 0.1):
        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Random rotation
    rotation_degrees = cfg.get("rotation_degrees", 15)
    if rotation_degrees > 0 and random.random() < 0.4:
        angle = random.uniform(-rotation_degrees, rotation_degrees)
        pil_image = pil_image.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Color jitter
    jitter_cfg = cfg.get("color_jitter", {})
    if jitter_cfg:
        brightness = jitter_cfg.get("brightness", 0.3)
        contrast = jitter_cfg.get("contrast", 0.3)
        saturation = jitter_cfg.get("saturation", 0.2)
        hue = jitter_cfg.get("hue", 0.05)

        if brightness > 0 and random.random() < 0.5:
            factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
            pil_image = ImageEnhance.Brightness(pil_image).enhance(factor)

        if contrast > 0 and random.random() < 0.5:
            factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
            pil_image = ImageEnhance.Contrast(pil_image).enhance(factor)

        if saturation > 0 and random.random() < 0.5:
            factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
            pil_image = ImageEnhance.Color(pil_image).enhance(factor)

        if hue > 0 and random.random() < 0.3:
            # BUG-13 FIX: real HSV hue rotation, not additive RGB shift
            shift = random.uniform(-hue, hue)
            pil_image = _hue_shift(pil_image, shift)

    # Random resized crop
    crop_scale = cfg.get("random_crop_scale", [0.8, 1.0])
    crop_ratio = cfg.get("random_crop_ratio", [0.9, 1.1])
    if random.random() < 0.5:
        w, h = pil_image.size
        scale = random.uniform(crop_scale[0], crop_scale[1])
        ratio = random.uniform(crop_ratio[0], crop_ratio[1])
        crop_w = min(int(w * scale * ratio), w)
        crop_h = min(int(h * scale / ratio), h)
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        pil_image = pil_image.crop((left, top, left + crop_w, top + crop_h))
        pil_image = pil_image.resize((w, h), Image.LANCZOS)

    return np.array(pil_image)


def build_image_transform(
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    augment: bool = False,
    aug_cfg: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Build an image transformation pipeline.

    Pipeline order: resize → [augment on raw pixels] → normalize → tensor.

    Args:
        size:    Target (width, height). Must match image_size in config.
        augment: Whether to apply augmentation (True for train, False for val/test).
        aug_cfg: Augmentation config dict. Uses _DEFAULT_AUG if None.

    Returns:
        Callable: (H,W,C) uint8 numpy array → (C,H,W) float32 tensor.
    """
    def transform(image: np.ndarray) -> torch.Tensor:
        image = resize_image(image, size)
        if augment:
            image = augment_image(image, aug_cfg=aug_cfg)
        image = normalize_image(image)
        # (H, W, C) → (C, H, W)
        return torch.from_numpy(image.transpose(2, 0, 1)).float()

    return transform
