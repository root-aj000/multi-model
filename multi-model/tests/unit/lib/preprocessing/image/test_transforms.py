import numpy as np
from lib.preprocessing.image.transforms import resize_image, normalize_image, build_image_transform


def test_normalize_image():
    """normalize_image applies ImageNet standardization to a 3-channel image."""
    img = np.array([[[0, 128, 255], [64, 32, 16]]], dtype=np.uint8)  # shape (1, 2, 3)
    result = normalize_image(img)
    assert result.dtype == np.float32
    # After ImageNet normalization, values should be finite
    assert np.isfinite(result).all()


def test_build_image_transform_returns_callable():
    """build_image_transform returns a callable."""
    transform = build_image_transform(size=(224, 224))
    assert callable(transform)
