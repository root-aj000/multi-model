"""
Tests for the image preprocessing pipeline builder.
"""

from lib.preprocessing.image.pipeline import build_image_pipeline


def test_build_image_pipeline_returns_callable():
    """build_image_pipeline returns a callable."""
    pipeline = build_image_pipeline({"image_size": (224, 224), "augment": False})
    assert callable(pipeline)


def test_build_image_pipeline_default_config():
    """build_image_pipeline works with an empty config dict."""
    pipeline = build_image_pipeline({})
    assert callable(pipeline)


def test_build_image_pipeline_with_augmentation():
    """build_image_pipeline returns a callable when augment is True."""
    pipeline = build_image_pipeline({"image_size": (224, 224), "augment": True})
    assert callable(pipeline)
