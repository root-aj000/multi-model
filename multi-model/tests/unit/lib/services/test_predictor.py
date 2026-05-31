"""
Tests for the Predictor service.
"""

from unittest.mock import MagicMock

import pytest
import torch

from lib.services.predictor import Predictor


def _make_mock_model():
    """Create a mock FG_MFN model that returns attribute logits."""
    model = MagicMock()
    sentiment_logits = torch.tensor([[0.1, 0.5, 0.4]])
    model.return_value = {"sentiment": sentiment_logits}
    model.eval.return_value = model
    return model


def test_predictor_init_rejects_none_model():
    """Predictor raises ValueError when model is None."""
    with pytest.raises(ValueError, match="model must not be None"):
        Predictor(model=None, label_maps={"sentiment": ["pos", "neg", "neu"]})


def test_predictor_init_rejects_none_label_maps():
    """Predictor raises ValueError when label_maps is None."""
    model = _make_mock_model()
    with pytest.raises(ValueError, match="label_maps must not be None"):
        Predictor(model=model, label_maps=None)


def test_predictor_predict_single_returns_dict():
    """Predictor.predict_single returns a dictionary with attribute keys."""
    model = _make_mock_model()
    label_maps = {"sentiment": ["positive", "neutral", "negative"]}
    predictor = Predictor(model, label_maps)

    images = torch.randn(1, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones(1, 128)

    result = predictor.predict_single(images, input_ids, attention_mask)
    assert isinstance(result, dict)
    assert "sentiment" in result


def test_predictor_predict_single_includes_confidence():
    """Predictor.predict_single includes confidence scores."""
    model = _make_mock_model()
    label_maps = {"sentiment": ["positive", "neutral", "negative"]}
    predictor = Predictor(model, label_maps)

    images = torch.randn(1, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones(1, 128)

    result = predictor.predict_single(images, input_ids, attention_mask)
    assert "sentiment_confidence" in result
