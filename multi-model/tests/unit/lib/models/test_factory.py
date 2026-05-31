"""
Tests for the model factory module.
"""

import pytest
import torch

from lib.models.factory import create_model, list_supported_encoders, load_tokenizer


def test_list_supported_encoders_returns_dict():
    """list_supported_encoders returns a non-empty dictionary."""
    encoders = list_supported_encoders()
    assert isinstance(encoders, dict)
    assert len(encoders) > 0


def test_load_tokenizer_returns_tokenizer():
    """load_tokenizer returns a tokenizer with expected methods."""
    tokenizer = load_tokenizer("distilbert-base-uncased")
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")


def test_create_model_returns_model():
    """create_model returns an FG_MFN model on the specified device."""
    cfg = {
        "IMAGE_BACKBONE": "resnet18",
        "TEXT_ENCODER": "distilbert-base-uncased",
        "HIDDEN_DIM": 32,
        "DROPOUT": 0.1,
        "FUSION_TYPE": "concat",
        "ATTRIBUTES": {
            "sentiment": {"num_classes": 3, "labels": ["pos", "neg", "neu"]},
        },
    }
    device = torch.device("cpu")
    model = create_model(cfg, device)
    assert model is not None
    assert len(model.attribute_heads) == 1
