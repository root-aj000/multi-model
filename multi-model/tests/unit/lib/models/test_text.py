"""
Tests for the TextModule.
"""

import pytest
import torch

from lib.models.text import TextModule


def test_text_module_init_default():
    """TextModule initializes with default parameters."""
    module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=32)
    assert module.encoder_name == "distilbert-base-uncased"
    assert module.hidden_size == 32


def test_text_module_rejects_non_string_encoder():
    """TextModule raises TypeError when encoder_name is not a string."""
    with pytest.raises(TypeError, match="string"):
        TextModule(encoder_name=123, hidden_size=32)


def test_text_module_rejects_negative_hidden_size():
    """TextModule raises ValueError when hidden_size is not positive."""
    with pytest.raises(ValueError, match="positive"):
        TextModule(encoder_name="distilbert-base-uncased", hidden_size=-1)


def test_text_module_forward_shape():
    """TextModule forward returns correct output shape."""
    module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=64)
    module.eval()
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)
    with torch.no_grad():
        output = module(input_ids, attention_mask)
    assert output.shape == (2, 64)


def test_text_module_rejects_shape_mismatch():
    """TextModule raises ValueError when input_ids and attention_mask shapes differ."""
    module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=32)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(3, 128)
    with pytest.raises(ValueError, match="does not match"):
        module(input_ids, attention_mask)


def test_text_module_get_model_info():
    """TextModule.get_model_info returns expected keys."""
    module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=32)
    info = module.get_model_info()
    assert "encoder_name" in info
    assert "hidden_size" in info
    assert info["encoder_name"] == "distilbert-base-uncased"


def test_text_module_get_cache_info():
    """TextModule.get_cache_info returns encoder_name key."""
    module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=32)
    info = module.get_cache_info()
    assert "encoder_name" in info
