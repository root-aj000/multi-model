"""
Tests for the FG_MFN multi-modal fusion model.
"""

import pytest
import torch

from lib.models.fg_mfn import FG_MFN


def _make_minimal_config():
    """Create a minimal valid configuration for FG_MFN."""
    return {
        "IMAGE_BACKBONE": "resnet18",
        "TEXT_ENCODER": "distilbert-base-uncased",
        "HIDDEN_DIM": 32,
        "DROPOUT": 0.1,
        "FUSION_TYPE": "concat",
        "ATTRIBUTES": {
            "sentiment": {
                "num_classes": 3,
                "labels": ["positive", "neutral", "negative"],
            },
        },
    }


def test_fg_mfn_init_with_valid_config():
    """FG_MFN initializes without error with a valid config."""
    cfg = _make_minimal_config()
    model = FG_MFN(cfg)
    assert len(model.attribute_heads) == 1
    assert "sentiment" in model.attribute_heads


def test_fg_mfn_init_rejects_missing_keys():
    """FG_MFN raises ValueError when required config keys are missing."""
    with pytest.raises(ValueError):
        FG_MFN({"HIDDEN_DIM": 32})


def test_fg_mfn_init_rejects_non_dict():
    """FG_MFN raises TypeError when config is not a dictionary."""
    with pytest.raises(TypeError):
        FG_MFN("not a dict")


def test_fg_mfn_init_rejects_invalid_hidden_dim():
    """FG_MFN raises ValueError when HIDDEN_DIM is not positive."""
    cfg = _make_minimal_config()
    cfg["HIDDEN_DIM"] = -1
    with pytest.raises(ValueError):
        FG_MFN(cfg)


def test_fg_mfn_init_rejects_invalid_dropout():
    """FG_MFN raises ValueError when DROPOUT is out of range."""
    cfg = _make_minimal_config()
    cfg["DROPOUT"] = 1.5
    with pytest.raises(ValueError):
        FG_MFN(cfg)


def test_fg_mfn_forward_returns_dict():
    """FG_MFN forward pass returns a dict with attribute keys."""
    cfg = _make_minimal_config()
    model = FG_MFN(cfg)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)

    with torch.no_grad():
        outputs = model(images, input_ids, attention_mask)

    assert "sentiment" in outputs
    assert outputs["sentiment"].shape[0] == 2
    assert outputs["sentiment"].shape[1] == 3


def test_fg_mfn_concat_fusion_doubles_dim():
    """FG_MFN with concat fusion creates a shared layer with doubled input dim."""
    cfg = _make_minimal_config()
    cfg["FUSION_TYPE"] = "concat"
    model = FG_MFN(cfg)
    assert model.shared_fc[0].in_features == 64  # 32 * 2


def test_fg_mfn_add_fusion_keeps_dim():
    """FG_MFN with add fusion creates a shared layer with same input dim."""
    cfg = _make_minimal_config()
    cfg["FUSION_TYPE"] = "add"
    model = FG_MFN(cfg)
    assert model.shared_fc[0].in_features == 32


def test_fg_mfn_invalid_fusion_type_raises():
    """FG_MFN raises ValueError for unsupported fusion type."""
    cfg = _make_minimal_config()
    cfg["FUSION_TYPE"] = "invalid"
    with pytest.raises(ValueError):
        FG_MFN(cfg)


def test_fg_mfn_validate_inputs_rejects_wrong_dims():
    """FG_MFN raises ValueError when image tensor is not 4D."""
    cfg = _make_minimal_config()
    model = FG_MFN(cfg)
    with pytest.raises(ValueError):
        model._validate_inputs(
            torch.randn(2, 224, 224),
            torch.randint(0, 1000, (2, 128)),
            torch.ones(2, 128),
        )


def test_fg_mfn_validate_inputs_rejects_batch_mismatch():
    """FG_MFN raises ValueError when batch sizes do not match."""
    cfg = _make_minimal_config()
    model = FG_MFN(cfg)
    with pytest.raises(ValueError):
        model._validate_inputs(
            torch.randn(2, 3, 224, 224),
            torch.randint(0, 1000, (3, 128)),
            torch.ones(3, 128),
        )


def test_fg_mfn_get_label_names():
    """FG_MFN returns label names for a known attribute."""
    cfg = _make_minimal_config()
    model = FG_MFN(cfg)
    labels = model.get_label_names("sentiment")
    assert labels == ["positive", "neutral", "negative"]


def test_fg_mfn_get_label_names_unknown_attribute():
    """FG_MFN returns None for an unknown attribute."""
    cfg = _make_minimal_config()
    model = FG_MFN(cfg)
    labels = model.get_label_names("nonexistent")
    assert labels is None


def test_fg_mfn_deep_shared_layer():
    """FG_MFN creates a deeper shared layer when DEEP_SHARED_LAYER is True."""
    cfg = _make_minimal_config()
    cfg["DEEP_SHARED_LAYER"] = True
    model = FG_MFN(cfg)
    assert len(model.shared_fc) == 6  # Linear, ReLU, Dropout, Linear, ReLU, Dropout
