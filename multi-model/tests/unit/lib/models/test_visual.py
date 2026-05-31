"""
Tests for the VisualModule.
"""

import pytest
import torch

from lib.models.visual import VisualModule


def test_visual_module_init_default():
    """VisualModule initializes with default parameters."""
    module = VisualModule(backbone="resnet18", pretrained=False, out_features=32)
    assert module.backbone_name == "resnet18"
    assert module.out_features == 32


def test_visual_module_rejects_unsupported_backbone():
    """VisualModule raises ValueError for unsupported backbone."""
    with pytest.raises(ValueError, match="not supported"):
        VisualModule(backbone="vgg99", pretrained=False, out_features=32)


def test_visual_module_rejects_negative_out_features():
    """VisualModule raises ValueError for non-positive out_features."""
    with pytest.raises(ValueError, match="positive"):
        VisualModule(backbone="resnet18", pretrained=False, out_features=0)


def test_visual_module_forward_shape():
    """VisualModule forward returns correct output shape."""
    module = VisualModule(backbone="resnet18", pretrained=False, out_features=64)
    module.eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = module(images)
    assert output.shape == (2, 64)


def test_visual_module_rejects_non_4d_input():
    """VisualModule raises ValueError for non-4D input."""
    module = VisualModule(backbone="resnet18", pretrained=False, out_features=32)
    with pytest.raises(ValueError, match="4D"):
        module(torch.randn(2, 224, 224))


def test_visual_module_get_model_info():
    """VisualModule.get_model_info returns expected keys."""
    module = VisualModule(backbone="resnet18", pretrained=False, out_features=32)
    info = module.get_model_info()
    assert "backbone" in info
    assert "pretrained" in info
    assert "out_features" in info
    assert info["backbone"] == "resnet18"


def test_visual_module_freeze_unfreeze():
    """VisualModule freeze/unfreeze changes requires_grad on parameters."""
    module = VisualModule(backbone="resnet18", pretrained=False, out_features=32)
    module.freeze_backbone()
    for param in module.backbone.parameters():
        assert not param.requires_grad
    module.unfreeze_backbone()
    for param in module.backbone.parameters():
        assert param.requires_grad
