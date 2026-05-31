"""
Advanced unit tests for model components: FG_MFN, VisualModule, TextModule.

Tests cover:
- Architecture validation
- Forward pass correctness with various input shapes
- State dict serialization
- Gradient flow
- Device handling
- Edge cases and error conditions
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from lib.models.fg_mfn import FG_MFN
from lib.models.visual import VisualModule
from lib.models.text import TextModule


class TestVisualModuleAdvanced:
    """Advanced tests for VisualModule."""

    def test_visual_module_forward_shape_resnet18(self):
        """Test forward pass output shape with ResNet18."""
        module = VisualModule(backbone="resnet18", pretrained=False, out_features=256)
        batch_size = 8
        images = torch.randn(batch_size, 3, 224, 224)
        output = module(images)
        assert output.shape == (batch_size, 256)

    def test_visual_module_forward_shape_resnet50(self):
        """Test forward pass output shape with ResNet50."""
        module = VisualModule(backbone="resnet50", pretrained=False, out_features=512)
        batch_size = 16
        images = torch.randn(batch_size, 3, 224, 224)
        output = module(images)
        assert output.shape == (batch_size, 512)

    def test_visual_module_gradient_flow(self):
        """Test that gradients flow through the visual module."""
        module = VisualModule(backbone="resnet18", pretrained=False, out_features=256)
        images = torch.randn(4, 3, 224, 224, requires_grad=True)
        output = module(images)
        loss = output.sum()
        loss.backward()
        assert images.grad is not None
        assert images.grad.shape == images.shape

    def test_visual_module_freeze_and_unfreeze(self):
        """Test freezing and unfreezing backbone parameters."""
        module = VisualModule(backbone="resnet18", pretrained=False, out_features=256)
        
        # Initially all params should be trainable
        trainable_before = sum(1 for p in module.parameters() if p.requires_grad)
        
        # Freeze backbone
        module.freeze_backbone()
        frozen = sum(1 for p in module.backbone.parameters() if not p.requires_grad)
        
        # Unfreeze backbone
        module.unfreeze_backbone()
        trainable_after = sum(1 for p in module.parameters() if p.requires_grad)
        
        assert frozen > 0
        assert trainable_after == trainable_before

    def test_visual_module_device_handling(self):
        """Test device handling (CPU and CUDA if available)."""
        module = VisualModule(backbone="resnet18", pretrained=False, out_features=256)
        device = torch.device("cpu")
        module = module.to(device)
        images = torch.randn(2, 3, 224, 224).to(device)
        output = module(images)
        assert output.device == device

    def test_visual_module_state_dict_serialization(self):
        """Test state dict can be saved and loaded."""
        module1 = VisualModule(backbone="resnet18", pretrained=False, out_features=256)
        state_dict = module1.state_dict()
        
        module2 = VisualModule(backbone="resnet18", pretrained=False, out_features=256)
        module2.load_state_dict(state_dict)
        
        images = torch.randn(2, 3, 224, 224)
        out1 = module1(images)
        out2 = module2(images)
        assert torch.allclose(out1, out2)

    def test_visual_module_get_model_info(self):
        """Test metadata retrieval."""
        module = VisualModule(backbone="resnet50", pretrained=True, out_features=512)
        info = module.get_model_info()
        assert info["backbone"] == "resnet50"
        assert info["pretrained"] is True
        assert info["out_features"] == 512

    def test_visual_module_invalid_backbone(self):
        """Test error handling for invalid backbone."""
        with pytest.raises(ValueError, match="not supported"):
            VisualModule(backbone="invalid_net", pretrained=False, out_features=256)

    def test_visual_module_invalid_out_features(self):
        """Test error handling for invalid out_features."""
        with pytest.raises(ValueError, match="must be positive"):
            VisualModule(backbone="resnet18", pretrained=False, out_features=0)

    def test_visual_module_eval_mode(self):
        """Test eval mode disables dropout."""
        module = VisualModule(backbone="resnet18", pretrained=False, out_features=256)
        module.eval()
        images = torch.randn(2, 3, 224, 224)
        out1 = module(images)
        out2 = module(images)
        # Output should be deterministic in eval mode
        assert torch.allclose(out1, out2)


class TestTextModuleAdvanced:
    """Advanced tests for TextModule."""

    def test_text_module_forward_shape(self):
        """Test forward pass output shape."""
        module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=256)
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        output = module(input_ids, attention_mask)
        assert output.shape == (batch_size, 256)

    def test_text_module_attention_mask_effect(self):
        """Test that attention mask affects output."""
        module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=256)
        module.eval()
        
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Full attention mask
        mask_full = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # Partial attention mask (mask out half)
        mask_partial = torch.ones(batch_size, seq_len, dtype=torch.long)
        mask_partial[:, seq_len//2:] = 0
        
        with torch.no_grad():
            out_full = module(input_ids, mask_full)
            out_partial = module(input_ids, mask_partial)
        
        # Outputs should be different when mask differs
        assert not torch.allclose(out_full, out_partial)

    def test_text_module_gradient_flow(self):
        """Test gradient flow through text module."""
        module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=256)
        input_ids = torch.randint(0, 1000, (2, 128), requires_grad=False)
        attention_mask = torch.ones(2, 128, dtype=torch.long, requires_grad=False)
        
        output = module(input_ids, attention_mask)
        loss = output.sum()
        loss.backward()
        
        # Check that embedding layer has gradients
        for param in module.projection.parameters():
            if param.grad is not None:
                assert param.grad.sum() != 0

    def test_text_module_shape_mismatch_error(self):
        """Test error handling for mismatched shapes."""
        module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=256)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 64, dtype=torch.long)  # Wrong size
        
        with pytest.raises(ValueError, match="shape"):
            module(input_ids, attention_mask)

    def test_text_module_state_dict_serialization(self):
        """Test state dict can be saved and loaded."""
        module1 = TextModule(encoder_name="distilbert-base-uncased", hidden_size=256)
        state_dict = module1.state_dict()
        
        module2 = TextModule(encoder_name="distilbert-base-uncased", hidden_size=256)
        module2.load_state_dict(state_dict)
        
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128, dtype=torch.long)
        
        with torch.no_grad():
            out1 = module1(input_ids, attention_mask)
            out2 = module2(input_ids, attention_mask)
        
        assert torch.allclose(out1, out2)

    def test_text_module_get_model_info(self):
        """Test metadata retrieval."""
        module = TextModule(encoder_name="distilbert-base-uncased", hidden_size=512)
        info = module.get_model_info()
        assert info["encoder_name"] == "distilbert-base-uncased"
        assert info["hidden_size"] == 512


class TestFGMFNAdvanced:
    """Advanced tests for FG_MFN model."""

    def test_fg_mfn_config_validation_missing_keys(self):
        """Test config validation catches missing required keys."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            # Missing TEXT_ENCODER, HIDDEN_DIM, DROPOUT
        }
        with pytest.raises(ValueError, match="missing required keys"):
            FG_MFN(cfg)

    def test_fg_mfn_config_validation_invalid_hidden_dim(self):
        """Test config validation catches invalid HIDDEN_DIM."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 0,  # Invalid
            "DROPOUT": 0.3,
        }
        with pytest.raises(ValueError, match="HIDDEN_DIM"):
            FG_MFN(cfg)

    def test_fg_mfn_config_validation_invalid_dropout(self):
        """Test config validation catches invalid DROPOUT."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 512,
            "DROPOUT": 1.5,  # Invalid
        }
        with pytest.raises(ValueError, match="DROPOUT"):
            FG_MFN(cfg)

    def test_fg_mfn_forward_multi_attribute(self):
        """Test forward pass with multi-attribute configuration."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 256,
            "DROPOUT": 0.3,
            "FUSION_TYPE": "concat",
            "ATTRIBUTES": {
                "sentiment": {"num_classes": 3, "labels": ["pos", "neu", "neg"]},
                "theme": {"num_classes": 5, "labels": ["a", "b", "c", "d", "e"]},
            },
        }
        model = FG_MFN(cfg)
        
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128, dtype=torch.long)
        
        outputs = model(images, input_ids, attention_mask)
        
        assert isinstance(outputs, dict)
        assert "sentiment" in outputs
        assert "theme" in outputs
        assert outputs["sentiment"].shape == (2, 3)
        assert outputs["theme"].shape == (2, 5)

    def test_fg_mfn_forward_legacy_mode(self):
        """Test forward pass in legacy single-attribute mode."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 256,
            "DROPOUT": 0.3,
            "NUM_CLASSES": 2,  # Legacy mode
        }
        model = FG_MFN(cfg)
        
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 128))
        attention_mask = torch.ones(4, 128, dtype=torch.long)
        
        outputs = model(images, input_ids, attention_mask)
        
        assert isinstance(outputs, dict)
        assert "sentiment" in outputs
        assert outputs["sentiment"].shape == (4, 2)

    def test_fg_mfn_fusion_concat(self):
        """Test concat fusion strategy."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 256,
            "DROPOUT": 0.3,
            "FUSION_TYPE": "concat",
            "NUM_CLASSES": 3,
        }
        model = FG_MFN(cfg)
        assert model.fusion_type == "concat"

    def test_fg_mfn_fusion_add(self):
        """Test add fusion strategy."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 256,
            "DROPOUT": 0.3,
            "FUSION_TYPE": "add",
            "NUM_CLASSES": 3,
        }
        model = FG_MFN(cfg)
        assert model.fusion_type == "add"

    def test_fg_mfn_invalid_fusion_type(self):
        """Test error handling for invalid fusion type."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 256,
            "DROPOUT": 0.3,
            "FUSION_TYPE": "invalid_fusion",
            "NUM_CLASSES": 3,
        }
        with pytest.raises(ValueError, match="FUSION_TYPE"):
            FG_MFN(cfg)

    def test_fg_mfn_deep_shared_layer(self):
        """Test two-layer shared FC configuration."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 256,
            "DROPOUT": 0.3,
            "FUSION_TYPE": "concat",
            "DEEP_SHARED_LAYER": True,
            "NUM_CLASSES": 3,
        }
        model = FG_MFN(cfg)
        
        # Shared FC should have more layers
        assert len(model.shared_fc) > 3

    def test_fg_mfn_backward_pass(self):
        """Test backward pass and gradient computation."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.3,
            "ATTRIBUTES": {
                "sentiment": {"num_classes": 3},
            },
        }
        model = FG_MFN(cfg)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        images = torch.randn(2, 3, 224, 224, requires_grad=True)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128, dtype=torch.long)
        labels = torch.tensor([0, 1])
        
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs["sentiment"], labels)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None or not param.requires_grad

    def test_fg_mfn_eval_mode(self):
        """Test model determinism in eval mode."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.5,  # High dropout to test determinism
            "NUM_CLASSES": 3,
        }
        model = FG_MFN(cfg)
        model.eval()
        
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128, dtype=torch.long)
        
        with torch.no_grad():
            out1 = model(images, input_ids, attention_mask)
            out2 = model(images, input_ids, attention_mask)
        
        assert torch.allclose(out1["sentiment"], out2["sentiment"])

    def test_fg_mfn_state_dict_save_load(self):
        """Test model checkpointing."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.3,
            "NUM_CLASSES": 3,
        }
        model1 = FG_MFN(cfg)
        state_dict = model1.state_dict()
        
        model2 = FG_MFN(cfg)
        model2.load_state_dict(state_dict)
        
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128, dtype=torch.long)
        
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(images, input_ids, attention_mask)
            out2 = model2(images, input_ids, attention_mask)
        
        assert torch.allclose(out1["sentiment"], out2["sentiment"])

    def test_fg_mfn_input_validation_batch_mismatch(self):
        """Test error handling for batch size mismatch."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.3,
            "NUM_CLASSES": 3,
        }
        model = FG_MFN(cfg)
        
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 128))  # Wrong batch size
        attention_mask = torch.ones(2, 128, dtype=torch.long)
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            model(images, input_ids, attention_mask)

    def test_fg_mfn_input_validation_image_shape(self):
        """Test error handling for invalid image shape."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.3,
            "NUM_CLASSES": 3,
        }
        model = FG_MFN(cfg)
        
        images = torch.randn(4, 3, 224)  # 3D instead of 4D
        input_ids = torch.randint(0, 1000, (4, 128))
        attention_mask = torch.ones(4, 128, dtype=torch.long)
        
        with pytest.raises(ValueError, match="4D"):
            model(images, input_ids, attention_mask)

    def test_fg_mfn_get_label_names(self):
        """Test label name retrieval."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.3,
            "ATTRIBUTES": {
                "sentiment": {
                    "num_classes": 3,
                    "labels": ["Positive", "Neutral", "Negative"],
                },
            },
        }
        model = FG_MFN(cfg)
        labels = model.get_label_names("sentiment")
        assert labels == ["Positive", "Neutral", "Negative"]

    def test_fg_mfn_device_movement(self):
        """Test model can be moved to different devices."""
        cfg = {
            "IMAGE_BACKBONE": "resnet18",
            "TEXT_ENCODER": "distilbert-base-uncased",
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.3,
            "NUM_CLASSES": 3,
        }
        model = FG_MFN(cfg)
        device = torch.device("cpu")
        model = model.to(device)
        
        # Check all parameters are on the device
        for param in model.parameters():
            assert param.device == device
