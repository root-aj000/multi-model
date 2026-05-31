"""
Visual encoder module using a CNN backbone.

Key design decision: VisualModule now outputs the backbone's NATIVE feature
dimension (e.g. 2048 for ResNet-50, 1280 for EfficientNet-B7) rather than
projecting down to HIDDEN_DIM here.

Why: Projecting 2048 → 512 inside the visual module discards information
before the cross-modal attention has a chance to use it. The projection to
HIDDEN_DIM now happens inside FG_MFN after cross-modal attention, so both
modalities contribute at full resolution.

The out_features parameter is kept for backward compatibility but defaults
to None (= native dimension). Set it only if you explicitly want a projection
at this stage (not recommended).

Config key: IMAGE_BACKBONE
Supported backbones and their native output dimensions:
    resnet18        →  512
    resnet50        → 2048
    efficientnet_b0 → 1280
    efficientnet_b3 → 1536
    efficientnet_b7 → 2560
    convnext_tiny   →  768
    convnext_small  →  768
    convnext_base   → 1024
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backbone registry
# native_dim: the in_features of the original final linear layer
# ---------------------------------------------------------------------------
_BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "constructor": torchvision.models.resnet18,
        "weights":     torchvision.models.ResNet18_Weights.DEFAULT,
        "fc_attr":     "fc",
        "native_dim":  512,
    },
    "resnet50": {
        "constructor": torchvision.models.resnet50,
        "weights":     torchvision.models.ResNet50_Weights.DEFAULT,
        "fc_attr":     "fc",
        "native_dim":  2048,
    },
    "efficientnet_b0": {
        "constructor": torchvision.models.efficientnet_b0,
        "weights":     torchvision.models.EfficientNet_B0_Weights.DEFAULT,
        "fc_attr":     "classifier",
        "native_dim":  1280,
    },
    "efficientnet_b3": {
        "constructor": torchvision.models.efficientnet_b3,
        "weights":     torchvision.models.EfficientNet_B3_Weights.DEFAULT,
        "fc_attr":     "classifier",
        "native_dim":  1536,
    },
    "efficientnet_b7": {
        "constructor": torchvision.models.efficientnet_b7,
        "weights":     torchvision.models.EfficientNet_B7_Weights.DEFAULT,
        "fc_attr":     "classifier",
        "native_dim":  2560,
    },
    "convnext_tiny": {
        "constructor": torchvision.models.convnext_tiny,
        "weights":     torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
        "fc_attr":     "classifier",
        "native_dim":  768,
    },
    "convnext_small": {
        "constructor": torchvision.models.convnext_small,
        "weights":     torchvision.models.ConvNeXt_Small_Weights.DEFAULT,
        "fc_attr":     "classifier",
        "native_dim":  768,
    },
    "convnext_base": {
        "constructor": torchvision.models.convnext_base,
        "weights":     torchvision.models.ConvNeXt_Base_Weights.DEFAULT,
        "fc_attr":     "classifier",
        "native_dim":  1024,
    },
}

SUPPORTED_BACKBONES = list(_BACKBONE_REGISTRY.keys())


class VisualModule(nn.Module):
    """
    Visual encoder that outputs native backbone features.

    The final classification head is removed so the backbone acts as a
    pure feature extractor. The output dimension equals the backbone's
    native penultimate-layer width (e.g. 2048 for ResNet-50).

    If out_features is specified, a Linear projection is added after the
    backbone. Leave it as None (default) to preserve the full native
    dimension for cross-modal attention.

    Attributes:
        native_dim  : int — the backbone's native output dimension
        out_features: int — actual output dimension (= native_dim if no projection)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        out_features: Optional[int] = None,
    ) -> None:
        """
        Args:
            backbone:     Backbone name. See SUPPORTED_BACKBONES.
            pretrained:   Load ImageNet-pretrained weights.
            out_features: If set, add a Linear(native_dim → out_features)
                          projection. Leave None to output native_dim.
        """
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Backbone '{backbone}' not supported. "
                f"Valid options: {SUPPORTED_BACKBONES}"
            )
        if out_features is not None and out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        self.backbone_name = backbone
        self.pretrained = pretrained

        entry = _BACKBONE_REGISTRY[backbone]
        self.native_dim: int = entry["native_dim"]

        # Load backbone and strip the final classification head
        self.backbone = self._load_and_strip(backbone, pretrained)

        # Optional projection layer (only if caller explicitly requests it)
        if out_features is not None and out_features != self.native_dim:
            self.projection: Optional[nn.Linear] = nn.Linear(self.native_dim, out_features)
            self.out_features: int = out_features
            logger.info(
                "VisualModule: %s native_dim=%d → projected to %d",
                backbone, self.native_dim, out_features,
            )
        else:
            self.projection = None
            self.out_features = self.native_dim
            logger.info(
                "VisualModule: %s native_dim=%d (no projection)",
                backbone, self.native_dim,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_strip(self, backbone: str, pretrained: bool) -> nn.Module:
        """
        Load the backbone and replace the final head with nn.Identity()
        so the model outputs penultimate-layer features at native_dim.
        """
        entry = _BACKBONE_REGISTRY[backbone]
        weights = entry["weights"] if pretrained else None
        model = entry["constructor"](weights=weights)
        fc_attr = entry["fc_attr"]
        head = getattr(model, fc_attr)

        if isinstance(head, nn.Linear):
            # ResNet: replace fc with Identity
            setattr(model, fc_attr, nn.Identity())

        elif isinstance(head, nn.Sequential):
            # EfficientNet / ConvNeXt: replace last Linear with Identity
            last_idx = len(head) - 1
            last_layer = head[last_idx]
            if not isinstance(last_layer, nn.Linear):
                raise RuntimeError(
                    f"Expected last layer of {fc_attr} to be nn.Linear, "
                    f"got {type(last_layer).__name__} for '{backbone}'"
                )
            head[last_idx] = nn.Identity()
        else:
            raise RuntimeError(
                f"Unrecognised head type {type(head).__name__} for '{backbone}'. "
                f"Add it to _BACKBONE_REGISTRY."
            )

        logger.debug("Stripped final head from %s; native_dim=%d", backbone, entry["native_dim"])
        return model

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters except the optional projection layer.

        The backbone body is frozen; the projection (if present) stays
        trainable so it can adapt to the frozen features.
        """
        frozen = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
            frozen += 1
        logger.info("VisualModule: backbone frozen (%d params)", frozen)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("VisualModule: backbone unfrozen")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.

        Returns:
            (B, out_features) feature tensor.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.shape}")

        features = self.backbone(x)

        # Some backbones return (B, C, 1, 1) after global pooling — flatten
        if features.dim() > 2:
            features = features.flatten(1)

        if self.projection is not None:
            features = self.projection(features)

        return features

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "pretrained": self.pretrained,
            "native_dim": self.native_dim,
            "out_features": self.out_features,
        }
