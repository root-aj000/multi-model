"""
Fine-Grained Multi-Modal Fusion Network (FG_MFN).

Architecture overview
─────────────────────
Image  → VisualModule  → v  (B, visual_dim)   e.g. 2048 for ResNet-50
Text   → TextModule    → t  (B, text_dim)     e.g.  768 for DistilBERT

Both modalities keep their NATIVE dimensions through the encoders.
No information is discarded before the fusion stage.

Fusion (set FUSION_TYPE in config)
───────────────────────────────────
  "concat"
      [v ; t] → Linear(visual_dim + text_dim, HIDDEN_DIM) → shared_fc

  "add"
      Requires visual_dim == text_dim.
      v + t → Linear(dim, HIDDEN_DIM) → shared_fc

  "attention"  ← recommended
      Full cross-modal multi-head attention:

      Step 1 — project both modalities to a common attention dimension
               (ATTENTION_DIM, default = HIDDEN_DIM):
               v_proj = Linear(visual_dim, ATTENTION_DIM)   → (B, ATTENTION_DIM)
               t_proj = Linear(text_dim,  ATTENTION_DIM)    → (B, ATTENTION_DIM)

      Step 2 — cross-modal attention in both directions:
               v_attended = MHA(query=v_proj, key=t_proj, value=t_proj)
               t_attended = MHA(query=t_proj, key=v_proj, value=v_proj)
               Each MHA uses ATTENTION_HEADS heads.

      Step 3 — residual connections + layer norm:
               v_out = LayerNorm(v_proj + v_attended)
               t_out = LayerNorm(t_proj + t_attended)

      Step 4 — fuse and project to HIDDEN_DIM:
               fused = [v_out ; t_out]  → Linear(2*ATTENTION_DIM, HIDDEN_DIM)

      Step 5 — shared_fc → per-attribute heads

Config keys (model_config.json)
────────────────────────────────
  IMAGE_BACKBONE    backbone name (resnet50, efficientnet_b3, …)
  TEXT_ENCODER      HuggingFace model name
  TEXT_POOLING      "cls" or "mean"
  FUSION_TYPE       "concat", "add", or "attention"
  HIDDEN_DIM        output dimension of shared FC and classification heads
  ATTENTION_DIM     internal dimension for cross-modal attention
                    (only used when FUSION_TYPE = "attention")
                    defaults to HIDDEN_DIM
  ATTENTION_HEADS   number of MHA heads
                    (only used when FUSION_TYPE = "attention")
                    defaults to 8; must divide ATTENTION_DIM evenly
  DROPOUT           dropout probability
  FREEZE_BACKBONE   freeze visual backbone convolutional weights
  DEEP_SHARED_LAYER use two-layer shared FC
  ATTRIBUTES        per-attribute num_classes and labels
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.text import TextModule
from lib.models.visual import VisualModule

logger = logging.getLogger(__name__)

ATTRIBUTE_NAMES = [
    "theme", "sentiment", "emotion", "dominant_colour",
    "attention_score", "trust_safety", "target_audience",
    "predicted_ctr", "likelihood_shares",
]

DEFAULT_NUM_CLASSES = 2
SUPPORTED_FUSION_TYPES = ("concat", "add", "attention")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Modal Attention module
# ──────────────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention between visual and text features.

    Both modalities are first projected to a common ATTENTION_DIM space,
    then attend to each other using multi-head attention (MHA). Residual
    connections and layer normalisation are applied after each attention
    step. The attended representations are concatenated and projected to
    HIDDEN_DIM.

    Why bidirectional?
        Visual → Text attention: "which text tokens are relevant to this image?"
        Text → Visual attention: "which visual features support this text?"
    Both directions are computed in parallel and contribute to the final
    fused representation.

    Dimensions:
        Input:  v (B, visual_dim),  t (B, text_dim)
        Output: (B, HIDDEN_DIM)
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        """
        Args:
            visual_dim:   Native visual feature dimension (e.g. 2048).
            text_dim:     Native text feature dimension (e.g. 768).
            hidden_dim:   Output dimension after fusion projection.
            attention_dim: Internal dimension for Q/K/V projections.
                           Must be divisible by num_heads.
            num_heads:    Number of attention heads.
            dropout:      Dropout probability applied after attention.
        """
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"ATTENTION_DIM ({attention_dim}) must be divisible by "
                f"ATTENTION_HEADS ({num_heads})."
            )

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        # Project each modality to the shared attention space
        self.visual_proj = nn.Linear(visual_dim, attention_dim)
        self.text_proj   = nn.Linear(text_dim,   attention_dim)

        # Cross-modal MHA: visual attends to text
        self.v2t_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # input shape: (B, seq, dim)
        )
        # Cross-modal MHA: text attends to visual
        self.t2v_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms for residual connections
        self.v_norm = nn.LayerNorm(attention_dim)
        self.t_norm = nn.LayerNorm(attention_dim)

        # Dropout after attention
        self.attn_dropout = nn.Dropout(dropout)

        # Final projection: [v_out ; t_out] → HIDDEN_DIM
        self.fusion_proj = nn.Linear(attention_dim * 2, hidden_dim)

        logger.info(
            "CrossModalAttention: visual_dim=%d, text_dim=%d → "
            "attention_dim=%d, heads=%d → hidden_dim=%d",
            visual_dim, text_dim, attention_dim, num_heads, hidden_dim,
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual: (B, visual_dim)
            text:   (B, text_dim)

        Returns:
            (B, hidden_dim) fused representation.
        """
        # Project to shared attention space
        # MHA expects (B, seq_len, dim); our features are single vectors,
        # so seq_len = 1 for each modality.
        v = self.visual_proj(visual).unsqueeze(1)   # (B, 1, attention_dim)
        t = self.text_proj(text).unsqueeze(1)       # (B, 1, attention_dim)

        # Visual attends to text: query=v, key/value=t
        v_attended, _ = self.v2t_attn(query=v, key=t, value=t)  # (B, 1, attention_dim)
        v_attended = self.attn_dropout(v_attended)

        # Text attends to visual: query=t, key/value=v
        t_attended, _ = self.t2v_attn(query=t, key=v, value=v)  # (B, 1, attention_dim)
        t_attended = self.attn_dropout(t_attended)

        # Residual + LayerNorm
        v_out = self.v_norm(v + v_attended).squeeze(1)  # (B, attention_dim)
        t_out = self.t_norm(t + t_attended).squeeze(1)  # (B, attention_dim)

        # Concatenate and project to HIDDEN_DIM
        fused = torch.cat([v_out, t_out], dim=1)        # (B, 2 * attention_dim)
        return self.fusion_proj(fused)                   # (B, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# FG_MFN
# ──────────────────────────────────────────────────────────────────────────────

class FG_MFN(nn.Module):
    """
    Fine-Grained Multi-Modal Fusion Network.

    See module docstring for full architecture description.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self._validate_config(cfg)

        fusion_type = cfg.get("FUSION_TYPE", "attention")
        if fusion_type not in SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"FUSION_TYPE '{fusion_type}' not supported. "
                f"Valid: {SUPPORTED_FUSION_TYPES}"
            )

        self.cfg = cfg
        self.fusion_type = fusion_type
        hidden_dim = cfg["HIDDEN_DIM"]
        dropout = cfg["DROPOUT"]

        # ── Visual encoder (outputs native backbone dimension) ──────────────
        self.visual_module = VisualModule(
            backbone=cfg["IMAGE_BACKBONE"],
            pretrained=True,
            out_features=None,   # keep native dim — no lossy projection here
        )
        visual_dim = self.visual_module.native_dim

        if cfg.get("FREEZE_BACKBONE", False):
            self.visual_module.freeze_backbone()

        logger.info(
            "Visual: backbone=%s, native_dim=%d", cfg["IMAGE_BACKBONE"], visual_dim
        )

        # ── Text encoder (outputs native encoder dimension) ─────────────────
        self.text_module = TextModule(
            encoder_name=cfg["TEXT_ENCODER"],
            hidden_size=None,    # keep native dim — no lossy projection here
            pooling=cfg.get("TEXT_POOLING", "mean"),
        )
        text_dim = self.text_module.native_dim

        logger.info(
            "Text: encoder=%s, native_dim=%d, pooling=%s",
            cfg["TEXT_ENCODER"], text_dim, cfg.get("TEXT_POOLING", "mean"),
        )

        # ── Fusion layer ────────────────────────────────────────────────────
        if fusion_type == "attention":
            attention_dim = cfg.get("ATTENTION_DIM", hidden_dim)
            num_heads = cfg.get("ATTENTION_HEADS", 8)

            self.fusion = CrossModalAttention(
                visual_dim=visual_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                attention_dim=attention_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            fusion_out_dim = hidden_dim  # CrossModalAttention already projects to hidden_dim

        elif fusion_type == "concat":
            # Project concatenated native features to HIDDEN_DIM
            self.fusion_proj = nn.Linear(visual_dim + text_dim, hidden_dim)
            fusion_out_dim = hidden_dim
            logger.info(
                "Concat fusion: (%d + %d) → %d", visual_dim, text_dim, hidden_dim
            )

        else:  # "add"
            if visual_dim != text_dim:
                raise ValueError(
                    f"FUSION_TYPE='add' requires visual_dim == text_dim, "
                    f"but got visual_dim={visual_dim}, text_dim={text_dim}. "
                    f"Use 'concat' or 'attention' instead, or choose a backbone "
                    f"whose native_dim matches the text encoder's native_dim."
                )
            self.fusion_proj = nn.Linear(visual_dim, hidden_dim)
            fusion_out_dim = hidden_dim
            logger.info("Add fusion: %d → %d", visual_dim, hidden_dim)

        # ── Shared FC after fusion ──────────────────────────────────────────
        self.shared_fc = self._build_shared_layer(
            fusion_out_dim, hidden_dim, dropout,
            cfg.get("DEEP_SHARED_LAYER", False),
        )

        # ── Per-attribute classification heads ──────────────────────────────
        self.attribute_heads = nn.ModuleDict()
        if "ATTRIBUTES" in cfg:
            self._create_multi_attribute_heads(cfg, hidden_dim)
        else:
            logger.warning("No ATTRIBUTES in config — using legacy single-head mode")
            self._create_legacy_head(cfg, hidden_dim)

        logger.info(
            "FG_MFN ready | fusion=%s | visual_dim=%d | text_dim=%d | "
            "hidden_dim=%d | heads=%d",
            fusion_type, visual_dim, text_dim, hidden_dim,
            len(self.attribute_heads),
        )

    # ── Config validation ───────────────────────────────────────────────────

    def _validate_config(self, cfg: Any) -> None:
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a dict, got {type(cfg).__name__}")
        required = ["IMAGE_BACKBONE", "TEXT_ENCODER", "HIDDEN_DIM", "DROPOUT"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(f"Config missing required keys: {missing}")
        if not isinstance(cfg["HIDDEN_DIM"], int) or cfg["HIDDEN_DIM"] <= 0:
            raise ValueError(f"HIDDEN_DIM must be a positive int, got {cfg['HIDDEN_DIM']}")
        if not isinstance(cfg["DROPOUT"], (int, float)) or not 0 <= cfg["DROPOUT"] < 1:
            raise ValueError(f"DROPOUT must be in [0, 1), got {cfg['DROPOUT']}")

    # ── Architecture builders ───────────────────────────────────────────────

    def _build_shared_layer(
        self, in_dim: int, hidden_dim: int, dropout: float, use_deep: bool,
    ) -> nn.Sequential:
        if use_deep:
            layer = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout / 2),
            )
            logger.info("Deep shared FC: %d → %d → %d", in_dim, hidden_dim, hidden_dim)
        else:
            layer = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            logger.info("Shared FC: %d → %d", in_dim, hidden_dim)
        return layer

    def _create_multi_attribute_heads(self, cfg: Dict[str, Any], hidden_dim: int) -> None:
        created = 0
        for attr_name in ATTRIBUTE_NAMES:
            if attr_name not in cfg["ATTRIBUTES"]:
                continue
            attr_cfg = cfg["ATTRIBUTES"][attr_name]
            num_classes = attr_cfg.get("num_classes")
            if not isinstance(num_classes, int) or num_classes <= 0:
                logger.warning("Skipping '%s': invalid num_classes=%s", attr_name, num_classes)
                continue
            # Task-specific two-layer head: each attribute gets its own private
            # MLP so gradient conflicts between heads are minimised.
            # The shared_fc output (hidden_dim) is projected per-attribute
            # before the final classification layer.
            head_hidden = max(hidden_dim // 2, num_classes * 4)
            self.attribute_heads[attr_name] = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(head_hidden, num_classes),
            )
            created += 1
            logger.info("Head '%s': %d → %d → %d classes", attr_name, hidden_dim, head_hidden, num_classes)
        if created == 0:
            raise ValueError("No attribute heads created. Check ATTRIBUTES in config.")

    def _create_legacy_head(self, cfg: Dict[str, Any], hidden_dim: int) -> None:
        num_classes = cfg.get("NUM_CLASSES", DEFAULT_NUM_CLASSES)
        self.attribute_heads["sentiment"] = nn.Linear(hidden_dim, num_classes)
        logger.info("Legacy head: %d classes", num_classes)

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images:         (B, C, H, W)
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)

        Returns:
            Dict[attr_name → (B, num_classes) logit tensor]
        """
        self._validate_inputs(images, input_ids, attention_mask)

        # Encode each modality at full native resolution
        visual_features = self.visual_module(images)   # (B, visual_dim)
        text_features   = self.text_module(input_ids, attention_mask)  # (B, text_dim)

        # Fuse
        fused = self._fuse(visual_features, text_features)  # (B, hidden_dim)

        # Shared FC
        shared = self.shared_fc(fused)  # (B, hidden_dim)

        # Per-attribute classification
        return {name: head(shared) for name, head in self.attribute_heads.items()}

    def _fuse(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual and text features according to FUSION_TYPE.

        attention : CrossModalAttention (bidirectional MHA) → (B, hidden_dim)
        concat    : [v ; t] → Linear → (B, hidden_dim)
        add       : v + t   → Linear → (B, hidden_dim)
        """
        if self.fusion_type == "attention":
            return self.fusion(visual, text)

        if self.fusion_type == "concat":
            return self.fusion_proj(torch.cat([visual, text], dim=1))

        # "add"
        return self.fusion_proj(visual + text)

    # ── Input validation ────────────────────────────────────────────────────

    def _validate_inputs(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if images.dim() != 4:
            raise ValueError(f"images must be 4D (B,C,H,W), got {images.shape}")
        if images.size(1) not in (1, 3, 4):
            raise ValueError(f"images channel dim must be 1/3/4, got {images.size(1)}")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D (B, seq_len), got {input_ids.shape}")
        if images.size(0) != input_ids.size(0):
            raise ValueError(
                f"Batch size mismatch: images={images.size(0)}, "
                f"input_ids={input_ids.size(0)}"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask {attention_mask.shape} != input_ids {input_ids.shape}"
            )

    # ── Utilities ───────────────────────────────────────────────────────────

    def get_label_names(self, attr_name: str) -> Optional[List[str]]:
        if "ATTRIBUTES" not in self.cfg:
            return None
        labels = self.cfg["ATTRIBUTES"].get(attr_name, {}).get("labels")
        return labels

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of the model dimensions."""
        return {
            "backbone": self.cfg["IMAGE_BACKBONE"],
            "visual_native_dim": self.visual_module.native_dim,
            "text_encoder": self.cfg["TEXT_ENCODER"],
            "text_native_dim": self.text_module.native_dim,
            "fusion_type": self.fusion_type,
            "hidden_dim": self.cfg["HIDDEN_DIM"],
            "attention_dim": self.cfg.get("ATTENTION_DIM", self.cfg["HIDDEN_DIM"]),
            "attention_heads": self.cfg.get("ATTENTION_HEADS", 8),
            "num_attribute_heads": len(self.attribute_heads),
        }
