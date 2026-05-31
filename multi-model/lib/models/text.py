"""
Text encoder module using a transformer-based model.

Pooling strategies (set TEXT_POOLING in model_config.json):

  "cls"
      Use the [CLS] token representation.
      Fast. Standard for sentence classification with BERT-family models.
      Ignores word-level information beyond what the encoder bakes into [CLS].

  "mean"
      Average all non-padding token representations.
      Better than CLS for most tasks. Treats all tokens equally.
      Slight noise from low-importance tokens.

  "attention_weighted"  ← recommended
      Weight each token by its average self-attention score from the
      encoder's final layer, then take a weighted sum.

      Why better than mean:
        - Tokens the model already attends to strongly (e.g. "sale", "50%
          off", brand names) contribute more to the sentence vector.
        - Padding tokens get near-zero weight automatically.
        - No extra parameters — uses the encoder's own learned attention.

      Implementation:
        1. Run encoder with output_attentions=True
        2. Take attention weights from the last layer: (B, heads, seq, seq)
        3. Average across heads: (B, seq, seq)
        4. Sum over query dimension to get per-token importance: (B, seq)
        5. Zero out padding positions using attention_mask
        6. Normalise to sum=1 per sample
        7. Weighted sum of last_hidden_state: (B, native_dim)

Native encoder dimensions:
    distilbert-base-uncased → 768
    bert-base-uncased       → 768
    bert-large-uncased      → 1024
    roberta-base            → 768
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

SUPPORTED_POOLING = ("cls", "mean", "attention_weighted")
DEFAULT_POOLING = "attention_weighted"


class TextModule(nn.Module):
    """
    Text encoder that outputs native transformer features.

    Loads a pretrained transformer, pools token representations using the
    configured strategy, and optionally projects to a target dimension.

    Leave hidden_size=None (default) to output the encoder's native
    dimension (e.g. 768 for DistilBERT) for use in cross-modal attention.

    Attributes:
        native_dim  : int — the encoder's native hidden size
        out_features: int — actual output dimension (= native_dim if no projection)
        hidden_size : int — alias for out_features (backward compat)
    """

    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        hidden_size: Optional[int] = None,
        pooling: str = DEFAULT_POOLING,
    ) -> None:
        """
        Args:
            encoder_name: HuggingFace model identifier.
            hidden_size:  If set, add a Linear(native_dim → hidden_size)
                          projection after pooling. Leave None to output
                          native_dim (recommended — no lossy compression).
            pooling:      "cls", "mean", or "attention_weighted".
        """
        super().__init__()

        if not isinstance(encoder_name, str):
            raise TypeError(f"encoder_name must be str, got {type(encoder_name).__name__}")
        if hidden_size is not None and (not isinstance(hidden_size, int) or hidden_size <= 0):
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size}")
        if pooling not in SUPPORTED_POOLING:
            raise ValueError(
                f"pooling must be one of {SUPPORTED_POOLING}, got '{pooling}'"
            )

        self.encoder_name = encoder_name
        self.pooling = pooling

        # Load encoder.
        # attention_weighted pooling needs output_attentions=True at forward time,
        # but we don't set it here — we pass it per-call so the model can also
        # be used without attention output when pooling != "attention_weighted".
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.native_dim: int = self.encoder.config.hidden_size

        # Some transformer implementations (e.g. sdpa) cannot return
        # attention weights unless attention is run in eager mode.
        if self.pooling == "attention_weighted":
            if getattr(self.encoder.config, "attention_type", None) == "sdpa":
                logger.info(
                    "TextModule: encoder %s uses sdpa attention; switching "
                    "attn_implementation to 'eager' for attention_weighted pooling.",
                    encoder_name,
                )
                self.encoder.config.attn_implementation = "eager"
            self.encoder.config.output_attentions = True

        # Optional projection (only if caller explicitly requests it)
        if hidden_size is not None and hidden_size != self.native_dim:
            self.projection: Optional[nn.Linear] = nn.Linear(self.native_dim, hidden_size)
            self.out_features: int = hidden_size
            logger.info(
                "TextModule: %s native_dim=%d → projected to %d, pooling=%s",
                encoder_name, self.native_dim, hidden_size, pooling,
            )
        else:
            self.projection = None
            self.out_features = self.native_dim
            logger.info(
                "TextModule: %s native_dim=%d (no projection), pooling=%s",
                encoder_name, self.native_dim, pooling,
            )

        # Backward-compat alias (feature_extractor.py uses self.text_module.hidden_size)
        self.hidden_size = self.out_features

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len) token IDs.
            attention_mask: (B, seq_len) attention mask (1=real, 0=padding).

        Returns:
            (B, out_features) pooled feature tensor.
        """
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids {input_ids.shape} != attention_mask {attention_mask.shape}"
            )

        # Request attention weights only when needed (saves ~5% compute otherwise)
        need_attentions = (self.pooling == "attention_weighted")

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=need_attentions,
        )

        last_hidden = outputs.last_hidden_state  # (B, seq_len, native_dim)

        if self.pooling == "attention_weighted":
            if outputs.attentions is None:
                logger.warning(
                    "TextModule: encoder %s did not return attentions for "
                    "attention_weighted pooling; falling back to mean pooling.",
                    self.encoder_name,
                )
                pooled = self._mean_pool(last_hidden, attention_mask)
            else:
                pooled = self._attention_weighted_pool(
                    last_hidden, attention_mask, outputs.attentions
                )
        elif self.pooling == "mean":
            pooled = self._mean_pool(last_hidden, attention_mask)
        else:
            # "cls"
            pooled = last_hidden[:, 0, :]

        if self.projection is not None:
            pooled = self.projection(pooled)

        return pooled

    # ------------------------------------------------------------------
    # Pooling implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool over non-padding tokens.

        Args:
            last_hidden:    (B, seq_len, dim)
            attention_mask: (B, seq_len)  1=real token, 0=padding

        Returns:
            (B, dim)
        """
        mask = attention_mask.unsqueeze(-1).float()          # (B, seq_len, 1)
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @staticmethod
    def _attention_weighted_pool(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        attentions: Any,
    ) -> torch.Tensor:
        """
        Weight each token by its average self-attention score from the
        encoder's final transformer layer, then take a weighted sum.

        This uses the encoder's own learned attention — no extra parameters.
        Tokens the model already focuses on (keywords, entities, prices)
        contribute more to the sentence representation.

        Args:
            last_hidden:    (B, seq_len, dim)
            attention_mask: (B, seq_len)  1=real, 0=padding
            attentions:     Tuple of per-layer attention tensors.
                            Each tensor: (B, num_heads, seq_len, seq_len)

        Returns:
            (B, dim)
        """
        if attentions is None:
            raise ValueError(
                "attention_weighted pooling requires attention weights from the "
                "encoder. Ensure the model supports output_attentions=True or use "
                "pooling='mean' / pooling='cls'."
            )

        # Use the last layer's attention weights
        # Shape: (B, num_heads, seq_len, seq_len)
        last_layer_attn = attentions[-1]

        # Average across all attention heads → (B, seq_len, seq_len)
        avg_attn = last_layer_attn.mean(dim=1)

        # Sum over the query dimension to get per-token importance.
        # avg_attn[b, i, j] = how much token i attends to token j.
        # Summing over i gives: how much each token j is attended to overall.
        # Shape: (B, seq_len)
        token_importance = avg_attn.sum(dim=1)

        # Zero out padding positions so they don't contribute
        token_importance = token_importance * attention_mask.float()

        # Normalise to a probability distribution (sum = 1 per sample)
        # clamp prevents division by zero on all-padding sequences
        token_importance = token_importance / token_importance.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-9)

        # Weighted sum of token representations
        # (B, seq_len, 1) * (B, seq_len, dim) → sum over seq_len → (B, dim)
        pooled = (token_importance.unsqueeze(-1) * last_hidden).sum(dim=1)
        return pooled

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "native_dim": self.native_dim,
            "out_features": self.out_features,
            "pooling": self.pooling,
        }

    def get_cache_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"encoder_name": self.encoder_name}
        cache_dir = Path("local/tokenizer") / self.encoder_name
        if cache_dir.exists():
            info["cache_dir"] = str(cache_dir)
        return info
