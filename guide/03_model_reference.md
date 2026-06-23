# 03 — Model Reference

Deep dive into the FG_MFN model architecture, sub-modules, fusion strategies, and factory methods.

## Overview

The **Fine-Grained Multi-Modal Fusion Network (FG_MFN)** is a PyTorch model that classifies ad creative images across 9 attributes simultaneously by fusing visual and text features.

**Architecture flow:**

```
Image ──→ VisualModule ──→ visual_features ──┐
                                              ├──→ _fuse_features() ──→ shared_layer ──→ 9× attribute heads
Text ───→ TextModule  ──→ text_features  ────┘
```

## FG_MFN Class

**Source:** [`lib/models/fg_mfn.py`](../multi-model/lib/models/fg_mfn.py)

```python
class FG_MFN(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None
    def forward(self, images, input_ids, attention_mask) -> Dict[str, Tensor]
    def _fuse_features(self, visual_features, text_features) -> Tensor
    def _validate_config(self, cfg) -> None
    def _build_shared_layer(self, hidden_dim, deep) -> nn.Sequential
    def _create_multi_attribute_heads(self, cfg) -> None
    def _create_legacy_sentiment_head(self, cfg) -> None
    def _validate_inputs(self, images, input_ids, attention_mask) -> None
    def get_label_names(self, attr_name) -> Optional[List[str]]
```

### Constructor — [`__init__()`](../multi-model/lib/models/fg_mfn.py:53)

Builds the full model from a configuration dictionary:

| Step | Component | Config Key |
|------|-----------|------------|
| 1 | Validate config | (all keys checked) |
| 2 | Create [`VisualModule`](../multi-model/lib/models/visual.py:27) | `IMAGE_BACKBONE`, `HIDDEN_DIM`, `FREEZE_BACKBONE` |
| 3 | Create [`TextModule`](../multi-model/lib/models/text.py:28) | `TEXT_ENCODER`, `HIDDEN_DIM` |
| 4 | Build fusion layer | `FUSION_TYPE`, `HIDDEN_DIM` |
| 5 | Build shared FC layer | `HIDDEN_DIM`, `DROPOUT`, `DEEP_SHARED_LAYER` |
| 6 | Create 9 attribute heads | `ATTRIBUTES` |

**Raises:**
- `TypeError` — if `cfg` is not a dictionary
- `ValueError` — if required keys are missing, `HIDDEN_DIM ≤ 0`, or `FUSION_TYPE` is unsupported

### Forward Pass — [`forward()`](../multi-model/lib/models/fg_mfn.py:259)

```python
outputs = model(images, input_ids, attention_mask)
# outputs = {
#     "theme":          Tensor(B, 10),
#     "sentiment":      Tensor(B, 3),
#     "emotion":        Tensor(B, 8),
#     "dominant_colour": Tensor(B, 10),
#     "attention_score": Tensor(B, 3),
#     "trust_safety":   Tensor(B, 3),
#     "target_audience": Tensor(B, 8),
#     "predicted_ctr":  Tensor(B, 3),
#     "likelihood_shares": Tensor(B, 3),
# }
```

**Input shapes:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `images` | `(B, 3, 224, 224)` | Batch of RGB images |
| `input_ids` | `(B, seq_len)` | Tokenized text IDs |
| `attention_mask` | `(B, seq_len)` | Text attention mask |

**Internal flow:**

1. `_validate_inputs()` — shape and type checks
2. `visual_module(images)` → `Tensor(B, HIDDEN_DIM)`
3. `text_module(input_ids, attention_mask)` → `Tensor(B, HIDDEN_DIM)`
4. `_fuse_features(visual, text)` → `Tensor(B, HIDDEN_DIM)`
5. `shared_layer(fused)` → `Tensor(B, HIDDEN_DIM)` (ReLU + Dropout)
6. For each attribute head: `Linear(HIDDEN_DIM, num_classes)` → `Tensor(B, num_classes)`

### Feature Fusion — [`_fuse_features()`](../multi-model/lib/models/fg_mfn.py:294)

Two fusion strategies controlled by `FUSION_TYPE`:

| Strategy | Operation | Output Shape | Parameters |
|----------|-----------|--------------|------------|
| `concat` | `torch.cat([visual, text], dim=1)` → `Linear(2×HIDDEN_DIM, HIDDEN_DIM)` | `(B, HIDDEN_DIM)` | Yes (projection layer) |
| `add` | `visual + text` | `(B, HIDDEN_DIM)` | No (element-wise) |

**`concat`** preserves all information from both modalities but requires a learned projection. **`add`** is parameter-free but assumes both feature spaces are already aligned.

### Config Validation — [`_validate_config()`](../multi-model/lib/models/fg_mfn.py:124)

Checks the following before any expensive operations:

| Check | Error Type |
|-------|------------|
| `cfg` is a dict | `TypeError` |
| Required keys exist: `IMAGE_BACKBONE`, `TEXT_ENCODER`, `HIDDEN_DIM`, `FUSION_TYPE`, `ATTRIBUTES` | `ValueError` |
| `HIDDEN_DIM > 0` | `ValueError` |
| Each attribute has `num_classes` (int > 0) and `labels` (list) | `ValueError` |

### Shared Layer — [`_build_shared_layer()`](../multi-model/lib/models/fg_mfn.py:172)

| Mode | Layers | When |
|------|--------|------|
| Deep (`DEEP_SHARED_LAYER=true`) | `Linear → ReLU → Dropout → Linear → ReLU → Dropout` | Better feature abstraction |
| Shallow | `Linear → ReLU → Dropout` | Lighter, faster |

### Attribute Heads — [`_create_multi_attribute_heads()`](../multi-model/lib/models/fg_mfn.py:211)

Creates a `nn.ModuleDict` named `attribute_heads` with one `nn.Linear(HIDDEN_DIM, num_classes)` per attribute. The 9 attributes are:

| Attribute | Classes | Labels |
|-----------|---------|--------|
| `theme` | 10 | Food, Fashion, Tech, Health, Travel, Finance, Entertainment, Sports, Education, Other |
| `sentiment` | 3 | Positive, Negative, Neutral |
| `emotion` | 8 | Excitement, Trust, Joy, Fear, Anger, Sadness, Surprise, Anticipation |
| `dominant_colour` | 10 | Red, Blue, Green, Yellow, Orange, Purple, Black, White, Brown, Multi |
| `attention_score` | 3 | High, Medium, Low |
| `trust_safety` | 3 | Safe, Unsafe, Questionable |
| `target_audience` | 8 | General, Food Lovers, Tech Enthusiasts, Fashionistas, Parents, Professionals, Fitness Enthusiasts, Students |
| `predicted_ctr` | 3 | High, Medium, Low |
| `likelihood_shares` | 3 | High, Medium, Low |

---

## TextModule

**Source:** [`lib/models/text.py`](../multi-model/lib/models/text.py)

```python
class TextModule(nn.Module):
    def __init__(self, encoder_name="distilbert-base-uncased", hidden_size=768)
    def forward(self, input_ids, attention_mask) -> Tensor(B, hidden_size)
    def get_model_info() -> Dict
    def get_cache_info() -> Dict
```

### How it works

1. Loads a pretrained transformer via `AutoModel.from_pretrained(encoder_name)`
2. On forward pass, extracts the **[CLS] token** representation: `last_hidden_state[:, 0, :]`
3. Projects through `nn.Linear(encoder_hidden_size, hidden_size)` if dimensions differ

**Supported encoders:**

| Name | Native Hidden Size |
|------|--------------------|
| `distilbert-base-uncased` | 768 |
| `bert-base-uncased` | 768 |

If `hidden_size == encoder_hidden_size`, the projection layer is still created but acts as an identity-like transform (the linear layer is initialized randomly, not skipped).

---

## VisualModule

**Source:** [`lib/models/visual.py`](../multi-model/lib/models/visual.py)

```python
class VisualModule(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, out_features=512)
    def forward(self, x) -> Tensor(B, out_features)
    def freeze_backbone(self)
    def unfreeze_backbone(self)
    def get_model_info() -> Dict
```

### How it works

1. Loads a pretrained CNN from torchvision (ResNet-18 or ResNet-50)
2. Replaces the final FC layer: `nn.Linear(1000, out_features)`
3. On forward pass, the image goes through all layers including the replaced FC

**Supported backbones:**

| Backbone | Pre-FC Features | Default Weights |
|----------|----------------|-----------------|
| `resnet18` | 512 | ResNet18_Weights.DEFAULT |
| `resnet50` | 2048 | ResNet50_Weights.DEFAULT |

### Freezing

- [`freeze_backbone()`](../multi-model/lib/models/visual.py:112) — sets `requires_grad=False` on all backbone parameters. Only the replaced FC layer is trained. Useful for fine-tuning on small datasets.
- [`unfreeze_backbone()`](../multi-model/lib/models/visual.py:120) — sets `requires_grad=True` on all parameters. Used for full fine-tuning.

---

## Model Factory

**Source:** [`lib/models/factory.py`](../multi-model/lib/models/factory.py)

### [`create_model()`](../multi-model/lib/models/factory.py:60)

Creates a fresh (randomly initialized) FG_MFN model from configuration:

```python
model = create_model(cfg, device)  # Returns FG_MFN on device in eval mode
```

### [`load_model()`](../multi-model/lib/models/factory.py:78)

Restores a model from a checkpoint file:

```python
model = load_model(cfg, device, "saved_models/best_model_epoch_5.pt")
```

**Checkpoint loading strategy:**
1. First tries `torch.load(path, weights_only=True)` (safe loading, PyTorch ≥2.0)
2. Falls back to `weights_only=False` for older checkpoint formats
3. Calls `model.load_state_dict(checkpoint)` — raises `RuntimeError` if architecture mismatch

### [`load_tokenizer()`](../multi-model/lib/models/factory.py:36)

Loads a HuggingFace tokenizer for the text encoder:

```python
tokenizer = load_tokenizer("distilbert-base-uncased")
```

Warns if the encoder name is not in the known supported list but attempts to load anyway.

---

## Parameter Count Reference

Approximate parameter counts for common configurations:

| Configuration | Total Params | Trainable (frozen backbone) |
|--------------|-------------|---------------------------|
| ResNet-18 + DistilBERT, HIDDEN_DIM=512 | ~67M | ~24M |
| ResNet-50 + DistilBERT, HIDDEN_DIM=512 | ~83M | ~28M |

Use the analysis script for exact counts:

```bash
python -m scripts.analyze_model --config configs/model/model_config.json
```
