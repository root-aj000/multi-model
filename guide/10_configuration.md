# 10 — Configuration

Complete configuration schema with all supported keys, defaults, and examples.

## Configuration File

The model configuration is stored in JSON format at [`configs/model/model_config.json`](../multi-model/configs/model/model_config.json).

### Full Example

```json
{
  "IMAGE_BACKBONE": "resnet18",
  "TEXT_ENCODER": "distilbert-base-uncased",
  "FUSION_TYPE": "concat",
  "DROPOUT": 0.3,
  "HIDDEN_DIM": 512,
  "FREEZE_BACKBONE": true,
  "DEEP_SHARED_LAYER": true,

  "ATTRIBUTES": {
    "theme": {
      "num_classes": 10,
      "labels": ["Food", "Fashion", "Tech", "Health", "Travel", "Finance", "Entertainment", "Sports", "Education", "Other"]
    },
    "sentiment": {
      "num_classes": 3,
      "labels": ["Positive", "Negative", "Neutral"]
    },
    "emotion": {
      "num_classes": 8,
      "labels": ["Excitement", "Trust", "Joy", "Fear", "Anger", "Sadness", "Surprise", "Anticipation"]
    },
    "dominant_colour": {
      "num_classes": 10,
      "labels": ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White", "Brown", "Multi"]
    },
    "attention_score": {
      "num_classes": 3,
      "labels": ["High", "Medium", "Low"]
    },
    "trust_safety": {
      "num_classes": 3,
      "labels": ["Safe", "Unsafe", "Questionable"]
    },
    "target_audience": {
      "num_classes": 8,
      "labels": ["General", "Food Lovers", "Tech Enthusiasts", "Fashionistas", "Parents", "Professionals", "Fitness Enthusiasts", "Students"]
    },
    "predicted_ctr": {
      "num_classes": 3,
      "labels": ["High", "Medium", "Low"]
    },
    "likelihood_shares": {
      "num_classes": 3,
      "labels": ["High", "Medium", "Low"]
    }
  }
}
```

---

## Model Configuration Keys

### Top-Level Keys

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `IMAGE_BACKBONE` | `str` | Yes | — | CNN backbone name: `"resnet18"` or `"resnet50"` |
| `TEXT_ENCODER` | `str` | Yes | — | HuggingFace model name: `"distilbert-base-uncased"` or `"bert-base-uncased"` |
| `FUSION_TYPE` | `str` | Yes | — | Feature fusion strategy: `"concat"` or `"add"` |
| `HIDDEN_DIM` | `int` | Yes | — | Hidden dimension for all sub-modules. Must be > 0. |
| `DROPOUT` | `float` | No | 0.0 | Dropout probability for shared layer. Range: [0.0, 1.0) |
| `FREEZE_BACKBONE` | `bool` | No | `false` | Whether to freeze CNN backbone weights during training |
| `DEEP_SHARED_LAYER` | `bool` | No | `false` | Use two-layer shared FC (true) or single-layer (false) |
| `ATTRIBUTES` | `dict` | Yes | — | Per-attribute classification head definitions |

### ATTRIBUTES Sub-Keys

Each key under `ATTRIBUTES` defines one classification head:

```json
"attribute_name": {
  "num_classes": 3,
  "labels": ["High", "Medium", "Low"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `num_classes` | `int` | Yes | Number of output classes. Must be > 0. |
| `labels` | `list[str]` | Yes | Human-readable label names, ordered by class index. Length must match `num_classes`. |

---

## Fusion Types

| Type | How it works | Output Shape | Parameters | When to use |
|------|-------------|--------------|------------|-------------|
| `concat` | Concatenate visual + text → Linear projection | `(B, HIDDEN_DIM)` | Yes (projection layer) | Default choice; preserves all info |
| `add` | Element-wise addition of visual + text | `(B, HIDDEN_DIM)` | No | When both features are pre-aligned |

---

## Training Configuration Keys

These keys are read by the training pipeline (in addition to the model keys above):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `batch_size` | `int` | 32 | Samples per training batch |
| `learning_rate` | `float` | 0.001 | Adam optimizer learning rate |
| `scheduler_step_size` | `int` | 10 | StepLR step size (epochs between LR decay) |
| `scheduler_gamma` | `float` | 0.1 | StepLR gamma (LR multiplication factor) |
| `epochs` | `int` | 100 | Total training epochs (overridable via CLI) |

### Training-Specific CLI Overrides

The training script accepts CLI arguments that override config values:

```bash
python -m scripts.train_model \
  --config configs/model/model_config.json \
  --epochs 50          # Overrides config["epochs"]
  --warmup-epochs 3    # Not in config; CLI only
  --log-dir local/logs # Not in config; CLI only
```

---

## OCR Configuration Keys

Read by the prediction pipeline:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ocr.engine` | `str` | `"easyocr"` | OCR engine name: `"easyocr"` or `"paddleocr"` |
| `ocr.model_dir` | `str` | `"local/ocr"` | Directory for OCR model files |

Example:

```json
{
  "ocr": {
    "engine": "easyocr",
    "model_dir": "local/ocr"
  }
}
```

---

## Label Maps (Alternative Format)

The [`get_label_maps()`](../multi-model/lib/utils/config.py:49) function supports two config formats:

### Format 1: Flat `label_maps` key

```json
{
  "label_maps": {
    "theme": ["Food", "Fashion", "Tech", "..."],
    "sentiment": ["Positive", "Negative", "Neutral"]
  }
}
```

### Format 2: Nested `ATTRIBUTES` key (recommended)

```json
{
  "ATTRIBUTES": {
    "theme": {
      "num_classes": 10,
      "labels": ["Food", "Fashion", "Tech", "..."]
    }
  }
}
```

Both formats can coexist; the function merges them. The `ATTRIBUTES` format is preferred because it also includes `num_classes` needed by the model constructor.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | `configs/model/model_config.json` | Path to config file (used by FastAPI app) |
| `ENV` | — | Set to `"dev"` to enable uvicorn auto-reload |

---

## Configuration Loading

### [`load_config()`](../multi-model/lib/utils/config.py:9)

```python
config = load_config("configs/model/model_config.json")
# Returns: Dict[str, Any]
```

- Validates file existence
- Parses JSON
- Raises `FileNotFoundError` if path doesn't exist
- Raises `ValueError` if JSON is malformed

### [`get_dataset_paths()`](../multi-model/lib/utils/config.py:32)

```python
paths = get_dataset_paths("train")
# {"csv": Path("dataset/train/train.csv"), "images": Path("dataset/train/images")}
```

### [`get_label_maps()`](../multi-model/lib/utils/config.py:49)

```python
label_maps = get_label_maps(config)
# {"theme": ["Food", "Fashion", ...], "sentiment": ["Positive", "Negative", "Neutral"], ...}
```

---

## Common Configurations

### Small Model (Fast Training)

```json
{
  "IMAGE_BACKBONE": "resnet18",
  "TEXT_ENCODER": "distilbert-base-uncased",
  "FUSION_TYPE": "add",
  "HIDDEN_DIM": 256,
  "DROPOUT": 0.2,
  "FREEZE_BACKBONE": true,
  "DEEP_SHARED_LAYER": false
}
```

### Large Model (Best Accuracy)

```json
{
  "IMAGE_BACKBONE": "resnet50",
  "TEXT_ENCODER": "bert-base-uncased",
  "FUSION_TYPE": "concat",
  "HIDDEN_DIM": 768,
  "DROPOUT": 0.3,
  "FREEZE_BACKBONE": false,
  "DEEP_SHARED_LAYER": true
}
```

### Inference-Only (No Training)

```json
{
  "IMAGE_BACKBONE": "resnet18",
  "TEXT_ENCODER": "distilbert-base-uncased",
  "FUSION_TYPE": "concat",
  "HIDDEN_DIM": 512,
  "DROPOUT": 0.3,
  "FREEZE_BACKBONE": true,
  "DEEP_SHARED_LAYER": true
}
```
