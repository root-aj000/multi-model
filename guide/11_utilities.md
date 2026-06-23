# 11 — Utilities

Cache management, logging, lifecycle, and configuration loading utilities.

## Configuration Utilities — [`lib/utils/config.py`](../multi-model/lib/utils/config.py)

### [`load_config()`](../multi-model/lib/utils/config.py:9)

Loads and parses a JSON configuration file:

```python
config = load_config("configs/model/model_config.json")
```

| Behavior | Detail |
|----------|--------|
| **Input** | `config_path: str` |
| **Output** | `Dict[str, Any]` |
| **Raises** | `FileNotFoundError` — if file doesn't exist |
| **Raises** | `ValueError` — if JSON is malformed |

### [`get_dataset_paths()`](../multi-model/lib/utils/config.py:32)

Resolves dataset file paths for a given split:

```python
paths = get_dataset_paths("train")
# {"csv": Path("dataset/train/train.csv"), "images": Path("dataset/train/images")}
```

| Split | CSV Path | Image Directory |
|-------|----------|-----------------|
| `train` | `dataset/train/train.csv` | `dataset/train/images/` |
| `val` | `dataset/val/val.csv` | `dataset/val/images/` |
| `test` | `dataset/test/test.csv` | `dataset/test/images/` |

### [`get_label_maps()`](../multi-model/lib/utils/config.py:49)

Extracts label-to-name mappings from config. Supports two formats:

```python
# Format 1: flat "label_maps" key
label_maps = get_label_maps({"label_maps": {"theme": ["Food", "Fashion"]}})

# Format 2: nested "ATTRIBUTES" key
label_maps = get_label_maps({"ATTRIBUTES": {"theme": {"num_classes": 2, "labels": ["Food", "Fashion"]}}})

# Both formats can coexist; results are merged
```

---

## Lifecycle Utilities — [`lib/utils/lifecycle.py`](../multi-model/lib/utils/lifecycle.py)

### [`setup_directories()`](../multi-model/lib/utils/lifecycle.py:7)

Creates required directories at application startup:

```python
setup_directories()  # Creates defaults
setup_directories([Path("custom/dir")])  # Creates custom list
```

**Default directories:**

| Directory | Purpose |
|-----------|---------|
| `uploads/` | Uploaded image files |
| `saved_models/` | Model checkpoints |
| `local/models/` | Cached visual model weights |
| `local/ocr/` | OCR model files |
| `local/cache/` | General cache storage |

Uses `mkdir(parents=True, exist_ok=True)` — safe to call multiple times.

### [`cleanup_upload_directory()`](../multi-model/lib/utils/lifecycle.py:26)

Removes all files and subdirectories from the upload directory:

```python
cleanup_upload_directory(Path("uploads/"))
```

- Silently returns if directory doesn't exist
- Deletes files with `item.unlink()`
- Deletes subdirectories with `shutil.rmtree()`
- Called during FastAPI shutdown to prevent disk fill-up

---

## Cache Management — [`lib/utils/cache.py`](../multi-model/lib/utils/cache.py)

### Standalone Functions

#### Visual Model Cache

| Function | Signature | Returns |
|----------|-----------|---------|
| [`get_model_cache_directory()`](../multi-model/lib/utils/cache.py:7) | `() → Path` | `Path("local/models")` |
| [`list_cached_models()`](../multi-model/lib/utils/cache.py:12) | `() → List[str]` | Names of cached model directories |
| [`get_cache_size()`](../multi-model/lib/utils/cache.py:20) | `() → float` | Total visual cache size in MB |
| [`clear_model_cache()`](../multi-model/lib/utils/cache.py:31) | `(confirm=False) → bool` | Delete visual cache (requires `confirm=True`) |

#### Text Model Cache

| Function | Signature | Returns |
|----------|-----------|---------|
| [`get_text_model_cache_directory()`](../multi-model/lib/utils/cache.py:41) | `() → Path` | `Path("local/tokenizer")` |
| [`list_cached_text_models()`](../multi-model/lib/utils/cache.py:46) | `() → List[str]` | Names of cached tokenizer directories |
| [`get_text_cache_size()`](../multi-model/lib/utils/cache.py:54) | `() → float` | Total text cache size in MB |
| [`clear_text_model_cache()`](../multi-model/lib/utils/cache.py:65) | `(confirm=False) → bool` | Delete text cache (requires `confirm=True`) |

#### Combined Operations

| Function | Signature | Returns |
|----------|-----------|---------|
| [`get_cache_info()`](../multi-model/lib/utils/cache.py:100) | `() → Dict` | `{"visual_cache_size_mb": X, "text_cache_size_mb": Y}` |
| [`clear_cache()`](../multi-model/lib/utils/cache.py:113) | `(engine=None, confirm=False) → bool` | Clear specific or all caches |

**`clear_cache()` engine parameter:**

| Value | Effect |
|-------|--------|
| `"visual"` | Clear only visual model cache |
| `"text"` | Clear only text model cache |
| `None` | Clear both caches |

#### Utility Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| [`get_dir_size()`](../multi-model/lib/utils/cache.py:75) | `(path) → float` | Compute directory size in bytes (handles permission errors gracefully) |
| [`count_files()`](../multi-model/lib/utils/cache.py:134) | `(path) → int` | Count files in a directory tree |

### CacheManager Class

#### [`CacheManager`](../multi-model/lib/utils/cache.py:141)

Manages multiple named cache directories:

```python
manager = CacheManager({
    "visual": Path("local/models"),
    "text": Path("local/tokenizer"),
    "ocr": Path("local/ocr"),
})

info = manager.info()
# {"visual": 5242880.0, "text": 2097152.0, "ocr": 10485760.0}
# (sizes in bytes; only existing directories are included)
```

---

## Logging Utilities — [`lib/utils/logging.py`](../multi-model/lib/utils/logging.py)

### TrainingLogger Class

#### [`TrainingLogger`](../multi-model/lib/utils/logging.py:11)

CSV-based training metrics logger:

```python
logger = TrainingLogger("local/logs")
logger.log_metrics({"train_loss": 0.5, "val_accuracy": 0.75}, epoch=1)
logger.log_metrics({"train_loss": 0.3, "val_accuracy": 0.82}, epoch=2)
logger.close()
```

| Method | Description |
|--------|-------------|
| [`__init__(training_log_dir)`](../multi-model/lib/utils/logging.py:16) | Create log directory and set up file path |
| [`log_metrics(metrics_dict, epoch)`](../multi-model/lib/utils/logging.py:30) | Write one row of metrics; creates file with header on first call |
| [`close()`](../multi-model/lib/utils/logging.py:66) | Close the file handle |

**Behavior details:**

- First call to `log_metrics()` creates the CSV file with sorted column headers
- Subsequent calls append rows
- Each row is flushed to disk immediately (`csv_file.flush()`)
- If the CSV already exists with content, headers are not re-written (preserves existing logs)
- Raises `TypeError` if metrics is not a dict

**CSV format:**

```csv
epoch,train_accuracy,train_loss,val_accuracy,val_loss
1,0.45,1.2,0.40,1.3
2,0.55,0.9,0.50,1.0
```

### create_logger Function

#### [`create_logger()`](../multi-model/lib/utils/logging.py:74)

Creates a named Python logger with optional file handler:

```python
logger = create_logger("training", logging.INFO, log_dir="local/logs")
# Creates: local/logs/training.log
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Logger name |
| `level` | `int` | Logging level (e.g., `logging.INFO`) |
| `log_dir` | `Optional[str]` | Directory for file handler (None = console only) |

**Features:**
- Prevents duplicate handlers (checks `logger.handlers` before adding)
- Console handler always added
- File handler added only if `log_dir` is provided
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

---

## Services

### Predictor — [`lib/services/predictor.py`](../multi-model/lib/services/predictor.py)

#### [`Predictor`](../multi-model/lib/services/predictor.py:18)

Orchestrates model inference and logit-to-label conversion:

```python
predictor = Predictor(model, label_maps)
result = predictor.predict_single(images, input_ids, attention_mask)
# {"theme": "Food", "theme_confidence": 0.92, "theme_predicted_label_num": 0, ...}
```

| Method | Input | Output |
|--------|-------|--------|
| [`__init__(model, label_maps)`](../multi-model/lib/services/predictor.py:26) | FG_MFN, Dict[str, List[str]] | Self |
| [`predict_single()`](../multi-model/lib/services/predictor.py:55) | Tensors (1,*) | Dict with labels, confidences, class indices |
| [`predict_batch()`](../multi-model/lib/services/predictor.py:102) | Tensors (B,*) | List of per-sample dicts |

**Per-attribute output keys:**

| Key | Example | Description |
|-----|---------|-------------|
| `{attr_name}` | `"theme": "Food"` | Predicted label string |
| `{attr_name}_confidence` | `"theme_confidence": 0.92` | Softmax confidence |
| `{attr_name}_predicted_label_num` | `"theme_predicted_label_num": 0` | Class index |

### FeatureExtractor — [`lib/services/feature_extractor.py`](../multi-model/lib/services/feature_extractor.py)

#### [`FeatureExtractor`](../multi-model/lib/services/feature_extractor.py:25)

Extracts visual and text features from an image:

```python
extractor = FeatureExtractor(visual_module, text_module, ocr_engine)
features = extractor.extract(image)
# {"visual_features": Tensor, "text_features": Tensor, "raw_text": str, "confidence": float}
```

**Fallback behavior:** If OCR detects no text, zero text features are used:

```python
text_features = torch.zeros(1, hidden_size, device=device)
```

### Postprocessor — [`lib/services/postprocessor.py`](../multi-model/lib/services/postprocessor.py)

#### [`format_prediction_result()`](../multi-model/lib/services/postprocessor.py:21)

Strips internal fields from prediction output:

```python
clean = format_prediction_result(raw_result)
```

**Excluded fields:**

| Pattern | Example | Reason |
|---------|---------|--------|
| Exact match: `confidence_score` | — | Internal metric |
| Exact match: `predicted_label_num` | — | Internal index |
| Suffix `_confidence` | `theme_confidence` | Score field |
| Suffix `_score` | `quality_score` | Score field |
| Suffix `_accuracy` | `theme_accuracy` | Score field |
| Suffix `_f1` | `theme_f1` | Score field |

**Field renaming:** `predicted_label_text` → `predicted_label`
