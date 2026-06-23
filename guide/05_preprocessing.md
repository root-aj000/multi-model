# 05 — Preprocessing

Text cleaning, tokenization, image transforms, dataset loading, and pipeline builders.

## Overview

The preprocessing layer prepares raw data (images and OCR text) for model consumption. It follows a **pipeline pattern** where individual transforms are composed into a single callable.

```
Raw Image ──→ resize ──→ normalize ──→ [augment] ──→ Tensor
Raw Text ──→ clean ──→ tokenize ──→ (input_ids, attention_mask)
```

---

## Text Preprocessing

### Text Cleaner — [`lib/preprocessing/text/cleaner.py`](../multi-model/lib/preprocessing/text/cleaner.py)

#### [`clean_text()`](../multi-model/lib/preprocessing/text/cleaner.py:18)

Normalizes OCR output by removing noise artifacts:

```python
cleaned = clean_text("  HELLO World!!! 123 \n\n")
# → "hello world 123"
```

| Step | Operation | Example |
|------|-----------|---------|
| 1 | Handle `None` → return `""` | `None` → `""` |
| 2 | Convert to lowercase | `"HELLO"` → `"hello"` |
| 3 | Strip leading/trailing whitespace | `"  hi  "` → `"hi"` |
| 4 | Remove non-alphanumeric chars (regex) | `"hello!!!"` → `"hello"` |
| 5 | Collapse multiple whitespace | `"a  b"` → `"a b"` |

**Raises:** `TypeError` if input is not `str` or `None`

---

### Tokenizer — [`lib/preprocessing/text/tokenizer.py`](../multi-model/lib/preprocessing/text/tokenizer.py)

#### [`load_tokenizer()`](../multi-model/lib/preprocessing/text/tokenizer.py:30)

Loads a HuggingFace tokenizer and caches the default instance:

```python
tokenizer = load_tokenizer("distilbert-base-uncased")
```

- The default tokenizer (`distilbert-base-uncased`) is cached at module level to avoid re-loading on every call
- Raises `OSError` if the model name is not found locally or on HuggingFace Hub

#### [`tokenize_text()`](../multi-model/lib/preprocessing/text/tokenizer.py:52)

Converts a string into padded/truncated token tensors:

```python
encoding = tokenize_text("Fresh Pizza Deal", max_length=512)
# encoding = {
#     "input_ids":      Tensor(512,),
#     "attention_mask":  Tensor(512,),
# }
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text` | (required) | String to tokenize |
| `max_length` | `512` | Maximum sequence length |
| `tokenizer` | `None` | Pre-loaded tokenizer; uses cached default if None |

**Safety guards:**
- Raises `ValueError` if text is empty after stripping
- Truncates input if length exceeds `MAX_RAW_TEXT_LENGTH` (100,000 chars) to prevent excessive memory use
- Uses `padding="max_length"` and `truncation=True` to ensure consistent tensor shapes

---

### Text Feature Pipeline — [`lib/preprocessing/text/pipeline.py`](../multi-model/lib/preprocessing/text/pipeline.py)

#### [`extract_keywords()`](../multi-model/lib/preprocessing/text/pipeline.py:40)

Identifies marketing keywords in OCR text:

```python
keywords = extract_keywords("Limited time sale! Buy now for free!")
# → ["sale", "free", "limited", "buy"]
```

Checks against `COMMON_KEYWORDS`: `sale, discount, offer, deal, free, new, limited, exclusive, special, buy`

#### [`extract_monetary_mention()`](../multi-model/lib/preprocessing/text/pipeline.py:54)

Detects price/currency mentions using regex:

```python
price = extract_monetary_mention("Only $9.99! Was ₹1,500.00")
# → "$9.99"  (returns first match only)
```

- Supports `$` (dollar) and `₹` (rupee) symbols
- Pattern: `\$[0-9,]+(?:\.[0-9]{2})?` or `₹[0-9,]+(?:\.[0-9]{2})?`
- Truncates input at 10,000 chars to prevent regex backtracking attacks
- Returns `None` if no match found

#### [`extract_call_to_action()`](../multi-model/lib/preprocessing/text/pipeline.py:78)

Detects call-to-action phrases:

```python
cta = extract_call_to_action("Click here to learn more!")
# → "click here"
```

Checks against `CTA_PHRASES`: `buy now, click here, learn more, shop now, get yours`

Returns `None` if no CTA found.

#### [`extract_objects_mentioned()`](../multi-model/lib/preprocessing/text/pipeline.py:95)

Identifies product category mentions:

```python
objects = extract_objects_mentioned("New phone and laptop deals!")
# → ["phone", "laptop"]
```

Checks against `PRODUCT_KEYWORDS`: `phone, laptop, car, dress, shoes, watch`

#### [`build_text_pipeline()`](../multi-model/lib/preprocessing/text/pipeline.py:109)

Composes a cleaner and tokenizer into a single callable:

```python
pipeline = build_text_pipeline(clean_text, tokenize_text)
result = pipeline("Hello World!!!", max_length=256)
# → {"input_ids": Tensor(256,), "attention_mask": Tensor(256,)}
```

The returned closure: `cleaner(text)` → `tokenizer(cleaned_text, max_length)`

---

## Image Preprocessing

### Image Transforms — [`lib/preprocessing/image/transforms.py`](../multi-model/lib/preprocessing/image/transforms.py)

#### [`resize_image()`](../multi-model/lib/preprocessing/image/transforms.py:21)

Resizes an image array using PIL's high-quality LANCZOS interpolation:

```python
resized = resize_image(image_array, size=(224, 224))
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | (required) | NumPy array `(H, W, C)` |
| `size` | (required) | Target `(width, height)` |
| `interpolation` | `Image.LANCZOS` | PIL interpolation method |

#### [`normalize_image()`](../multi-model/lib/preprocessing/image/transforms.py:43)

Scales pixel values from `[0, 255]` to `[0.0, 1.0]`:

```python
normalized = normalize_image(image_array)  # dtype: float32
```

#### [`augment_image()`](../multi-model/lib/preprocessing/image/transforms.py:56)

Applies random horizontal flip with 50% probability:

```python
augmented = augment_image(image_array)
```

This is the only augmentation currently implemented. Additional strategies (color jitter, rotation, etc.) can be added here.

#### [`build_image_transform()`](../multi-model/lib/preprocessing/image/transforms.py:74)

Creates a composed image transform pipeline:

```python
# Inference pipeline (no augmentation)
transform = build_image_transform(size=(224, 224), augment=False)

# Training pipeline (with augmentation)
transform = build_image_transform(size=(224, 224), augment=True)

processed = transform(image_array)
```

The returned closure always: `resize → normalize → [augment if enabled]`

---

### Image Pipeline Builder — [`lib/preprocessing/image/pipeline.py`](../multi-model/lib/preprocessing/image/pipeline.py)

#### [`build_image_pipeline()`](../multi-model/lib/preprocessing/image/pipeline.py:16)

Builds the image transform pipeline from a config dictionary:

```python
pipeline = build_image_pipeline({"image_size": (224, 224), "augment": True})
processed = pipeline(image_array)
```

| Config Key | Default | Description |
|------------|---------|-------------|
| `image_size` | `(224, 224)` | Target (width, height) |
| `augment` | `False` | Enable random horizontal flip |

---

## Dataset

### CustomDataset — [`lib/preprocessing/dataset.py`](../multi-model/lib/preprocessing/dataset.py)

```python
class CustomDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_pipeline=None,
                 text_pipeline=None, augment=False)
    def __len__(self) -> int
    def __getitem__(self, idx) -> Tuple[processed_image, label_dict]
    def _load_image(self, image_path) -> np.ndarray
```

#### CSV Format

The dataset reads a CSV file where each row is one sample:

| Column | Required | Description |
|--------|----------|-------------|
| `image_filename` | Yes | Filename of the image (relative to `image_dir`) |
| Any other columns | No | Treated as label columns, returned in `label_dict` |

Example CSV:

```csv
image_filename,theme,sentiment,emotion
ad_001.jpg,Food,Positive,Joy
ad_002.jpg,Tech,Negative,Fear
```

#### Expected Directory Structure

```
dataset/
├── train/
│   ├── train.csv
│   └── images/
│       ├── ad_001.jpg
│       └── ad_002.jpg
├── val/
│   ├── val.csv
│   └── images/
└── test/
    ├── test.csv
    └── images/
```

#### [`__getitem__()`](../multi-model/lib/preprocessing/dataset.py:87)

Returns one sample:

1. Read row at `idx` from the pandas DataFrame
2. Construct image path: `image_dir / row["image_filename"]`
3. Load image as RGB NumPy array via [`_load_image()`](../multi-model/lib/preprocessing/dataset.py:122)
4. Apply `image_pipeline` if provided
5. Apply `augment_image()` if `augment=True`
6. Build `label_dict` from all columns except `image_filename`
7. Return `(processed_image, label_dict)`

#### [`load_dataset()`](../multi-model/lib/preprocessing/dataset.py:147)

Convenience function that constructs a dataset from a split name:

```python
train_ds = load_dataset("train", augment=True, image_pipeline=my_pipeline)
val_ds = load_dataset("val", augment=False)
```

Uses [`get_dataset_paths()`](../multi-model/lib/utils/config.py:32) to resolve paths:
- CSV: `dataset/{split}/{split}.csv`
- Images: `dataset/{split}/images/`

---

## Tensor Preparation (Prediction Path)

These functions in [`use_cases/prediction/predict_image.py`](../multi-model/use_cases/prediction/predict_image.py) bridge between preprocessing and model input:

### [`_to_numpy_array()`](../multi-model/use_cases/prediction/predict_image.py:161)

Converts any image format to a NumPy array:

| Input Type | Conversion |
|------------|------------|
| `str` (file path) | `Image.open(path).convert("RGB")` → `np.array()` |
| `PIL.Image` | `.convert("RGB")` → `np.array()` |
| `np.ndarray` | Pass through unchanged |
| Other | `np.array(image)` (best-effort) |

### [`_prepare_image_tensor()`](../multi-model/use_cases/prediction/predict_image.py:185)

Converts any image to a batched PyTorch tensor for model input:

| Step | Operation |
|------|-----------|
| 1 | Convert to NumPy array via `_to_numpy_array()` |
| 2 | Handle grayscale: `(H,W)` or `(H,W,1)` → replicate to `(H,W,3)` |
| 3 | Handle RGBA: `(H,W,4)` → drop alpha → `(H,W,3)` |
| 4 | Normalize `uint8 [0,255]` → `float32 [0,1]` |
| 5 | Transpose `(H,W,C)` → `(C,H,W)` |
| 6 | Add batch dim: `.unsqueeze(0)` → `(1,C,H,W)` |

### [`_prepare_text_tensors()`](../multi-model/use_cases/prediction/predict_image.py:224)

Tokenizes cleaned text and returns batched tensors:

| Input | Output |
|-------|--------|
| `text: str` | `(input_ids: Tensor(1, seq_len), attention_mask: Tensor(1, seq_len))` |

If text is empty, tokenizes a single space `" "` as a placeholder to avoid errors.
