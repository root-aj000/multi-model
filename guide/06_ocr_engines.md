# 06 — OCR Engines

EasyOCR and PaddleOCR integration, abstract base class, factory pattern, and configuration.

## Architecture

The OCR layer uses the **Strategy pattern** with a factory for engine creation:

```
OCREngine (ABC)
├── EasyOCREngine    — backed by easyocr library
└── PaddleOCREngine  — backed by paddleocr library

create_ocr_engine("easyocr")  → EasyOCREngine
create_ocr_engine("paddleocr") → PaddleOCREngine
```

Callers depend only on the [`OCREngine`](../multi-model/lib/ocr/engine.py:5) interface, making it trivial to swap OCR backends without changing any downstream code.

---

## OCREngine Abstract Base Class

**Source:** [`lib/ocr/engine.py`](../multi-model/lib/ocr/engine.py)

```python
class OCREngine(ABC):
    @abstractmethod
    def _init_engine(self) -> None: ...

    @abstractmethod
    def extract_text(self, image: Any) -> Tuple[str, float]: ...

    @abstractmethod
    def get_status(self) -> Dict[str, Any]: ...

    @abstractmethod
    def clear_cache(self, confirm: bool = False) -> bool: ...
```

| Method | Purpose | Returns |
|--------|---------|---------|
| `_init_engine()` | Initialize the backend reader/client | `None` |
| `extract_text()` | Run OCR on an image | `(text, confidence)` |
| `get_status()` | Report engine status | Dict with `engine`, `initialized`, `model_dir` |
| `clear_cache()` | Remove cached model files | `bool` (True if cleared) |

All concrete implementations must implement these four methods. The `confirm=True` safety on `clear_cache()` prevents accidental cache deletion.

---

## EasyOCREngine

**Source:** [`lib/ocr/easyocr.py`](../multi-model/lib/ocr/easyocr.py)

### Constructor

```python
engine = EasyOCREngine(model_dir=Path("local/ocr"), languages=["en"])
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_dir` | (required) | Directory for EasyOCR model storage |
| `languages` | `["en"]` | Language codes to load |

On construction, [`_init_engine()`](../multi-model/lib/ocr/easyocr.py:52) is called immediately, which imports `easyocr` and creates a `Reader`. Raises `ImportError` if easyocr is not installed; `RuntimeError` if initialization fails.

### [`extract_text()`](../multi-model/lib/ocr/easyocr.py:85)

```python
text, confidence = engine.extract_text(image)
# text = "Fresh Pizza Deal"
# confidence = 0.87
```

**Flow:**

1. Check `self.reader` is initialized (else raise `RuntimeError`)
2. `reader.readtext(image)` → list of `(bbox, text, confidence)` tuples
3. If no results → return `("", 0.0)`
4. Collect all text strings → join with spaces
5. Compute average confidence across all detections
6. Return `(combined_text, average_confidence)`

**Input types supported:** PIL Image, NumPy array, or file path string.

### [`get_status()`](../multi-model/lib/ocr/easyocr.py:124)

```python
status = engine.get_status()
# {"engine": "easyocr", "languages": ["en"], "initialized": True, "model_dir": "local/ocr"}
```

### [`clear_cache()`](../multi-model/lib/ocr/easyocr.py:138)

```python
engine.clear_cache(confirm=True)  # Deletes model_dir, sets reader=None
engine.clear_cache(confirm=False) # No-op, returns False
```

---

## PaddleOCREngine

**Source:** [`lib/ocr/paddleocr.py`](../multi-model/lib/ocr/paddleocr.py)

### Constructor

```python
engine = PaddleOCREngine(model_dir=Path("local/ocr"))
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_dir` | (required) | Directory for PaddleOCR model storage |

Language is fixed to `en` (English). On construction, [`_init_engine()`](../multi-model/lib/ocr/paddleocr.py:43) creates a `PaddleOCR` instance with angle classification enabled.

### [`extract_text()`](../multi-model/lib/ocr/paddleocr.py:68)

```python
text, confidence = engine.extract_text(image)
```

**Flow:**

1. Check `self.ocr` is available (else raise `RuntimeError`)
2. `ocr.ocr(image, cls=True)` → nested result structure
3. If no results → return `("", 0.0)`
4. Parse `result[0]`: each line is `[bbox, (text, confidence)]`
5. Collect texts and confidences → compute average
6. Return `(combined_text, average_confidence)`

### [`get_status()`](../multi-model/lib/ocr/paddleocr.py:108)

```python
status = engine.get_status()
# {"engine": "paddleocr", "initialized": True, "model_dir": "local/ocr"}
```

### [`clear_cache()`](../multi-model/lib/ocr/paddleocr.py:121)

Same safety pattern as EasyOCR: requires `confirm=True`.

---

## OCR Factory

**Source:** [`lib/ocr/factory.py`](../multi-model/lib/ocr/factory.py)

### [`create_ocr_engine()`](../multi-model/lib/ocr/factory.py:22)

```python
# Create EasyOCR (default)
engine = create_ocr_engine("easyocr", Path("local/ocr"))

# Create EasyOCR with custom languages
engine = create_ocr_engine("easyocr", Path("local/ocr"), languages=["en", "fr"])

# Create PaddleOCR
engine = create_ocr_engine("paddleocr", Path("local/ocr"))

# Unsupported engine → ValueError
engine = create_ocr_engine("tesseract", Path("local/ocr"))  # raises ValueError
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `engine_name` | `str` | `"easyocr"` or `"paddleocr"` |
| `model_dir` | `Path` | Directory for model file storage |
| `**kwargs` | — | Passed to engine constructor (e.g., `languages` for EasyOCR) |

**Raises:** `ValueError` if `engine_name` not in `["easyocr", "paddleocr"]`

---

## Comparison

| Feature | EasyOCR | PaddleOCR |
|---------|---------|-----------|
| **Languages** | Configurable (`["en"]`, `["en","fr"]`, etc.) | Fixed English |
| **Angle classification** | No | Yes (enabled) |
| **Model storage** | `model_storage_directory` param | `det_model_dir` param |
| **Result format** | Flat list of `(bbox, text, conf)` | Nested `[[bbox, (text, conf)]]` |
| **Best for** | Multilingual, simple setup | CJK languages, angle-variant text |
| **Install** | `pip install easyocr` | `pip install paddleocr` |

---

## Integration in Prediction Pipeline

The OCR engine is created once at server startup and reused for all requests:

```python
# In app/app.py lifespan():
ocr_engine = create_ocr_engine("easyocr", Path("local/ocr"))
configure_predictor(predictor, ocr_engine)

# In predict_image():
raw_text, confidence = extract_text(image, ocr_engine)
```

The [`extract_text()`](../multi-model/use_cases/prediction/predict_image.py:29) use-case function converts the image to a NumPy array before passing to the OCR engine, since both backends expect NumPy input rather than PIL Images.
