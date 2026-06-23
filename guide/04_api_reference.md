# 04 — API Reference

FastAPI endpoints, request/response schemas, and usage examples.

## Server Startup

The prediction server is a FastAPI application defined in [`app/app.py`](../multi-model/app/app.py). It uses the [lifespan pattern](../multi-model/app/app.py:36) for startup/shutdown:

**Startup:**
1. [`setup_directories()`](../multi-model/lib/utils/lifecycle.py:7) — creates `uploads/`, `saved_models/`, `local/models/`, `local/ocr/`, `local/cache/`
2. [`build_prediction_pipeline()`](../multi-model/use_cases/prediction/pipeline.py:29) — loads config, creates model, OCR engine, and Predictor
3. [`configure_predictor()`](../multi-model/app/predict.py:29) — wires predictor + OCR into the router's module-level variables

**Shutdown:**
1. [`cleanup_upload_directory()`](../multi-model/lib/utils/lifecycle.py:26) — removes all uploaded files

## Endpoints

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

**Source:** [`health_check()`](../multi-model/app/app.py:157)

---

### `GET /model/info`

Model information endpoint.

**Response:**

```json
{
  "model": "fg_mfn",
  "version": "1.0"
}
```

**Source:** [`model_info()`](../multi-model/app/app.py:168)

---

### `POST /predict`

Prediction endpoint for uploaded images. Accepts one or more image files and returns multi-attribute predictions.

**Source:** [`predict_endpoint()`](../multi-model/app/predict.py:73)

**Request:**

- Content-Type: `multipart/form-data`
- Field: `files` — one or more image files

```bash
curl -X POST http://localhost:8000/predict \
  -F "files=@ad_creative_1.jpg" \
  -F "files=@ad_creative_2.png"
```

**Constraints:**

| Constraint | Value | Error Code |
|------------|-------|------------|
| Allowed extensions | png, jpg, jpeg, gif, bmp, tiff, webp | 400 |
| Max file size | 10 MB | 413 |
| Service unavailability | Model or OCR not loaded | 503 |
| Prediction failure | Runtime error during inference | 500 |

**Response (200):**

```json
{
  "predictions": [
    {
      "theme": "Food",
      "sentiment": "Positive",
      "emotion": "Joy",
      "dominant_colour": "Red",
      "attention_score": "High",
      "trust_safety": "Safe",
      "target_audience": "Food Lovers",
      "predicted_ctr": "High",
      "likelihood_shares": "Medium",
      "predicted_label": "Food",
      "ocr_text": "Fresh Pizza Deal Only $9.99 Buy Now!",
      "keywords": "deal, free, buy",
      "monetary_mention": "$9.99",
      "call_to_action": "buy now",
      "object_detected": "",
      "filename": "ad_creative_1.jpg"
    }
  ],
  "total_images": 1,
  "processing_time_ms": 342
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | `List[Dict]` | One prediction dict per image |
| `predictions[].theme` | `str` | Ad theme classification |
| `predictions[].sentiment` | `str` | Positive / Negative / Neutral |
| `predictions[].emotion` | `str` | Emotional tone classification |
| `predictions[].dominant_colour` | `str` | Dominant color classification |
| `predictions[].attention_score` | `str` | High / Medium / Low |
| `predictions[].trust_safety` | `str` | Safe / Unsafe / Questionable |
| `predictions[].target_audience` | `str` | Target audience segment |
| `predictions[].predicted_ctr` | `str` | Predicted click-through rate |
| `predictions[].likelihood_shares` | `str` | Share likelihood |
| `predictions[].predicted_label` | `str` | Primary label (first attribute) |
| `predictions[].ocr_text` | `str` | Raw text extracted by OCR |
| `predictions[].keywords` | `str` | Comma-separated marketing keywords |
| `predictions[].monetary_mention` | `str` | Price/currency detected (or empty) |
| `predictions[].call_to_action` | `str` | CTA phrase detected (or empty) |
| `predictions[].object_detected` | `str` | Product categories detected (or empty) |
| `predictions[].filename` | `str` | Original uploaded filename |
| `total_images` | `int` | Number of images processed |
| `processing_time_ms` | `int` | Wall-clock processing time in ms |

**Error responses:**

| Status | When | Detail |
|--------|------|--------|
| 400 | File extension not allowed | `"File type not allowed. Allowed extensions: {...}"` |
| 413 | File exceeds 10 MB | `"File too large. Maximum size is 10485760 bytes."` |
| 503 | Model/OCR not configured | `"Prediction service is not configured..."` |
| 500 | Prediction runtime error | `"Prediction failed: <error message>"` |

## Internal Request Flow

```
POST /predict
  │
  ├─ Check _predictor and _ocr_engine are not None (else 503)
  │
  ├─ For each uploaded file:
  │   ├─ _load_image_from_upload() → PIL.Image (RGB)
  │   │   └─ Read bytes → Image.open() → .convert("RGB")
  │   │
  │   └─ predict_image(image, model, ocr_engine, label_maps, filename, predictor)
  │       ├─ extract_text(image, ocr_engine) → (raw_text, confidence)
  │       ├─ clean_text(raw_text) → cleaned_text
  │       ├─ _prepare_image_tensor(image) → Tensor(1, C, H, W)
  │       ├─ _prepare_text_tensors(cleaned_text) → (input_ids, attention_mask)
  │       ├─ predictor.predict_single() → raw result dict
  │       ├─ format_prediction_result() → stripped result dict
  │       ├─ Add ocr_text, filename, predicted_label
  │       └─ Add keywords, monetary_mention, call_to_action, object_detected
  │
  └─ Return {predictions, total_images, processing_time_ms}
```

## CORS Configuration

The server allows all origins for development:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

> **Warning:** For production, restrict `allow_origins` to your frontend's domain.

## Helper Functions

### [`allowed_file()`](../multi-model/app/app.py:99)

Validates file extension against the allowlist:

```python
allowed_file("photo.jpg")   # True
allowed_file("script.exe")  # False
allowed_file("noext")       # False
```

### [`save_upload_file()`](../multi-model/app/app.py:115)

Persists an uploaded file to disk with safety checks:

```python
path = save_upload_file(upload_file, Path("uploads/"))
# Returns: "uploads/photo.jpg"
```

- Sanitizes filename to prevent path traversal (`Path(filename).name`)
- Enforces 10 MB size limit
- Creates destination directory if needed

### [`_load_image_from_upload()`](../multi-model/app/predict.py:46)

Converts an uploaded file to a PIL Image in RGB mode:

```python
pil_image = _load_image_from_upload(upload_file)
# Returns: PIL.Image.Image in RGB mode
```

- Reads file bytes into memory
- Opens via `Image.open(BytesIO(content))`
- Converts to RGB (handles grayscale, RGBA, palette images)
- Resets file pointer for potential re-reads
