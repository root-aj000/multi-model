# 02 — Architecture

System architecture, module relationships, design patterns, and data flow.

## Directory Structure

```
multi-model/
├── app/                          # FastAPI application layer
│   ├── app.py                    # Application factory, lifespan, routes
│   └── predict.py                # Prediction router and endpoint
├── configs/
│   └── model/
│       └── model_config.json     # Model configuration
├── lib/                          # Core library (business logic)
│   ├── models/
│   │   ├── fg_mfn.py             # FG_MFN multi-modal fusion model
│   │   ├── text.py               # TextModule (transformer encoder)
│   │   ├── visual.py             # VisualModule (CNN backbone)
│   │   └── factory.py            # Model creation and loading
│   ├── ocr/
│   │   ├── engine.py             # OCREngine abstract base class
│   │   ├── easyocr.py            # EasyOCR implementation
│   │   ├── paddleocr.py          # PaddleOCR implementation
│   │   └── factory.py            # OCR engine factory
│   ├── preprocessing/
│   │   ├── dataset.py            # CustomDataset + load_dataset()
│   │   ├── image/
│   │   │   ├── pipeline.py       # Image pipeline builder
│   │   │   └── transforms.py     # resize, normalize, augment
│   │   └── text/
│   │       ├── cleaner.py        # Text cleaning (OCR artifacts)
│   │       ├── pipeline.py       # Keyword/monetary/CTA extraction
│   │       └── tokenizer.py      # HuggingFace tokenization
│   ├── services/
│   │   ├── predictor.py          # Predictor (logits → labels)
│   │   ├── feature_extractor.py  # Multi-modal feature extraction
│   │   └── postprocessor.py      # Strip internal fields from output
│   └── utils/
│       ├── cache.py              # Cache management
│       ├── config.py             # JSON config loading + label maps
│       ├── lifecycle.py          # Directory setup/cleanup
│       └── logging.py            # TrainingLogger + create_logger
├── use_cases/                    # Application use cases (orchestration)
│   ├── prediction/
│   │   ├── pipeline.py           # Build prediction pipeline
│   │   ├── predict_image.py      # Single image prediction
│   │   └── predict_batch.py      # Batch image prediction
│   └── training/
│       ├── pipeline.py           # Build training pipeline
│       ├── train_model.py        # Train/validation epoch loops
│       └── evaluate.py           # Evaluate + compute metrics
├── scripts/                      # CLI entry points
│   ├── train_model.py            # Training CLI
│   ├── evaluate.py               # Evaluation CLI
│   ├── predict_server.py         # Server startup CLI
│   └── analyze_model.py          # Model analysis/benchmarking
└── tests/                        # Test suite
    ├── conftest.py
    └── unit/
```

## Layered Architecture

The project follows a **clean architecture** with four distinct layers:

```
┌─────────────────────────────────────────────┐
│              scripts/ (CLI)                  │  Entry points
├─────────────────────────────────────────────┤
│              app/ (API)                      │  HTTP interface
├─────────────────────────────────────────────┤
│           use_cases/ (Orchestration)         │  Business workflows
├─────────────────────────────────────────────┤
│             lib/ (Core Library)              │  Domain logic
└─────────────────────────────────────────────┘
```

### Dependency Rule

Dependencies flow **inward only** — outer layers may import inner layers, but inner layers never import outer layers:

- `scripts/` → imports from `use_cases/`, `lib/`
- `app/` → imports from `use_cases/`, `lib/`
- `use_cases/` → imports from `lib/`
- `lib/` → imports only `torch`, `transformers`, `PIL`, and stdlib

## Design Patterns

### 1. Factory Pattern

Used for both models and OCR engines to decouple creation from usage:

- [`create_model()`](../multi-model/lib/models/factory.py:60) — creates a fresh FG_MFN from config
- [`load_model()`](../multi-model/lib/models/factory.py:78) — loads FG_MFN from checkpoint
- [`create_ocr_engine()`](../multi-model/lib/ocr/factory.py:22) — creates EasyOCR or PaddleOCR by name

### 2. Strategy Pattern

[`OCREngine`](../multi-model/lib/ocr/engine.py:5) is the strategy interface; [`EasyOCREngine`](../multi-model/lib/ocr/easyocr.py:24) and [`PaddleOCREngine`](../multi-model/lib/ocr/paddleocr.py:21) are concrete strategies. Callers depend on the abstract interface, not the implementation.

### 3. Pipeline Builder

Both prediction and training use a "build pipeline" function that assembles all dependencies in one call:

- [`build_prediction_pipeline()`](../multi-model/use_cases/prediction/pipeline.py:29) → returns `Predictor`
- [`build_training_pipeline()`](../multi-model/use_cases/training/pipeline.py:133) → returns `Dict` with model, loaders, optimizer, etc.

### 4. Async Context Manager (Lifespan)

[`lifespan()`](../multi-model/app/app.py:36) uses FastAPI's async context manager pattern for startup/shutdown lifecycle management. Resources are initialized before the server accepts requests and cleaned up on shutdown.

### 5. Template Method

[`OCREngine`](../multi-model/lib/ocr/engine.py:5) defines abstract methods (`_init_engine()`, `extract_text()`, `get_status()`, `clear_cache()`) that subclasses must implement, ensuring a consistent interface across OCR backends.

## Three System Flows

### Serving Flow

```
Client POST /predict
    → app.lifespan() [startup: load model, OCR, configure predictor]
    → predict_endpoint()
    → _load_image_from_upload() → PIL.Image
    → predict_image()
        → extract_text() → (raw_text, confidence)
        → clean_text() → cleaned_text
        → _prepare_image_tensor() → Tensor(1,C,H,W)
        → _prepare_text_tensors() → (input_ids, attention_mask)
        → predictor.predict_single() → raw result dict
        → format_prediction_result() → clean result dict
        → extract_keywords/monetary/CTA/objects → enriched result
    → JSON response
```

### Training Flow

```
CLI: python -m scripts.train_model --config <path>
    → build_training_pipeline()
        → load_config() → cfg
        → load_datasets() → (train_ds, val_ds)
        → create_data_loaders() → (train_loader, val_loader)
        → create_model() → FG_MFN
        → setup_training_components() → {optimizer, criterion, scheduler}
    → for each epoch:
        → apply_warmup() [if no scheduler]
        → train_epoch() → (loss, accuracy)
        → validate_epoch() → (loss, accuracy)
        → scheduler.step()
        → TrainingLogger.log_metrics()
        → if best: torch.save(model.state_dict())
```

### Evaluation Flow

```
CLI: python -m scripts.evaluate --config <path> --checkpoint <path>
    → load_model() → FG_MFN with restored weights
    → load_dataset("test") → CustomDataset
    → DataLoader
    → evaluate_model() → {all_preds, all_labels}
    → compute_metrics() → {accuracy, total}
    → save_results() → results/results.json
```

## Cross-Module Dependencies

| Layer | Module | Depends On |
|-------|--------|------------|
| **App** | [`app.app`](../multi-model/app/app.py) | `lib.utils.lifecycle`, `use_cases.prediction.pipeline`, `app.predict`, `lib.ocr.factory` |
| **App** | [`app.predict`](../multi-model/app/predict.py) | `lib.ocr.engine`, `lib.services.predictor`, `use_cases.prediction.predict_image` |
| **Use Cases** | [`prediction.pipeline`](../multi-model/use_cases/prediction/pipeline.py) | `lib.models.factory`, `lib.ocr.factory`, `lib.services.predictor`, `lib.utils.config` |
| **Use Cases** | [`prediction.predict_image`](../multi-model/use_cases/prediction/predict_image.py) | `lib.ocr.engine`, `lib.preprocessing.text.*`, `lib.services.postprocessor`, `lib.services.predictor` |
| **Use Cases** | [`training.pipeline`](../multi-model/use_cases/training/pipeline.py) | `lib.models.factory`, `lib.preprocessing.dataset`, `lib.utils.config`, `use_cases.training.train_model` |
| **Lib** | [`models.fg_mfn`](../multi-model/lib/models/fg_mfn.py) | `lib.models.text`, `lib.models.visual` |
| **Lib** | [`services.predictor`](../multi-model/lib/services/predictor.py) | `lib.models.fg_mfn` |
| **Lib** | [`services.feature_extractor`](../multi-model/lib/services/feature_extractor.py) | `lib.models.text`, `lib.models.visual`, `lib.ocr.engine`, `lib.preprocessing.text.*` |
| **Lib** | [`preprocessing.dataset`](../multi-model/lib/preprocessing/dataset.py) | `lib.preprocessing.image.transforms`, `lib.utils.config` |

## Key Architectural Decisions

### Multi-Attribute Heads over Single Classifier

FG_MFN uses 9 separate classification heads instead of one large classifier. Benefits:
- Each attribute can have a different number of classes
- Loss is computed per-head and summed, allowing the model to specialize
- Individual attribute accuracy can be tracked independently

### Zero-Text Fallback

When OCR detects no text in an image, the system creates a zero text-feature tensor instead of failing. This allows the model to still make predictions based on visual features alone, which is critical for image-heavy ads with minimal text.

### Atomic Checkpoint Writes

The training script writes model checkpoints to a `.tmp` file first, then uses `os.replace()` to atomically move it to the final path. This prevents corrupt checkpoints from partial writes during crashes.

### Two-Format Config Support

[`get_label_maps()`](../multi-model/lib/utils/config.py:49) supports both a flat `label_maps` key and the nested `ATTRIBUTES` format. This provides backward compatibility with older configuration files while supporting the richer nested format.
