# 13 — Dataflow Diagrams

> Visual dataflow documentation for every function in the Multi-Model Prediction System codebase.

This guide indexes and explains the complete set of **Mermaid dataflow diagrams** created for the project. Each diagram traces how data enters a function, how it transforms, and what data exits — providing a function-level view of the entire system.

---

## Table of Contents

1. [Overview](#overview)
2. [Diagram Index](#diagram-index)
3. [Rendering Methods](#rendering-methods)
4. [Diagram Summaries](#diagram-summaries)
5. [Cross-Module Dependencies](#cross-module-dependencies)
6. [Data Types Reference](#data-types-reference)

---

## Overview

The codebase was analyzed at the function level and **39 Mermaid flowchart diagrams** were produced, covering:

- **3 system-level pipelines** (serving, training, evaluation)
- **4 API-layer functions** (lifespan, file validation, upload, prediction endpoint)
- **4 model-layer diagrams** (FG_MFN init, forward, fusion, config validation)
- **2 sub-module diagrams** (TextModule init + forward)
- **2 sub-module diagrams** (VisualModule init + forward)
- **2 factory diagrams** (model create + load)
- **4 OCR diagrams** (ABC, EasyOCR, PaddleOCR, factory)
- **3 text preprocessing diagrams** (cleaner, tokenizer, feature extraction)
- **1 image transforms diagram**
- **1 dataset diagram**
- **3 service diagrams** (Predictor, FeatureExtractor, Postprocessor)
- **2 config utility diagrams** (load_config, get_label_maps)
- **1 lifecycle diagram**
- **1 logging diagram**
- **1 training script diagram**
- **2 use-case diagrams** (predict_image, prepare_image_tensor)
- **1 training epoch diagram**
- **1 evaluation diagram**
- **1 cross-module dependency map**

All diagram source files (`.mmd`) live in [`docs/diagrams/`](../docs/diagrams/). The master document with all diagrams inline is [`docs/dataflow_diagrams.md`](../docs/dataflow_diagrams.md). Detailed function explanations are in [`docs/function_explanations.md`](../docs/function_explanations.md).

---

## Diagram Index

| # | Diagram File | Module | Top-Level Function |
|---|-------------|--------|--------------------|
| 01 | [`01_system_level.mmd`](../docs/diagrams/01_system_level.mmd) | System | Three paths: Serving, Training, Evaluation |
| 02 | [`02_prediction_pipeline.mmd`](../docs/diagrams/02_prediction_pipeline.mmd) | `use_cases.prediction.pipeline` | [`build_prediction_pipeline()`](../multi-model/use_cases/prediction/pipeline.py:29) |
| 03 | [`03_training_pipeline.mmd`](../docs/diagrams/03_training_pipeline.mmd) | `use_cases.training.pipeline` | [`build_training_pipeline()`](../multi-model/use_cases/training/pipeline.py:133) |
| 04 | [`04_evaluation_pipeline.mmd`](../docs/diagrams/04_evaluation_pipeline.mmd) | `use_cases.training.evaluate` | [`evaluate_model()`](../multi-model/use_cases/training/evaluate.py:21) |
| 05a | [`05_lifespan.mmd`](../docs/diagrams/05_lifespan.mmd) | `app.app` | [`lifespan()`](../multi-model/app/app.py:36) |
| 05b | [`05_allowed_file.mmd`](../docs/diagrams/05_allowed_file.mmd) | `app.app` | [`allowed_file()`](../multi-model/app/app.py:99) |
| 05c | [`05_save_upload_file.mmd`](../docs/diagrams/05_save_upload_file.mmd) | `app.app` | [`save_upload_file()`](../multi-model/app/app.py:115) |
| 05d | [`05_predict_endpoint.mmd`](../docs/diagrams/05_predict_endpoint.mmd) | `app.predict` | [`predict_endpoint()`](../multi-model/app/predict.py:73) |
| 06a | [`06_fg_mfn_init.mmd`](../docs/diagrams/06_fg_mfn_init.mmd) | `lib.models.fg_mfn` | [`FG_MFN.__init__()`](../multi-model/lib/models/fg_mfn.py:53) |
| 06b | [`06_fg_mfn_forward.mmd`](../docs/diagrams/06_fg_mfn_forward.mmd) | `lib.models.fg_mfn` | [`FG_MFN.forward()`](../multi-model/lib/models/fg_mfn.py:259) |
| 06c | [`06_fg_mfn_fuse_features.mmd`](../docs/diagrams/06_fg_mfn_fuse_features.mmd) | `lib.models.fg_mfn` | [`FG_MFN._fuse_features()`](../multi-model/lib/models/fg_mfn.py:294) |
| 06d | [`06_fg_mfn_validate_config.mmd`](../docs/diagrams/06_fg_mfn_validate_config.mmd) | `lib.models.fg_mfn` | [`FG_MFN._validate_config()`](../multi-model/lib/models/fg_mfn.py:124) |
| 07a | [`07_textmodule_init.mmd`](../docs/diagrams/07_textmodule_init.mmd) | `lib.models.text` | [`TextModule.__init__()`](../multi-model/lib/models/text.py:36) |
| 07b | [`07_textmodule_forward.mmd`](../docs/diagrams/07_textmodule_forward.mmd) | `lib.models.text` | [`TextModule.forward()`](../multi-model/lib/models/text.py:84) |
| 08a | [`08_visualmodule_init.mmd`](../docs/diagrams/08_visualmodule_init.mmd) | `lib.models.visual` | [`VisualModule.__init__()`](../multi-model/lib/models/visual.py:36) |
| 08b | [`08_visualmodule_forward.mmd`](../docs/diagrams/08_visualmodule_forward.mmd) | `lib.models.visual` | [`VisualModule.forward()`](../multi-model/lib/models/visual.py:128) |
| 09a | [`09_model_factory_create.mmd`](../docs/diagrams/09_model_factory_create.mmd) | `lib.models.factory` | [`create_model()`](../multi-model/lib/models/factory.py:60) |
| 09b | [`09_model_factory_load.mmd`](../docs/diagrams/09_model_factory_load.mmd) | `lib.models.factory` | [`load_model()`](../multi-model/lib/models/factory.py:78) |
| 10 | [`10_ocr_engine_abc.mmd`](../docs/diagrams/10_ocr_engine_abc.mmd) | `lib.ocr.engine` | [`OCREngine`](../multi-model/lib/ocr/engine.py:5) (ABC) |
| 11 | [`11_easyocr_extract.mmd`](../docs/diagrams/11_easyocr_extract.mmd) | `lib.ocr.easyocr` | [`EasyOCREngine.extract_text()`](../multi-model/lib/ocr/easyocr.py:85) |
| 12 | [`12_paddleocr_extract.mmd`](../docs/diagrams/12_paddleocr_extract.mmd) | `lib.ocr.paddleocr` | [`PaddleOCREngine.extract_text()`](../multi-model/lib/ocr/paddleocr.py:68) |
| 13 | [`13_ocr_factory.mmd`](../docs/diagrams/13_ocr_factory.mmd) | `lib.ocr.factory` | [`create_ocr_engine()`](../multi-model/lib/ocr/factory.py:22) |
| 14 | [`14_clean_text.mmd`](../docs/diagrams/14_clean_text.mmd) | `lib.preprocessing.text.cleaner` | [`clean_text()`](../multi-model/lib/preprocessing/text/cleaner.py:18) |
| 15 | [`15_tokenize_text.mmd`](../docs/diagrams/15_tokenize_text.mmd) | `lib.preprocessing.text.tokenizer` | [`tokenize_text()`](../multi-model/lib/preprocessing/text/tokenizer.py:52) |
| 16a | [`16_extract_keywords.mmd`](../docs/diagrams/16_extract_keywords.mmd) | `lib.preprocessing.text.pipeline` | [`extract_keywords()`](../multi-model/lib/preprocessing/text/pipeline.py:40) |
| 16b | [`16_extract_monetary.mmd`](../docs/diagrams/16_extract_monetary.mmd) | `lib.preprocessing.text.pipeline` | [`extract_monetary_mention()`](../multi-model/lib/preprocessing/text/pipeline.py:54) |
| 17 | [`17_image_transforms.mmd`](../docs/diagrams/17_image_transforms.mmd) | `lib.preprocessing.image.transforms` | [`build_image_transform()`](../multi-model/lib/preprocessing/image/transforms.py:74) |
| 19 | [`19_dataset_getitem.mmd`](../docs/diagrams/19_dataset_getitem.mmd) | `lib.preprocessing.dataset` | [`CustomDataset.__getitem__()`](../multi-model/lib/preprocessing/dataset.py:87) |
| 20 | [`20_predictor_predict_single.mmd`](../docs/diagrams/20_predictor_predict_single.mmd) | `lib.services.predictor` | [`Predictor.predict_single()`](../multi-model/lib/services/predictor.py:55) |
| 21 | [`21_feature_extractor_extract.mmd`](../docs/diagrams/21_feature_extractor_extract.mmd) | `lib.services.feature_extractor` | [`FeatureExtractor.extract()`](../multi-model/lib/services/feature_extractor.py:52) |
| 22 | [`22_format_prediction_result.mmd`](../docs/diagrams/22_format_prediction_result.mmd) | `lib.services.postprocessor` | [`format_prediction_result()`](../multi-model/lib/services/postprocessor.py:21) |
| 23a | [`23_load_config.mmd`](../docs/diagrams/23_load_config.mmd) | `lib.utils.config` | [`load_config()`](../multi-model/lib/utils/config.py:9) |
| 23b | [`23_get_label_maps.mmd`](../docs/diagrams/23_get_label_maps.mmd) | `lib.utils.config` | [`get_label_maps()`](../multi-model/lib/utils/config.py:49) |
| 24 | [`24_setup_directories.mmd`](../docs/diagrams/24_setup_directories.mmd) | `lib.utils.lifecycle` | [`setup_directories()`](../multi-model/lib/utils/lifecycle.py:7) |
| 26 | [`26_training_logger.mmd`](../docs/diagrams/26_training_logger.mmd) | `lib.utils.logging` | [`TrainingLogger.log_metrics()`](../multi-model/lib/utils/logging.py:30) |
| 27 | [`27_train_model_script.mmd`](../docs/diagrams/27_train_model_script.mmd) | `scripts.train_model` | [`main()`](../multi-model/scripts/train_model.py:32) |
| 31a | [`31_predict_image.mmd`](../docs/diagrams/31_predict_image.mmd) | `use_cases.prediction.predict_image` | [`predict_image()`](../multi-model/use_cases/prediction/predict_image.py:47) |
| 31b | [`31_prepare_image_tensor.mmd`](../docs/diagrams/31_prepare_image_tensor.mmd) | `use_cases.prediction.predict_image` | [`_prepare_image_tensor()`](../multi-model/use_cases/prediction/predict_image.py:185) |
| 35 | [`35_train_epoch.mmd`](../docs/diagrams/35_train_epoch.mmd) | `use_cases.training.train_model` | [`train_epoch()`](../multi-model/use_cases/training/train_model.py:103) |
| 36 | [`36_evaluate_model.mmd`](../docs/diagrams/36_evaluate_model.mmd) | `use_cases.training.evaluate` | [`evaluate_model()`](../multi-model/use_cases/training/evaluate.py:21) |
| — | [`cross_module_dependency_map.mmd`](../docs/diagrams/cross_module_dependency_map.mmd) | Cross-cutting | Module import relationships |

---

## Rendering Methods

The `.mmd` files contain pure Mermaid flowchart syntax. You can render them using any of the following methods:

### Method 1: Interactive HTML Viewer

A self-contained HTML file with all diagrams embedded, rendered client-side by the Mermaid.js CDN.

```bash
# Open directly in a browser
xdg-open docs/diagrams/viewer.html
```

Features:
- Sidebar navigation with diagram categories
- Dark theme with syntax highlighting
- All 39 diagrams rendered inline
- No build tools required — works offline after first CDN load

### Method 2: Generated HTML (All Diagramagrams in One Page)

A Python script that reads all `.mmd` files and produces a single-page HTML file:

```bash
python docs/diagrams/generate_html.py
# Output: docs/diagrams/all_diagrams.html
```

Open the output in any browser:

```bash
xdg-open docs/diagrams/all_diagrams.html
```

### Method 3: PNG Rendering via mermaid.ink API

A Python script that sends each `.mmd` file to the public [mermaid.ink](https://mermaid.ink) API and saves PNG images:

```bash
python docs/diagrams/render_via_api.py
# Output: docs/diagrams/out/*.png (one PNG per diagram)
```

Requires: `requests` package (`pip install requests`). No npm or Puppeteer needed.

### Method 4: PlantUML Alternative

A PlantUML version of the 6 major system diagrams is available at [`docs/diagrams/dataflow.plantuml`](../docs/diagrams/dataflow.plantuml). Render with any PlantUML viewer:

```bash
java -jar plantuml.jar docs/diagrams/dataflow.plantuml
```

### Method 5: Mermaid CLI (mmdc)

If you have `@mermaid-js/mermaid-cli` installed with Puppeteer/Chrome dependencies:

```bash
bash docs/diagrams/render_all.sh
```

> **Note:** Method 5 requires npm and a Chromium-based browser. If `mmdc` fails due to Puppeteer issues, use Methods 1–3 instead.

---

## Diagram Summaries

Each summary describes the top-level function's dataflow. For full Input/Steps/Output tables, see [`docs/function_explanations.md`](../docs/function_explanations.md).

### System-Level Flows

#### 01 — System-Level Dataflow

The three major execution paths through the codebase:

- **Serving Path:** `POST /predict` → FastAPI lifespan → [`predict_endpoint()`](../multi-model/app/predict.py:73) → [`predict_image()`](../multi-model/use_cases/prediction/predict_image.py:47) → FG_MFN inference → JSON response
- **Training Path:** CLI `train_model.py` → [`build_training_pipeline()`](../multi-model/use_cases/training/pipeline.py:133) → [`train_epoch()`](../multi-model/use_cases/training/train_model.py:103) / [`validate_epoch()`](../multi-model/use_cases/training/train_model.py:192) → checkpoint
- **Evaluation Path:** CLI `evaluate.py` → [`evaluate_model()`](../multi-model/use_cases/training/evaluate.py:21) → [`compute_metrics()`](../multi-model/use_cases/training/evaluate.py:101) → `results.json`

#### 02 — Prediction Pipeline

[`build_prediction_pipeline()`](../multi-model/use_cases/prediction/pipeline.py:29) assembles all inference dependencies at startup: loads config → extracts label maps → creates FG_MFN model → creates OCR engine → wraps in [`Predictor`](../multi-model/lib/services/predictor.py:18). The returned `Predictor` is reused for every request.

#### 03 — Training Pipeline

[`build_training_pipeline()`](../multi-model/use_cases/training/pipeline.py:133) assembles training objects: loads config → loads train/val datasets → creates DataLoaders → creates model → sets up optimizer/criterion/scheduler. Returns a dict consumed by the training loop.

#### 04 — Evaluation Pipeline

[`evaluate_model()`](../multi-model/use_cases/training/evaluate.py:21) runs inference on the full test set: `model.eval()` → iterate batches → forward pass → argmax → collect predictions and labels → return for metric computation.

---

### App / API Layer

#### 05a — Lifespan

[`lifespan()`](../multi-model/app/app.py:36) is FastAPI's async context manager. On startup: `setup_directories()` → `build_prediction_pipeline()` → `create_ocr_engine()` → `configure_predictor()`. On shutdown: `cleanup_upload_directory()`.

#### 05b — Allowed File

[`allowed_file()`](../multi-model/app/app.py:99) checks if an uploaded filename has a permitted extension (png/jpg/jpeg/gif/bmp/tiff/webp). Returns `bool`.

#### 05c — Save Upload File

[`save_upload_file()`](../multi-model/app/app.py:115) persists an uploaded file: extension check → mkdir → sanitize filename → read content → size check (≤10 MB) → write to disk. Raises `HTTPException(400)` or `HTTPException(413)` on failure.

#### 05d — Predict Endpoint

[`predict_endpoint()`](../multi-model/app/predict.py:73) is the primary API endpoint. Accepts multipart file uploads → validates service is configured → for each file: load image → `predict_image()` → collect results → return JSON with `predictions`, `total_images`, `processing_time_ms`.

---

### Model Layer

#### 06a — FG_MFN Initialization

[`FG_MFN.__init__()`](../multi-model/lib/models/fg_mfn.py:53) constructs the multi-modal architecture: validate config → create [`VisualModule`](../multi-model/lib/models/visual.py:27) → create [`TextModule`](../multi-model/lib/models/text.py:28) → build shared FC layer → create 9 attribute classification heads → optionally create legacy sentiment head.

#### 06b — FG_MFN Forward Pass

[`FG_MFN.forward()`](../multi-model/lib/models/fg_mfn.py:259) runs full inference: validate inputs → visual encoder → text encoder → fuse features → shared FC + ReLU + Dropout → 9 attribute heads → return dict of logits per attribute.

#### 06c — Feature Fusion

[`FG_MFN._fuse_features()`](../multi-model/lib/models/fg_mfn.py:294) combines modalities. **Concat mode:** `torch.cat([visual, text], dim=1)` → Linear projection to `hidden_dim`. **Add mode:** element-wise `visual + text` (parameter-free, same shape).

#### 06d — Config Validation

[`FG_MFN._validate_config()`](../multi-model/lib/models/fg_mfn.py:124) performs fail-fast checks before model construction: config is dict → required keys exist → `HIDDEN_DIM > 0` → each attribute has `num_classes` and `labels`.

#### 07a — TextModule Initialization

[`TextModule.__init__()`](../multi-model/lib/models/text.py:36) loads a pretrained transformer (e.g., DistilBERT) via `AutoModel.from_pretrained()`. If the encoder's native hidden size differs from the requested size, creates a linear projection layer.

#### 07b — TextModule Forward Pass

[`TextModule.forward()`](../multi-model/lib/models/text.py:84) extracts text features: encoder forward → extract `[CLS]` token (`last_hidden_state[:, 0, :]`) → projection → `Tensor(B, hidden_size)`.

#### 08a — VisualModule Initialization

[`VisualModule.__init__()`](../multi-model/lib/models/visual.py:36) loads a pretrained CNN backbone (ResNet50/ResNet18/ResNet101) → replaces the 1000-class ImageNet FC head with a projection to `out_features`.

#### 08b — VisualModule Forward Pass

[`VisualModule.forward()`](../multi-model/lib/models/visual.py:128) extracts visual features: validate 4D input → backbone forward → `Tensor(B, out_features)`.

#### 09a — Model Factory: Create

[`create_model()`](../multi-model/lib/models/factory.py:60) creates a fresh FG_MFN from config → moves to device → sets eval mode.

#### 09b — Model Factory: Load

[`load_model()`](../multi-model/lib/models/factory.py:78) restores a trained model: check checkpoint exists → `create_model()` → `torch.load()` (tries `weights_only=True`, falls back to `False`) → `load_state_dict()`.

---

### OCR Layer

#### 10 — OCREngine ABC

[`OCREngine`](../multi-model/lib/ocr/engine.py:5) defines the contract: `_init_engine()`, `extract_text(image) → (str, float)`, `get_status() → Dict`, `clear_cache(confirm) → bool`. Enables swapping EasyOCR/PaddleOCR without changing callers.

#### 11 — EasyOCREngine

[`EasyOCREngine.extract_text()`](../multi-model/lib/ocr/easyocr.py:85) runs `reader.readtext(image)` → collects text regions and confidences → averages confidence → joins texts with spaces. Returns `("", 0.0)` if no text detected.

#### 12 — PaddleOCREngine

[`PaddleOCREngine.extract_text()`](../multi-model/lib/ocr/paddleocr.py:68) runs `ocr.ocr(image, cls=True)` → parses nested result structure → collects texts and confidences → averages confidence → joins texts. Same interface as EasyOCR but with PaddleOCR's detection+recognition pipeline.

#### 13 — OCR Factory

[`create_ocr_engine()`](../multi-model/lib/ocr/factory.py:22) maps engine name strings to implementations: `"easyocr"` → `EasyOCREngine`, `"paddleocr"` → `PaddleOCREngine`. Raises `ValueError` for unsupported names.

---

### Preprocessing Layer

#### 14 — Text Cleaner

[`clean_text()`](../multi-model/lib/preprocessing/text/cleaner.py:18) normalizes OCR output: None → `""` → lowercase → strip → regex remove `[^a-z0-9\s]` → collapse whitespace. Essential before tokenization.

#### 15 — Tokenizer

[`tokenize_text()`](../multi-model/lib/preprocessing/text/tokenizer.py:52) converts text to model input: validate non-empty → truncate if >100K chars → load/cached tokenizer → `tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")` → squeeze batch dim → `Dict[input_ids, attention_mask]`.

#### 16a — Keyword Extraction

[`extract_keywords()`](../multi-model/lib/preprocessing/text/pipeline.py:40) checks each word against `COMMON_KEYWORDS` (sale, discount, offer, deal, free, new, limited, exclusive, special, buy). Returns matched marketing keywords.

#### 16b — Monetary Mention Extraction

[`extract_monetary_mention()`](../multi-model/lib/preprocessing/text/pipeline.py:54) searches for price patterns (`$9.99`, `₹1,500.00`) via regex. Input length guard prevents backtracking attacks. Returns first match or `None`.

#### 17 — Image Transforms Pipeline

[`build_image_transform()`](../multi-model/lib/preprocessing/image/transforms.py:74) returns a closure: `resize_image()` → `normalize_image()` → optionally `augment_image()` (training only). The closure pattern avoids re-specifying parameters on every call.

#### 19 — CustomDataset

[`CustomDataset.__getitem__()`](../multi-model/lib/preprocessing/dataset.py:87) loads one sample: read CSV row → construct image path → `_load_image()` → apply image pipeline → optionally augment → build `label_dict` from CSV columns. Returns `(processed_image, label_dict)`.

---

### Services Layer

#### 20 — Predictor

[`Predictor.predict_single()`](../multi-model/lib/services/predictor.py:55) converts logits to predictions: `model.eval()` → `torch.no_grad()` → forward pass → for each attribute: `softmax()` → `argmax()` → lookup label name. Returns dict with label + confidence per attribute.

#### 21 — FeatureExtractor

[`FeatureExtractor.extract()`](../multi-model/lib/services/feature_extractor.py:52) runs end-to-end feature extraction: visual encoder → OCR → clean text → tokenize → text encoder. Falls back to zero text features if no text detected.

#### 22 — Postprocessor

[`format_prediction_result()`](../multi-model/lib/services/postprocessor.py:21) strips internal metrics (confidence scores, numeric indices) from the prediction result, keeping only human-readable label names. Renames `predicted_label_text` → `predicted_label`.

---

### Utilities Layer

#### 23a — Load Config

[`load_config()`](../multi-model/lib/utils/config.py:9) reads a JSON file: check file exists → `open()` → `json.load()`. Single entry point for all config loading.

#### 23b — Get Label Maps

[`get_label_maps()`](../multi-model/lib/utils/config.py:49) supports two config formats (flat `label_maps` key and nested `ATTRIBUTES` key) for backward compatibility. Merges both sources into `Dict[str, List[str]]`.

#### 24 — Setup Directories

[`setup_directories()`](../multi-model/lib/utils/lifecycle.py:7) creates all required directories (uploads, saved_models, local/models, local/ocr, local/cache) with `mkdir(parents=True, exist_ok=True)`. Called once at startup.

#### 26 — Training Logger

[`TrainingLogger.log_metrics()`](../multi-model/lib/utils/logging.py:30) persists per-epoch metrics to CSV: validate input → initialize CSV with headers (first call) → append row → flush. Ensures data survives training crashes.

---

### Scripts

#### 27 — Train Model Script

[`main()`](../multi-model/scripts/train_model.py:32) orchestrates the training loop: parse CLI args → `load_config()` → `build_training_pipeline()` → for each epoch: warmup → `train_epoch()` → `validate_epoch()` → `scheduler.step()` → log metrics → if best accuracy: atomic checkpoint write (`.tmp` → `os.replace()`).

---

### Use Cases

#### 31a — Predict Image

[`predict_image()`](../multi-model/use_cases/prediction/predict_image.py:47) is the core prediction use case: create Predictor if needed → OCR extract → clean text → prepare image tensor → prepare text tensors → `predict_single()` → `format_prediction_result()` → add OCR text, filename, keywords, monetary mention, call-to-action, objects detected.

#### 31b — Prepare Image Tensor

[`_prepare_image_tensor()`](../multi-model/use_cases/prediction/predict_image.py:185) converts any image format to model input: `_to_numpy_array()` → handle grayscale (1ch → 3ch) → handle RGBA (4ch → drop alpha) → normalize uint8 to float32 → transpose HWC→CHW → `torch.from_numpy()` → `.unsqueeze(0)` → `Tensor(1, C, H, W)`.

#### 35 — Train Epoch

[`train_epoch()`](../multi-model/use_cases/training/train_model.py:103) runs one training pass: `model.train()` → for each batch: move to device → handle label format → prepare text inputs → zero grad → forward → sum multi-head losses → backward → optimizer step → track running loss and accuracy.

#### 36 — Evaluate Model

[`evaluate_model()`](../multi-model/use_cases/training/evaluate.py:21) runs inference without gradients: `model.eval()` → `torch.no_grad()` → for each batch: forward → argmax → collect predictions and labels → return for metric computation.

---

## Cross-Module Dependencies

The [`cross_module_dependency_map.mmd`](../docs/diagrams/cross_module_dependency_map.mmd) diagram shows how modules import from each other:

| Layer | Module | Depends On |
|-------|--------|------------|
| **App** | `app.app` | `lib.utils.lifecycle`, `use_cases.prediction.pipeline`, `app.predict`, `lib.ocr.factory` |
| **App** | `app.predict` | `lib.ocr.engine`, `lib.services.predictor`, `use_cases.prediction.predict_image` |
| **Use Cases** | `prediction.pipeline` | `lib.models.factory`, `lib.ocr.factory`, `lib.services.predictor`, `lib.utils.config` |
| **Use Cases** | `prediction.predict_image` | `lib.ocr.engine`, `lib.preprocessing.text.*`, `lib.services.postprocessor`, `lib.services.predictor` |
| **Use Cases** | `training.pipeline` | `lib.models.factory`, `lib.preprocessing.dataset`, `lib.utils.config`, `use_cases.training.train_model` |
| **Use Cases** | `training.train_model` | `torch`, `DataLoader` |
| **Use Cases** | `training.evaluate` | `torch` |
| **Lib** | `models.fg_mfn` | `lib.models.text`, `lib.models.visual` |
| **Lib** | `services.predictor` | `lib.models.fg_mfn` |
| **Lib** | `services.feature_extractor` | `lib.models.text`, `lib.models.visual`, `lib.ocr.engine`, `lib.preprocessing.text.*` |
| **Lib** | `preprocessing.dataset` | `lib.preprocessing.image.transforms`, `lib.utils.config` |

Key architectural observations:
- The **App layer** depends only on use-case entry points and lib utilities — never directly on model internals
- **Use cases** orchestrate lib components but don't depend on each other (training and prediction are independent)
- The **Lib layer** is the deepest with no upward dependencies — models, OCR, preprocessing, and services are self-contained
- [`FG_MFN`](../multi-model/lib/models/fg_mfn.py:39) is the central hub, depending only on [`TextModule`](../multi-model/lib/models/text.py:28) and [`VisualModule`](../multi-model/lib/models/visual.py:27)

---

## Data Types Reference

These are the primary tensor shapes and data structures that flow through the diagrams:

| Type | Shape / Structure | Used By |
|------|-------------------|---------|
| `images` | `Tensor(B, C, H, W)` — float32, C=3, H=W=224 | [`FG_MFN.forward()`](../multi-model/lib/models/fg_mfn.py:259), [`VisualModule.forward()`](../multi-model/lib/models/visual.py:128) |
| `input_ids` | `Tensor(B, seq_len)` — int64, padded to 512 | [`FG_MFN.forward()`](../multi-model/lib/models/fg_mfn.py:259), [`TextModule.forward()`](../multi-model/lib/models/text.py:84) |
| `attention_mask` | `Tensor(B, seq_len)` — int64, 0/1 | [`FG_MFN.forward()`](../multi-model/lib/models/fg_mfn.py:259), [`TextModule.forward()`](../multi-model/lib/models/text.py:84) |
| `visual_features` | `Tensor(B, hidden_dim)` — float32 | [`FG_MFN._fuse_features()`](../multi-model/lib/models/fg_mfn.py:294) |
| `text_features` | `Tensor(B, hidden_dim)` — float32 | [`FG_MFN._fuse_features()`](../multi-model/lib/models/fg_mfn.py:294) |
| `fused_features` | `Tensor(B, hidden_dim)` — float32 | Shared FC layer input |
| `logits` | `Tensor(B, num_classes)` — per attribute | Attribute classification heads |
| `label_dict` | `Dict[str, int]` — attribute → class index | [`CustomDataset.__getitem__()`](../multi-model/lib/preprocessing/dataset.py:87) |
| `label_maps` | `Dict[str, List[str]]` — attribute → label names | [`Predictor.predict_single()`](../multi-model/lib/services/predictor.py:55) |
| `prediction_result` | `Dict[str, Any]` — attribute → label + confidence | [`predict_image()`](../multi-model/use_cases/prediction/predict_image.py:47) |
| `ocr_result` | `Tuple[str, float]` — (text, confidence) | [`OCREngine.extract_text()`](../multi-model/lib/ocr/engine.py:22) |
| `config` | `Dict[str, Any]` — parsed JSON | All pipeline builders |

> **B** = batch size, **C** = channels (3 for RGB), **H/W** = height/width (224×224), **hidden_dim** = configured (default 512), **seq_len** = max token length (512)

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [`docs/dataflow_diagrams.md`](../docs/dataflow_diagrams.md) | Master document with all 36 Mermaid diagrams inline |
| [`docs/function_explanations.md`](../docs/function_explanations.md) | Detailed Input/Steps/Output/Purpose tables for every top-level function |
| [02 — Architecture](02_architecture.md) | System architecture, layered design, and patterns |
| [03 — Model Reference](03_model_reference.md) | FG_MFN model deep dive with layer details |
| [07 — Training Guide](07_training_guide.md) | Training pipeline with loss computation and checkpointing |
