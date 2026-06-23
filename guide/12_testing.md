# 12 — Testing

Test suite structure, running tests, writing new tests, and conventions.

## Test Suite Overview

The test suite uses **pytest** with a unit-test structure mirroring the source code layout.

```
multi-model/tests/
├── conftest.py                          # Adds project root to sys.path
├── fixtures/                            # Shared test fixtures
├── integration/                         # Integration tests (placeholder)
└── unit/
    ├── lib/
    │   ├── models/
    │   │   ├── test_factory.py          # Model factory tests
    │   │   ├── test_fg_mfn.py           # FG_MFN model tests
    │   │   ├── test_text.py             # TextModule tests
    │   │   └── test_visual.py           # VisualModule tests
    │   ├── ocr/
    │   │   ├── test_easyocr.py          # EasyOCREngine tests
    │   │   ├── test_engine.py           # OCREngine ABC tests
    │   │   ├── test_factory.py          # OCR factory tests
    │   │   └── test_paddleocr.py        # PaddleOCREngine tests
    │   ├── preprocessing/
    │   │   ├── test_dataset.py          # CustomDataset tests
    │   │   ├── image/
    │   │   │   ├── test_pipeline.py     # Image pipeline tests
    │   │   │   └── test_transforms.py   # Image transform tests
    │   │   └── text/
    │   │       ├── test_cleaner.py      # clean_text() tests
    │   │       ├── test_pipeline.py     # Text pipeline tests
    │   │       └── test_tokenizer.py    # tokenize_text() tests
    │   ├── services/
    │   │   ├── test_feature_extractor.py
    │   │   ├── test_postprocessor.py
    │   │   └── test_predictor.py
    │   └── utils/
    │       ├── test_cache.py
    │       ├── test_config.py
    │       ├── test_lifecycle.py
    │       └── test_logging.py
    └── use_cases/
        ├── prediction/
        │   ├── test_pipeline.py
        │   ├── test_predict_batch.py
        │   └── test_predict_image.py
        └── training/
            ├── test_evaluate.py
            ├── test_pipeline.py
            └── test_train_model.py
```

---

## Running Tests

### Run all tests

```bash
cd multi-model/
python -m pytest tests/ -v
```

### Run a specific test module

```bash
python -m pytest tests/unit/lib/models/test_fg_mfn.py -v
```

### Run a specific test function

```bash
python -m pytest tests/unit/lib/models/test_fg_mfn.py::test_fg_mfn_init_with_valid_config -v
```

### Run by keyword

```bash
python -m pytest tests/ -v -k "predictor"
```

### Run with coverage

```bash
python -m pytest tests/ --cov=lib --cov=use_cases --cov=app --cov-report=term-missing
```

---

## Test Configuration

### [`conftest.py`](../multi-model/tests/conftest.py)

Adds the project root to `sys.path` so that internal modules can be imported:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

This allows imports like `from lib.models.fg_mfn import FG_MFN` to work in tests.

---

## Test Conventions

### Naming

- Test files: `test_<module_name>.py`
- Test functions: `test_<what_is_tested>_<condition>()`
- Helper functions: prefixed with underscore (e.g., `_make_minimal_config()`)

### Test Structure

Each test follows the **Arrange-Act-Assert** pattern:

```python
def test_predictor_predict_single_returns_dict():
    # Arrange
    model = _make_mock_model()
    label_maps = {"sentiment": ["positive", "neutral", "negative"]}
    predictor = Predictor(model, label_maps)
    images = torch.randn(1, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones(1, 128)

    # Act
    result = predictor.predict_single(images, input_ids, attention_mask)

    # Assert
    assert isinstance(result, dict)
    assert "sentiment" in result
```

### Mocking Strategy

Heavy external dependencies (model loading, OCR engines) are mocked with `unittest.mock.MagicMock`:

```python
def _make_mock_model():
    model = MagicMock()
    sentiment_logits = torch.tensor([[0.1, 0.5, 0.4]])
    model.return_value = {"sentiment": sentiment_logits}
    model.eval.return_value = model
    return model
```

### Minimal Config Helper

For FG_MFN tests, a minimal config factory avoids loading real pretrained models:

```python
def _make_minimal_config():
    return {
        "IMAGE_BACKBONE": "resnet18",
        "TEXT_ENCODER": "distilbert-base-uncased",
        "HIDDEN_DIM": 32,
        "DROPOUT": 0.1,
        "FUSION_TYPE": "concat",
        "ATTRIBUTES": {
            "sentiment": {
                "num_classes": 3,
                "labels": ["positive", "neutral", "negative"],
            },
        },
    }
```

---

## Test Coverage by Module

### Models

| Test File | Tests |
|-----------|-------|
| [`test_fg_mfn.py`](../multi-model/tests/unit/lib/models/test_fg_mfn.py) | Valid config init, missing keys, non-dict config, invalid HIDDEN_DIM, forward pass shapes, fusion types |
| [`test_text.py`](../multi-model/tests/unit/lib/models/test_text.py) | TextModule init, forward pass, invalid encoder name |
| [`test_visual.py`](../multi-model/tests/unit/lib/models/test_visual.py) | VisualModule init, forward pass, unsupported backbone |
| [`test_factory.py`](../multi-model/tests/unit/lib/models/test_factory.py) | create_model(), load_model() with missing checkpoint |

### OCR

| Test File | Tests |
|-----------|-------|
| [`test_engine.py`](../multi-model/tests/unit/lib/ocr/test_engine.py) | OCREngine ABC cannot be instantiated directly |
| [`test_easyocr.py`](../multi-model/tests/unit/lib/ocr/test_easyocr.py) | EasyOCREngine interface |
| [`test_paddleocr.py`](../multi-model/tests/unit/lib/ocr/test_paddleocr.py) | PaddleOCREngine interface |
| [`test_factory.py`](../multi-model/tests/unit/lib/ocr/test_factory.py) | Unsupported engine raises ValueError, supported engines list |

### Preprocessing

| Test File | Tests |
|-----------|-------|
| [`test_cleaner.py`](../multi-model/tests/unit/lib/preprocessing/text/test_cleaner.py) | None input, special chars, whitespace collapse |
| [`test_tokenizer.py`](../multi-model/tests/unit/lib/preprocessing/text/test_tokenizer.py) | Empty text, tokenization output shapes |
| [`test_pipeline.py`](../multi-model/tests/unit/lib/preprocessing/text/test_pipeline.py) | Keyword extraction, monetary mention, CTA detection |
| [`test_transforms.py`](../multi-model/tests/unit/lib/preprocessing/image/test_transforms.py) | Resize, normalize, augment, pipeline composition |
| [`test_dataset.py`](../multi-model/tests/unit/lib/preprocessing/test_dataset.py) | CSV loading, missing files |

### Services

| Test File | Tests |
|-----------|-------|
| [`test_predictor.py`](../multi-model/tests/unit/lib/services/test_predictor.py) | None model/label_maps rejection, predict_single dict output, confidence inclusion |
| [`test_postprocessor.py`](../multi-model/tests/unit/lib/services/test_postprocessor.py) | Field exclusion, field renaming |
| [`test_feature_extractor.py`](../multi-model/tests/unit/lib/services/test_feature_extractor.py) | Feature extraction interface |

### Utils

| Test File | Tests |
|-----------|-------|
| [`test_config.py`](../multi-model/tests/unit/lib/utils/test_config.py) | FileNotFoundError, get_dataset_paths dict keys, get_label_maps empty config |
| [`test_cache.py`](../multi-model/tests/unit/lib/utils/test_cache.py) | Cache size, clear cache with/without confirm |
| [`test_lifecycle.py`](../multi-model/tests/unit/lib/utils/test_lifecycle.py) | Directory creation, cleanup |
| [`test_logging.py`](../multi-model/tests/unit/lib/utils/test_logging.py) | TrainingLogger CSV output, metrics validation |

### Use Cases

| Test File | Tests |
|-----------|-------|
| [`test_predict_image.py`](../multi-model/tests/unit/use_cases/prediction/test_predict_image.py) | Function importability and callability |
| [`test_predict_batch.py`](../multi-model/tests/unit/use_cases/prediction/test_predict_batch.py) | Function importability and callability |
| [`test_pipeline.py`](../multi-model/tests/unit/use_cases/prediction/test_pipeline.py) | Pipeline builder interface |
| [`test_evaluate.py`](../multi-model/tests/unit/use_cases/training/test_evaluate.py) | Evaluate model, compute metrics |
| [`test_pipeline.py`](../multi-model/tests/unit/use_cases/training/test_pipeline.py) | Training pipeline builder |
| [`test_train_model.py`](../multi-model/tests/unit/use_cases/training/test_train_model.py) | Mixup, warmup, train_epoch |

---

## Writing New Tests

### Template for a new unit test

```python
"""Tests for <module description>."""

import pytest
from lib.<module_path> import <function_or_class>


def test_<function>_<expected_behavior>():
    """<One-line description of what this test verifies>."""
    # Arrange
    input_data = ...

    # Act
    result = <function>(input_data)

    # Assert
    assert result == expected


def test_<function>_raises_on_<invalid_condition>():
    """<One-line description of the error case>."""
    with pytest.raises(<ExpectedError>, match="<partial error message>"):
        <function>(invalid_input)
```

### Testing with mocks

For functions that depend on expensive operations (model loading, OCR):

```python
from unittest.mock import MagicMock, patch

def test_predict_with_mock_ocr():
    ocr_engine = MagicMock()
    ocr_engine.extract_text.return_value = ("Hello World", 0.95)

    text, conf = ocr_engine.extract_text(image)

    assert text == "Hello World"
    assert conf == 0.95
    ocr_engine.extract_text.assert_called_once()
```

### Testing FG_MFN without pretrained weights

Use the minimal config helper with small HIDDEN_DIM to speed up tests:

```python
def _make_minimal_config():
    return {
        "IMAGE_BACKBONE": "resnet18",
        "TEXT_ENCODER": "distilbert-base-uncased",
        "HIDDEN_DIM": 32,       # Small for fast tests
        "DROPOUT": 0.1,
        "FUSION_TYPE": "concat",
        "ATTRIBUTES": {
            "sentiment": {"num_classes": 3, "labels": ["pos", "neu", "neg"]},
        },
    }
```

> **Note:** Even with `HIDDEN_DIM=32`, the first test run will download DistilBERT weights (~250MB). Subsequent runs use the HuggingFace cache at `~/.cache/huggingface/`.
