## Full Coding Standards — `rules.md`

---

### Rule 1: Code Must Be Human Readable

Every line of code should be understandable by someone who knows Python but does not know this codebase.

**Bad:**
```python
r = [x for x in d if x['s'] > 0.5 and x['t'] in ['a','b','c']]
```

**Good:**
```python
high_confidence_predictions = [
    prediction
    for prediction in all_predictions
    if prediction['score'] > 0.5
    and prediction['type'] in VALID_PREDICTION_TYPES
]
```

**Rules:**
- No single-letter variable names except loop counters (`i`, `j`) and math (`x`, `y`)
- No abbreviations unless they are industry-standard (`ocr`, `cfg`, `url`, `id`)
- One logical operation per line when chaining would hide intent
- Prefer named variables over inline expressions when the expression is longer than 40 characters

---

### Rule 2: Every Function Definition Gets a Docstring

Every function and method must have a docstring directly below its `def` line. The docstring explains:
1. What the function does (one sentence)
2. What each parameter is
3. What it returns
4. Any exceptions it raises

**Bad:**
```python
def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text
```

**Good:**
```python
def clean_text(text: str) -> str:
    """
    Remove OCR artifacts and normalize whitespace from raw text.

    Strips leading/trailing whitespace, collapses multiple spaces into one,
    and removes non-printable characters that EasyOCR sometimes produces.

    Args:
        text: Raw string output from an OCR engine.

    Returns:
        Cleaned string with normalized whitespace.

    Raises:
        TypeError: If text is not a string.
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text
```

---

### Rule 3: No Function Does More Than One Thing

Each function has exactly one responsibility. If you need the word "and" to describe what a function does, it should be two functions.

**Bad:**
```python
def load_and_preprocess_image(path):
    """Load image from disk and resize and normalize it."""
    image = PIL.Image.open(path)
    image = image.resize((224, 224))
    image = numpy.array(image) / 255.0
    return image
```

**Good:**
```python
def load_image(path: str) -> PIL.Image.Image:
    """Load an image from disk and return a PIL Image object."""
    return PIL.Image.open(path)


def resize_image(image: PIL.Image.Image, size: tuple) -> PIL.Image.Image:
    """Resize a PIL image to the given (width, height) dimensions."""
    return image.resize(size)


def normalize_image(image: numpy.ndarray) -> numpy.ndarray:
    """Scale image pixel values from 0-255 range to 0.0-1.0 range."""
    return image / 255.0
```

---

### Rule 4: No Magic Numbers or Strings

Every constant must have a name that explains what it means.

**Bad:**
```python
if confidence > 0.7:
    ...
image = image.resize((224, 224))
if engine not in ['easyocr', 'paddleocr']:
    ...
```

**Good:**
```python
MINIMUM_CONFIDENCE_THRESHOLD = 0.7
IMAGE_INPUT_SIZE = (224, 224)
SUPPORTED_OCR_ENGINES = ['easyocr', 'paddleocr']

if confidence > MINIMUM_CONFIDENCE_THRESHOLD:
    ...
image = image.resize(IMAGE_INPUT_SIZE)
if engine not in SUPPORTED_OCR_ENGINES:
    ...
```

**Where constants go:**
- Constants used in one file → top of that file
- Constants used across multiple files → `lib/utils/config.py` or a dedicated `lib/utils/constants.py`

---

### Rule 5: No Hidden Dependencies

Every dependency a function needs must be passed in as a parameter. No reading from global variables, no module-level singletons, no importing and using a global object inside a function body.

**Bad:**
```python
# At top of file
_model = None

def predict(image):
    """Run prediction on image."""
    global _model          # hidden dependency
    if _model is None:
        _model = load_model()
    return _model(image)
```

**Good:**
```python
def predict(image: numpy.ndarray, model: FG_MFN) -> dict:
    """
    Run prediction on a single image using the provided model.

    Args:
        image: Preprocessed image array of shape (3, 224, 224).
        model: A loaded FG_MFN model instance in eval mode.

    Returns:
        Dictionary of attribute names to predicted label strings.
    """
    return model(image)
```

---

### Rule 6: Explicit Error Messages

When raising an exception, the message must tell the developer exactly what went wrong and what the valid options are.

**Bad:**
```python
if engine not in SUPPORTED_OCR_ENGINES:
    raise ValueError("Invalid engine")
```

**Good:**
```python
if engine not in SUPPORTED_OCR_ENGINES:
    raise ValueError(
        f"OCR engine '{engine}' is not supported. "
        f"Valid options are: {SUPPORTED_OCR_ENGINES}"
    )
```

---

### Rule 7: Type Hints on Every Function Signature

Every parameter and return value must have a type hint.

**Bad:**
```python
def tokenize_text(text, max_length):
    ...
```

**Good:**
```python
def tokenize_text(text: str, max_length: int) -> dict[str, torch.Tensor]:
    ...
```

**Allowed shorthands:**
```python
from typing import Optional, List, Dict, Tuple, Any, Callable
```

---

### Rule 8: One Import Per Line, Grouped and Ordered

Imports must be in three groups, separated by blank lines, in this order:
1. Standard library
2. Third-party packages
3. Internal project modules

**Bad:**
```python
import torch, numpy, os, sys
from models.fg_mfn import FG_MFN
import pandas as pd
from typing import List
```

**Good:**
```python
import os
import sys
from typing import List, Dict, Optional

import numpy
import pandas as pd
import torch

from lib.models.fg_mfn import FG_MFN
from lib.utils.config import load_config
```

---

### Rule 9: No File Longer Than 300 Lines

If a file exceeds 300 lines, it is doing too many things. Split it.

**Why 300:**
- Fits in one screen with context
- Forces single responsibility at the file level
- Makes code review tractable

**Exception:** Files that are pure data (large constants, label maps) may exceed this limit with a comment explaining why.

---

### Rule 10: No Commented-Out Code in Committed Files

Commented-out code is not documentation. It creates confusion about what is active.

**Bad:**
```python
def predict(image, model):
    # old_result = legacy_predict(image)  # keeping this just in case
    result = model(image)
    # result = postprocess(result)  # disabled for now
    return result
```

**Good:**
```python
def predict(image: numpy.ndarray, model: FG_MFN) -> dict:
    """Run model prediction on a preprocessed image."""
    return model(image)
```

If you need to preserve old logic, use git. Do not commit commented code.

---

### Rule 11: Log at the Right Level

Use the correct log level. Do not use `print()` in library code.

| Level | When to use |
|-------|------------|
| `logger.debug()` | Internal state useful only during development |
| `logger.info()` | Normal operational events (model loaded, epoch started) |
| `logger.warning()` | Something unexpected happened but execution continues |
| `logger.error()` | Something failed; operation could not complete |
| `logger.critical()` | Application cannot continue at all |

**Bad:**
```python
print("Model loaded")
print("ERROR: file not found")
```

**Good:**
```python
logger.info("Model loaded from checkpoint: %s", checkpoint_path)
logger.error("Checkpoint file not found at path: %s", checkpoint_path)
```

**Rule:** `print()` is only allowed in `scripts/` files for CLI output to the user.

---

### Rule 12: Test File Mirrors Source File

Every source file must have a corresponding test file at the same relative path under `tests/`.

| Source file | Test file |
|-------------|-----------|
| `lib/utils/config.py` | `tests/unit/lib/utils/test_config.py` |
| `lib/preprocessing/text/cleaner.py` | `tests/unit/lib/preprocessing/text/test_cleaner.py` |
| `use_cases/training/evaluate.py` | `tests/unit/use_cases/training/test_evaluate.py` |

---

### Rule 13: No Try/Except Without a Specific Exception Type

Never catch all exceptions silently.

**Bad:**
```python
try:
    result = model(image)
except:
    return None
```

**Also bad:**
```python
try:
    result = model(image)
except Exception:
    return None
```

**Good:**
```python
try:
    result = model(image)
except torch.cuda.OutOfMemoryError as error:
    logger.error("GPU out of memory during inference: %s", error)
    raise RuntimeError("Inference failed due to insufficient GPU memory") from error
```

---

## Quick Reference Card

Print this and keep it next to your screen.

```
CODING STANDARDS — QUICK REFERENCE

1.  READABLE       Variables and functions named for what they mean, not how short they are
2.  DOCSTRING      Every function: what it does, args, returns, raises
3.  ONE THING      If you need "and" to describe it, split it into two functions
4.  NO MAGIC       Every number and string constant has a named variable
5.  NO GLOBALS     Every dependency is a parameter, never a global or singleton
6.  CLEAR ERRORS   Error messages say what went wrong AND what is valid
7.  TYPE HINTS     Every parameter and return value has a type annotation
8.  IMPORTS        stdlib → third-party → internal, one per line, three groups
9.  300 LINES MAX  If the file is longer, it is doing too many things
10. NO DEAD CODE   No commented-out code in committed files, use git instead
11. LOG LEVELS     Use logger.*, not print(), in all lib/ and use_cases/ code
12. MIRROR TESTS   Every source file has a matching test file
13. SPECIFIC CATCH Never catch bare Exception or use bare except:
```