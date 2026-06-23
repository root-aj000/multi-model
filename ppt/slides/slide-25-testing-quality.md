# Slide 25 — Testing & Quality

## What to Say (Speaker Notes)
"Quality is important, so I wrote a **comprehensive test suite** using **pytest**. The tests live in `tests/unit/` and cover all the major components: model architecture, OCR engines, preprocessing pipelines, training loop, evaluation metrics, and the REST API. There's a `run_tests.sh` script that runs the full suite with one command. On the frontend side, I use **ESLint** for TypeScript linting. On the backend side, I follow **PEP 8** style guidelines. The tests caught many bugs during development — including 28 documented bugs that were fixed before the final release. This gave me confidence that the model and API work correctly across many edge cases."

## What to Show on Screen

```
🧪 TESTING & QUALITY

   BACKEND (Python):
   ┌────────────────────┬──────────────────────────┐
   │ Framework          │ pytest                   │
   │ Location           │ tests/unit/              │
   │ Coverage           │ models, ocr, preprocess, │
   │                    │ training, eval, api      │
   │ Run command        │ ./run_tests.sh           │
   └────────────────────┴──────────────────────────┘

   FRONTEND (TypeScript):
   ┌────────────────────┬──────────────────────────┐
   │ Linter             │ ESLint                   │
   │ Style guide        │ Airbnb config            │
   └────────────────────┴──────────────────────────┘

   📊 STATS:
      • 28 documented bugs fixed
      • Tests run on every change
      • Edge cases covered (empty input, large file, etc.)

   🎯 BENEFIT:
      Confidence that the system works correctly
      across many scenarios
```

## Visual Suggestion
- Show a **green checkmark** for passing tests.
- Add a **pie chart** showing test coverage by component.
- Include a **screenshot** of the test runner output.

## Key Talking Points
- **Tests are not optional** — they're part of the development workflow.
- **28 bugs fixed** shows the value of testing.
- The test suite runs in **under 2 minutes** — fast feedback loop.
