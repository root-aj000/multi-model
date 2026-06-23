# Slide 24 — Project Structure

## What to Say (Speaker Notes)
"Here is the **project structure** — the folder layout of the entire codebase. The main `multi-model/` folder contains 6 subfolders: `app/` for the FastAPI app, `configs/` for YAML configuration files, `lib/` for the core domain code, `scripts/` for CLI entry points, `tests/` for the pytest suite, and `use_cases/` for business logic. There's also a `frontend/` folder for the Next.js app, a `docs/` folder with 13 markdown guides, and a `notebook/` folder for Jupyter experiments. This clean separation means each part can be **developed, tested, and deployed independently**. For example, you can run the tests without starting the API, or build the frontend without touching the model code."

## What to Show on Screen

```
📁 PROJECT STRUCTURE

   multi-model/
   ├── app/              ← FastAPI HTTP layer
   │   ├── main.py
   │   └── routes/
   ├── configs/          ← YAML configuration
   │   └── default.yaml
   ├── lib/              ← Core domain
   │   ├── models/       ← FG_MFN, ResNet, DistilBERT
   │   ├── ocr/          ← EasyOCR, PaddleOCR
   │   ├── preprocessing/← Image + text pipelines
   │   ├── services/     ← Training, evaluation
   │   └── utils/        ← Helpers
   ├── scripts/          ← CLI entry points
   │   ├── train.py
   │   └── predict.py
   ├── tests/            ← pytest suite
   │   └── unit/
   └── use_cases/        ← Business logic
       └── predict_ad.py

   frontend/             ← Next.js app
   docs/                 ← 13 markdown guides
   notebook/             ← Jupyter experiments
```

## Visual Suggestion
- Use a **tree view** with folder icons.
- Color-code by layer: **blue** for app, **green** for lib, **orange** for scripts, **purple** for tests.
- Highlight the **4-layer architecture** from slide 9.

## Key Talking Points
- This structure **enforces the clean architecture** from slide 9.
- Each folder has a **single responsibility**.
- New developers can **find code quickly** — predictable layout.
