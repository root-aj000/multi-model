# Slide 9 — System Architecture (Layered)

## What to Say (Speaker Notes)
"Now let me zoom out and show you the **software architecture** of the project. I followed a **4-layer clean architecture** pattern. The outermost layer is **scripts/** — these are command-line entry points like `train.py` and `predict.py`. The next layer is **app/** — this is the FastAPI HTTP layer that exposes REST endpoints. The next layer is **use_cases/** — these orchestrate the business logic, like 'predict attributes for an image'. The innermost layer is **lib/** — this contains the core domain code: models, OCR engines, preprocessing, services, and utilities. The **dependency rule** is strict: outer layers can depend on inner layers, but **never the other way around**. This makes the code easy to test, easy to swap components, and easy to maintain. I also used 3 design patterns: **Factory** (to create models and OCR engines), **Strategy** (to swap OCR engines), and **Dependency Injection** (to pass config and services around)."

## What to Show on Screen

```
🏛️ 4-LAYER CLEAN ARCHITECTURE

   ┌─────────────────────────────────────────────┐
   │  scripts/   ← CLI entry points             │  ← OUTERMOST
   │             (train.py, predict.py)          │
   ├─────────────────────────────────────────────┤
   │  app/       ← FastAPI HTTP layer            │
   │             (/health, /predict)             │
   ├─────────────────────────────────────────────┤
   │  use_cases/ ← Business logic orchestration  │
   │             (PredictAdUseCase)              │
   ├─────────────────────────────────────────────┤
   │  lib/       ← Core domain                   │  ← INNERMOST
   │             (models, ocr, preprocessing,    │
   │              services, utils)               │
   └─────────────────────────────────────────────┘

   ⬆ Dependency rule: OUTER → INNER only (never reverse)

   🎨 Design Patterns Used:
      • Factory       — create models / OCR engines by name
      • Strategy      — swap OCR engines at runtime
      • Dependency    — inject config + services
        Injection
```

## Visual Suggestion
- Use **concentric rectangles** or **nested boxes** to show the layers.
- Color each layer differently (outer = light, inner = dark).
- Add small icons for each design pattern on the right side.

## Key Talking Points
- This is **engineering rigor** — not just ML, but **software engineering**.
- The same architecture is used in **large production systems** at companies like Netflix and Uber.
- This makes the project **testable** (we'll see tests later).
