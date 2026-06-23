# Slide 14 — OCR Engine

## What to Say (Speaker Notes)
"You might be wondering — why do we need OCR? Can't we just use the image? The answer is: **text inside an ad is part of the creative**. A red circle with the word 'SALE' means something completely different from a red circle alone. So we need to **read the text** from the image. I support **two OCR engines** using the **Strategy pattern**. The default is **EasyOCR**, which is easy to use and works well for English. The optional alternative is **PaddleOCR**, which is faster and supports more languages. A **Factory** function called `create_ocr_engine` takes the engine name as input and returns the right engine object. The OCR models are cached locally in the `local/ocr/` folder so we don't have to download them every time. This design makes it trivial to swap engines — just change one config value."

## What to Show on Screen

```
🔤 OCR ENGINE — STRATEGY PATTERN

   ┌──────────────────────────────────────────────┐
   │           create_ocr_engine(name)            │
   │                  (Factory)                   │
   └──────────────┬───────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   ┌─────────┐         ┌──────────┐
   │ EasyOCR │         │PaddleOCR │
   │(default)│         │(optional)│
   └─────────┘         └──────────┘

   ❓ WHY OCR?
      • Text inside an ad IS part of the creative
      • "SALE" + red circle ≠ red circle alone
      • Captions, slogans, prices — all matter

   ⚙️  CONFIG:
      OCR_ENGINE=easyocr   ← default
      OCR_ENGINE=paddleocr ← optional

   📁 Models cached in: local/ocr/
```

## Visual Suggestion
- Show a **small ad image** on the left and the **extracted text** on the right with arrows.
- Use the **Strategy pattern diagram** — a context box with two strategy boxes inside.
- Add the **Factory** label on top.

## Key Talking Points
- OCR is the **bridge** between the visual and textual modalities.
- The **Strategy pattern** makes it easy to A/B test different OCR engines.
- EasyOCR is the default because it's the easiest to install and works well for English.
