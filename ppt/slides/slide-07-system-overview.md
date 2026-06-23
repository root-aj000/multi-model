# Slide 7 — Proposed System Overview

## What to Say (Speaker Notes)
"Here is the **big picture** of my system in one slide. The input is a **single ad image**. From that image, two things happen in parallel. **Branch 1**: the image goes through a CNN (ResNet-50) to extract visual features. **Branch 2**: the image goes through an OCR engine to extract the text inside it, and then through a Transformer (DistilBERT) to extract textual features. These two feature vectors are then **fused** using cross-modal attention — meaning the visual features pay attention to the text features and vice versa. The fused representation goes through a shared layer and then splits into **9 independent heads**, each predicting one attribute. The output is **9 structured labels** in a single forward pass. So the one-liner is: **one image in, nine structured insights out.**"

## What to Show on Screen

```
🔭 SYSTEM OVERVIEW — ONE IMAGE IN, NINE INSIGHTS OUT

                    ┌──────────────┐
                    │  AD IMAGE    │
                    │  (input)     │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     ┌────────────────┐        ┌────────────────┐
     │  ResNet-50     │        │  OCR Engine    │
     │  (CNN)         │        │  (EasyOCR)     │
     │                │        │      │         │
     │  Visual        │        │      ▼         │
     │  Features      │        │  DistilBERT    │
     │  (2048-d)      │        │  (Transformer) │
     │                │        │      │         │
     │                │        │  Text Features │
     │                │        │  (768-d)       │
     └────────┬───────┘        └────────┬───────┘
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  CROSS-MODAL ATTENTION  │
              │  (8 heads, dim 768)     │
              │  bidirectional          │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  SHARED FC + DROPOUT    │
              │  (512-d)                │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │ Head 1 │  │ Head 2 │  │  ...   │  ×9
         │ theme  │  │sentiment│  │ heads  │
         └────────┘  └────────┘  └────────┘
```

## Visual Suggestion
- This is the **most important diagram** of the talk — make it big and clear.
- Use **arrows** to show data flow.
- Color the **visual branch blue**, the **text branch green**, and the **fusion + heads orange**.
- Save this as `images/system-overview.png` and embed it.

## Key Talking Points
- Walk through the diagram **left to right**, top to bottom.
- Emphasize **"single forward pass"** — the model is fast at inference.
- Mention that the next few slides will zoom into each box.
