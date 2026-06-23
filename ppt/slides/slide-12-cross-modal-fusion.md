# Slide 12 — Cross-Modal Fusion

## What to Say (Speaker Notes)
"This is the **heart** of the model — the **fusion step**. We have a 2048-d visual vector and a 768-d text vector. How do we combine them? The naive ways are **concatenation** (just stick them together) or **addition** (add them element-wise). These work but they don't let the two modalities **interact**. My approach uses **Cross-Modal Multi-Head Attention** with **8 heads** and **dimension 768**. Here's the idea: the visual features act as a **query**, and the text features act as a **key and value**. The visual branch then **pays attention** to the most relevant parts of the text. We also do this in reverse — text attends to visual. This is **bidirectional**. The result is two attended vectors, which we concatenate and project down to a **hidden dimension of 512**. Why attention? Because it **learns the alignment** between image regions and text tokens — for example, the word 'sale' should align with the red discount tag in the image."

## What to Show on Screen

```
🔀 CROSS-MODAL FUSION

   Visual (2048-d) ──┐
                     │   ┌──────────────────────────────┐
                     ├──▶│  Cross-Modal Multi-Head       │
   Text   (768-d)  ──┘   │  Attention                    │
                         │  • 8 heads                    │
                         │  • dim 768                    │
                         │  • BIDIRECTIONAL              │
                         │    (visual ↔ text)            │
                         └──────────────┬────────────────┘
                                        │
                                        ▼
                         Concat + Linear → 512-d fused vector

   ❓ WHY ATTENTION > CONCAT / ADD?
      • Learns INTER-MODAL ALIGNMENT
        (e.g., "sale" ↔ red discount tag)
      • Parameter-efficient
      • Each modality can focus on relevant parts of the other
```

## Visual Suggestion
- Draw **two arrows** between the visual and text boxes — one going each way (bidirectional).
- Use a **different color** for the attention block to make it stand out.
- Add a small **example** showing "sale" word aligning with a red tag in the image.

## Key Talking Points
- This is the **most technical slide** — keep it high-level.
- The key idea: **the two modalities talk to each other** before producing a final prediction.
- This is what makes the model **multi-modal** rather than just multi-input.
