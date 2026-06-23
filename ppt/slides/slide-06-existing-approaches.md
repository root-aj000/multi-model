# Slide 6 — Existing Approaches (Literature)

## What to Say (Speaker Notes)
"Before building my own system, I studied **what others have done**. There are 3 main approaches in the literature. **First**, single-task CNN classifiers — one model per attribute. This is expensive (you need 9 models) and ignores the fact that attributes are correlated. **Second**, CLIP-style contrastive models — these are very powerful but need huge amounts of data, which I don't have. **Third**, late-fusion vs. early-fusion — late fusion combines separate model outputs, early fusion combines raw inputs; both have trade-offs. **My approach** is different: I use **mid-level cross-modal attention fusion** with multi-task heads. This means the image and text features interact **inside** the model, not just at the output. I drew inspiration from papers like **CLIP, VisualBERT, MMBT, and ALBEF**."

## What to Show on Screen

```
📚 EXISTING APPROACHES

┌─────────────────────────┬──────────────────────────────────┐
│ Approach                │ Limitation                       │
├─────────────────────────┼──────────────────────────────────┤
│ Single-task CNNs        │ 9 models needed, ignores         │
│ (one per attribute)     │ correlation between attributes   │
├─────────────────────────┼──────────────────────────────────┤
│ CLIP-style contrastive  │ Needs millions of image-text     │
│                         │ pairs — too data-hungry           │
├─────────────────────────┼──────────────────────────────────┤
│ Late fusion             │ Modalities don't interact        │
│ (concat at output)      │ inside the model                  │
├─────────────────────────┼──────────────────────────────────┤
│ Early fusion            │ Hard to align image + text       │
│ (concat at input)       │ features at raw level             │
└─────────────────────────┴──────────────────────────────────┘

⭐ OUR APPROACH:
   Mid-level CROSS-MODAL ATTENTION fusion
   + Multi-task heads (9 outputs in one forward pass)

📖 Inspired by: CLIP, VisualBERT, MMBT, ALBEF
```

## Visual Suggestion
- Use a **comparison table** with the existing approaches on the left and our approach highlighted on the right.
- Add small **paper icons** next to the references at the bottom.

## Key Talking Points
- Don't go too deep into the papers — just mention the names.
- Emphasize that **our approach is novel in the combination** — cross-modal attention + multi-task heads.
