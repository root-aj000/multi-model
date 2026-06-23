# Slide 20 — Evaluation Visualizations

## What to Say (Speaker Notes)
"After training, I generate **3 evaluation plots** to understand the final model. The first is `confusion_matrices.png` — a heatmap for each of the 9 attributes. The diagonal shows correct predictions, off-diagonal shows mistakes. Darker diagonal = better. The second is `per_attribute_metrics.png` — bar charts showing accuracy, precision, recall, and F1 for each attribute side by side. The third is `macro_weighted_comparison.png` — a comparison of macro vs weighted metrics across attributes. These plots are saved as PNGs and also embedded in the web dashboard. They make it easy to **spot weaknesses** — for example, if the 'emotion' attribute has a low F1, we know to focus on improving that head."

## What to Show on Screen

```
🎨 EVALUATION VISUALIZATIONS

   ┌─────────────────────────────────────────────┐
   │  confusion_matrices.png                     │
   │  9 heatmaps (one per attribute)             │
   │  Diagonal = correct, off-diag = mistakes    │
   │                                             │
   │  Example (theme):                           │
   │        Pred →                               │
   │  True ↓  S  F  P  ...                       │
   │     S  [89  2  1  ...]  ← mostly correct    │
   │     F  [ 3 85  4  ...]                       │
   │     P  [ 1  2 90  ...]                       │
   └─────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────┐
   │  per_attribute_metrics.png                  │
   │  Bar charts: accuracy, precision,           │
   │  recall, F1 for each of 9 attributes        │
   └─────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────┐
   │  macro_weighted_comparison.png              │
   │  Side-by-side: macro vs weighted metrics    │
   └─────────────────────────────────────────────┘
```

## Visual Suggestion
- **Embed the actual confusion_matrices.png** from the project.
- Use **color coding**: green for high values, red for low values.
- Add **arrows** pointing to interesting cells in the confusion matrix.

## Key Talking Points
- **Confusion matrices** show not just *how many* mistakes, but *what kind* of mistakes.
- **Per-attribute bar charts** make it easy to compare heads at a glance.
- These plots are also shown in the **web dashboard** for stakeholders.
