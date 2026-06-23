# Slide 19 — Training Visualizations

## What to Say (Speaker Notes)
"Numbers are useful, but **visualizations** tell the story faster. I generate **2 key training plots**. The first is `training_curves.png` — a 2×2 grid showing training loss, validation loss, training accuracy, and validation accuracy across epochs. There's a marker on the best epoch. The second is `per_attribute_training.png` — separate accuracy curves for each of the 9 attributes. These plots help us **diagnose overfitting** — if training accuracy goes up but validation accuracy goes down, we're overfitting. They also help us see **which attributes are harder** to learn — some curves may plateau early, others may keep improving. Looking at these plots is how I tuned the hyperparameters — dropout, learning rate, label smoothing — until the curves looked healthy."

## What to Show on Screen

```
📈 TRAINING VISUALIZATIONS

   ┌─────────────────────────────────────────────┐
   │  training_curves.png                        │
   │  ┌──────────┬──────────┐                    │
   │  │ Train    │ Val      │                    │
   │  │ Loss     │ Loss     │  ↘ ↘               │
   │  ├──────────┼──────────┤                    │
   │  │ Train    │ Val      │                    │
   │  │ Accuracy │ Accuracy │  ↗ ↗  ★ best epoch │
   │  └──────────┴──────────┘                    │
   └─────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────┐
   │  per_attribute_training.png                 │
   │  9 small subplots — one per attribute       │
   │  Shows which attributes are easier/harder   │
   └─────────────────────────────────────────────┘

   🔍 WHAT TO LOOK FOR:
      • Train loss ↓, Val loss ↓ → healthy
      • Train loss ↓, Val loss ↑ → OVERFITTING
      • Train loss ↑, Val loss ↑ → UNDERFITTING
      • Plateau → try different LR or regularization
```

## Visual Suggestion
- **Embed the actual training_curves.png** from the project's `results/` folder.
- Use **arrows** to show the direction of healthy curves.
- Highlight the **best epoch marker** with a star.

## Key Talking Points
- **Visualizations are essential** — they catch problems that numbers hide.
- The **best epoch marker** tells us when to stop training (even without early stopping).
- Per-attribute curves reveal **which heads need more work**.
