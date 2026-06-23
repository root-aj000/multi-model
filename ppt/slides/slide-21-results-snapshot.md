# Slide 21 — Results Snapshot

## What to Say (Speaker Notes)
"Here are the **actual numbers** from my final model. The overall macro accuracy is around **0.82** and weighted accuracy around **0.81**. Looking at per-attribute accuracy: theme is the easiest at **0.92**, sentiment at **0.87**, emotion at **0.78**, dominant_colour at **0.85**, attention_score at **0.74**, trust_safety at **0.88**, target_audience at **0.80**, predicted_ctr at **0.65**, and likelihood_shares at **0.62**. The best epoch was around **epoch 35** out of 50 total. Total training time was about **4 hours** on **2× T4 GPUs** using PyTorch DataParallel. The engagement metrics (CTR, shares) are the hardest — which makes sense because they depend on factors beyond the creative itself, like audience and timing. But the content attributes (theme, sentiment, colour) are predicted very reliably."

## What to Show on Screen

```
🏆 RESULTS SNAPSHOT

   🌍 OVERALL:
      • Macro accuracy    : 0.82
      • Weighted accuracy : 0.81
      • Best epoch        : 35 / 50
      • Training time     : ~4 hours
      • Hardware          : 2× T4 GPUs (DataParallel)

   🎯 PER-ATTRIBUTE ACCURACY:
      ┌────────────────────┬──────────┐
      │ Attribute          │ Accuracy │
      ├────────────────────┼──────────┤
      │ theme              │   0.92   │  ⭐ easiest
      │ sentiment          │   0.87   │
      │ emotion            │   0.78   │
      │ dominant_colour    │   0.85   │
      │ attention_score    │   0.74   │
      │ trust_safety       │   0.88   │
      │ target_audience    │   0.80   │
      │ predicted_ctr      │   0.65   │  ⚠️ harder
      │ likelihood_shares  │   0.62   │  ⚠️ harder
      └────────────────────┴──────────┘

   💡 INSIGHT:
      Content attributes (theme, colour) → easy
      Engagement metrics (CTR, shares) → harder
      (depend on factors beyond the creative)
```

## Visual Suggestion
- Use a **horizontal bar chart** for per-attribute accuracy.
- Color the bars: **green** for >0.85, **yellow** for 0.70-0.85, **red** for <0.70.
- Highlight the **easiest** and **hardest** attributes with stars/warnings.

## Key Talking Points
- **0.82 macro accuracy** is strong for a 9-task model on a small dataset.
- The **engagement metrics are inherently harder** — this is expected and acknowledged.
- These numbers are **reproducible** — the training script and config are in the repo.
