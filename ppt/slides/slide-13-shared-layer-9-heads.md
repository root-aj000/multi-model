# Slide 13 — Shared Layer + 9 Attribute Heads

## What to Say (Speaker Notes)
"After fusion, we have a 512-d vector that summarizes both the image and the text. This vector goes through a **shared fully-connected layer** — a Linear layer, ReLU activation, and Dropout with probability 0.5. The shared layer learns **common patterns** that are useful for all 9 attributes. Then the vector is split into **9 independent heads**, one per attribute. Each head is just a single Linear layer that outputs the class probabilities for that attribute. For example, the 'theme' head outputs 9 probabilities (one per theme class), the 'sentiment' head outputs 3 probabilities, and so on. This is **multi-task learning** — one shared backbone, 9 task-specific heads. The advantage is that the shared representation acts as a **regularizer** — the model can't overfit to one task without affecting the others."

## What to Show on Screen

```
🎯 SHARED LAYER + 9 ATTRIBUTE HEADS

   Fused vector (512-d)
            │
            ▼
   ┌────────────────────────┐
   │  Shared FC             │
   │  Linear(512 → 512)     │
   │  ReLU                  │
   │  Dropout(0.5)          │
   └────────────┬───────────┘
                │
   ┌────────────┼────────────┬────────────┬─────────┐
   ▼            ▼            ▼            ▼         ▼
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│theme │   │senti-│   │emotion│  │domin.│   │atten-│  ...
│  9   │   │ment  │   │   5   │  │colour│   │tion  │
│classes   │  3   │   │classes│  │  10  │   │  3   │
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘

   📋 THE 9 ATTRIBUTES:
   ┌────┬────────────────────┬─────────┐
   │ #  │ Attribute          │ Classes │
   ├────┼────────────────────┼─────────┤
   │ 1  │ theme              │    9    │
   │ 2  │ sentiment          │    3    │
   │ 3  │ emotion            │    5    │
   │ 4  │ dominant_colour    │   10    │
   │ 5  │ attention_score    │    3    │
   │ 6  │ trust_safety       │    3    │
   │ 7  │ target_audience    │    6    │
   │ 8  │ predicted_ctr      │    3    │
   │ 9  │ likelihood_shares  │    3    │
   └────┴────────────────────┴─────────┘
```

## Visual Suggestion
- Show the **9 heads as 9 small boxes** fanning out from the shared layer.
- Use a **table** for the 9 attributes with class counts.
- Color-code the heads by category: **content** (theme, sentiment, emotion), **visual** (colour, attention), **business** (CTR, shares, audience, safety).

## Key Talking Points
- **Multi-task learning** is efficient — one model, 9 outputs.
- The **shared layer** is where most of the "intelligence" lives.
- The **9 heads** are tiny — just one Linear layer each.
