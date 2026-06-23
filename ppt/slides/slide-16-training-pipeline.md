# Slide 16 — Training Pipeline

## What to Say (Speaker Notes)
"Now let's talk about **training**. This is where the model actually learns. I use the **AdamW optimizer** with **differential learning rates** — this is a key trick. The 9 heads and the shared FC layer learn at a **higher rate of 2e-4**, but the DistilBERT text encoder learns at a **much lower rate of 1.5e-5** — that's 10 to 25 times slower. Why? Because DistilBERT is already pre-trained on huge text data, and we don't want to **catastrophically forget** what it knows. The learning rate follows a **warmup + cosine annealing** schedule — it ramps up linearly for the first 5 epochs, then decays following a cosine curve. For **regularization**, I use **label smoothing 0.2** (which prevents the model from being overconfident), **dropout 0.5** (which randomly drops half the neurons during training), and **weight decay 0.01** (which keeps the weights small). I also use **early stopping with patience 10** — if the validation loss doesn't improve for 10 epochs, training stops automatically. Finally, I use **per-attribute loss weights** — the engagement heads (CTR, shares) are down-weighted to 0.1 because they're harder to predict from just image+text."

## What to Show on Screen

```
🏋️ TRAINING PIPELINE

   ⚙️  OPTIMIZER: AdamW
       • Heads / Shared FC  →  lr = 2e-4
       • DistilBERT encoder →  lr = 1.5e-5  (10–25× lower)

   📈 SCHEDULER: Linear Warmup + Cosine Annealing
       • Warmup: 5 epochs (linear ramp up)
       • Decay: cosine curve down

   🛡️  REGULARIZATION:
       • Label smoothing = 0.2
       • Dropout = 0.5
       • Weight decay = 0.01

   ⏹️  EARLY STOPPING:
       • Patience = 10 epochs
       • Monitors validation loss

   ⚖️  PER-ATTRIBUTE LOSS WEIGHTS:
       • Content heads (theme, sentiment, ...) → 1.0
       • Engagement heads (CTR, shares)      → 0.1
         (harder to predict from image+text alone)
```

## Visual Suggestion
- Show a **learning rate curve** — ramps up, then cosine decay.
- Use **icons** for each regularization technique (shield for label smoothing, etc.).
- Highlight the **differential LR** as the key trick.

## Key Talking Points
- **Differential learning rates** are critical when fine-tuning pre-trained models.
- **Early stopping** prevents wasting compute on a model that has already converged.
- **Per-attribute loss weights** acknowledge that some attributes are inherently harder.
