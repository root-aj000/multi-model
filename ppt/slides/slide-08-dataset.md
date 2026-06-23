# Slide 8 — Dataset

## What to Say (Speaker Notes)
"Let me tell you about the **data** I used. The dataset is from Kaggle, called **Ads Dataset with Images 2025-2026 v1**. It contains about **4,860 ad images**, each labeled with the 9 attributes I mentioned. I split it into training and validation sets. Before training, I applied **data augmentation** — horizontal flip, rotation, color jitter, and random crop — to artificially increase the variety of training samples and reduce overfitting. On the screen you can see 3 sample images with their labels. Notice how each ad has a clear theme, a dominant color, and a target audience — these are exactly the kinds of patterns the model learns to recognize."

## What to Show on Screen

```
📦 DATASET

Source:    Kaggle — "Ads Dataset with Images 2025-2026 v1"
Samples:   ~4,860 ad images
Labels:    9 attributes per image (multi-task)
Split:     Train / Validation

┌──────────────────────────────────────────────────────┐
│  Sample 1        Sample 2        Sample 3            │
│  ┌────────┐      ┌────────┐      ┌────────┐          │
│  │  IMG   │      │  IMG   │      │  IMG   │          │
│  └────────┘      └────────┘      └────────┘          │
│  Theme: Sale     Theme: Festival Theme: Product      │
│  Color: Red      Color: Gold    Color: Blue          │
│  Audience: 18-24 Audience: 25-34 Audience: 35-44     │
└──────────────────────────────────────────────────────┘

DATA AUGMENTATION (training only):
   • Horizontal flip
   • Rotation (±15°)
   • Color jitter (brightness, contrast, saturation)
   • Random crop + resize
```

## Visual Suggestion
- Show **3 real sample images** from the dataset with their labels below.
- Use a **grid layout** for the augmentation techniques with small icons.

## Key Talking Points
- **4,860 samples is small** for deep learning — this is one of the challenges (mentioned later).
- Augmentation is **critical** to prevent overfitting on a small dataset.
- Each sample has **9 labels** — that's why we need multi-task learning.
