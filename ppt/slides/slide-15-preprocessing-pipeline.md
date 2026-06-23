# Slide 15 — Preprocessing Pipeline

## What to Say (Speaker Notes)
"Before any data goes into the model, it must be **preprocessed**. For **images**, we resize them to 224×224, normalize the pixel values using ImageNet statistics (mean and std), and apply data augmentation during training only. For **text**, we clean up OCR artifacts (like extra spaces, weird characters), tokenize using DistilBERT's tokenizer, and create an **attention mask** so the model knows which tokens are real and which are padding. I built a custom PyTorch **Dataset** class that returns a tuple of `(image, input_ids, attention_mask, labels)` for each sample. The **DataLoader** then batches these tuples with batch size 64, using 8 worker processes for speed, and drops the last incomplete batch. This preprocessing is critical — garbage in, garbage out."

## What to Show on Screen

```
🧹 PREPROCESSING PIPELINE

   ┌─────────────────────┐         ┌─────────────────────┐
   │      IMAGE          │         │       TEXT          │
   ├─────────────────────┤         ├─────────────────────┤
   │ • Resize to 224×224 │         │ • Clean OCR noise   │
   │ • Normalize         │         │ • Tokenize          │
   │   (ImageNet stats)  │         │   (DistilBERT)      │
   │ • Augment (train)   │         │ • Build attention   │
   │   - flip, rotate    │         │   mask              │
   │   - color jitter    │         │ • Max length: 256   │
   │   - random crop     │         │                     │
   └──────────┬──────────┘         └──────────┬──────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Custom PyTorch Dataset       │
              │  Returns:                     │
              │   (image, input_ids,          │
              │    attention_mask, labels)    │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  DataLoader                   │
              │  • batch_size = 64            │
              │  • num_workers = 8            │
              │  • drop_last = True           │
              └───────────────────────────────┘
```

## Visual Suggestion
- Show **two parallel pipelines** (image and text) merging into the Dataset.
- Use **arrows** to show the flow.
- Add small icons for each preprocessing step (resize, normalize, tokenize, etc.).

## Key Talking Points
- **Preprocessing is where most bugs happen** — getting it right saves hours of debugging later.
- The custom Dataset class is **reusable** — same code for training and inference.
- DataLoader workers **parallelize** data loading so the GPU doesn't sit idle.
