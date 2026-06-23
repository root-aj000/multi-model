# Slide 27 — Challenges Faced

## What to Say (Speaker Notes)
"Every project has challenges, and this one was no exception. The **first challenge** was the **small dataset** — only ~4,860 samples. This led to **overfitting**, where the model memorized the training data but failed on new data. I solved this by **freezing the ResNet backbone**, adding **dropout (0.5)**, and using **label smoothing (0.2)**. The **second challenge** was **multi-task label conflict** — some attributes contradicted each other in the training data. I disabled **Mixup augmentation** because it made the conflict worse. The **third challenge** was that **engagement metrics** (CTR, shares) are inherently hard to predict from just the creative. I **reduced their loss weights** to focus the model on the more learnable attributes. The **fourth challenge** was **OCR noise** — extracted text often had errors. I added a **text cleaner** and used **attention-weighted pooling** to down-weight noisy tokens. In total, I documented and fixed **28 bugs** during development."

## What to Show on Screen

```
⚠️ CHALLENGES & SOLUTIONS

   1️⃣  SMALL DATASET → OVERFITTING
       Problem : Only ~4,860 samples
       Solution: Freeze ResNet + Dropout 0.5 + Label smoothing 0.2

   2️⃣  MULTI-TASK LABEL CONFLICT
       Problem : Some labels contradicted each other
       Solution: Disabled Mixup augmentation

   3️⃣  ENGAGEMENT METRICS HARD TO LEARN
       Problem : CTR & shares depend on factors beyond the creative
       Solution: Reduced their loss weights

   4️⃣  OCR NOISE
       Problem : Extracted text had errors
       Solution: Text cleaner + attention-weighted pooling

   📊 TOTAL: 28 documented bugs fixed
```

## Visual Suggestion
- Use a **table** with 3 columns: Challenge, Problem, Solution.
- Add **warning icons** for each challenge.
- Use **green checkmarks** for solutions.

## Key Talking Points
- **Overfitting** is the #1 problem in deep learning with small data.
- **Multi-task learning** has unique challenges — label conflicts are real.
- **Engagement metrics** are inherently noisy — even humans can't predict them perfectly.
- **OCR** is never perfect — always plan for noise.
