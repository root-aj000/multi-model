# Slide 28 — Key Learnings

## What to Say (Speaker Notes)
"What did I learn from this project? **First**, **multi-modal fusion beats single modality** — combining image and text gives much better results than using either alone. **Second**, **cross-modal attention beats naive concatenation** — letting the model learn which image regions and text tokens are related is more powerful than just concatenating features. **Third**, **differential learning rates are critical** — the pretrained encoder needs a tiny learning rate (1.5e-5) while the new heads need a larger one (2e-4). Using the same rate for both either destroys the pretrained weights or fails to train the heads. **Fourth**, **engineering hygiene matters** — clean code, tests, and documentation save time in the long run. **Fifth**, **clean architecture pays off** — the 4-layer structure made it easy to swap components (e.g., trying PaddleOCR instead of EasyOCR) without touching the rest of the code."

## What to Show on Screen

```
💡 KEY LEARNINGS

   1️⃣  Multi-modal fusion > single modality
       (image + text together beats either alone)

   2️⃣  Cross-modal attention > naive concat/add
       (let the model learn the relationships)

   3️⃣  Differential learning rates are CRITICAL
       (encoder: 1.5e-5, heads: 2e-4)

   4️⃣  Engineering hygiene matters
       (clean code, tests, docs save time)

   5️⃣  Clean architecture pays off
       (easy to swap components, easy to test)

   🎯 TAKEAWAY:
      A good ML project is 20% modeling
      and 80% engineering.
```

## Visual Suggestion
- Use **numbered cards** (1-5) for each learning.
- Add **icons** for each point (e.g., brain for fusion, gears for engineering).
- Highlight the **takeaway** in a different color.

## Key Talking Points
- The **engineering** is often harder than the modeling.
- **Pretrained models** need careful tuning — don't treat them like random init.
- **Architecture decisions** made early have long-lasting impact.
