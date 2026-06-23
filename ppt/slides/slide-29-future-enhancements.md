# Slide 29 — Future Enhancements

## What to Say (Speaker Notes)
"This project is a **solid foundation**, but there's a lot of room to grow. **First**, I could add a **multilingual encoder** like XLM-RoBERTa to support ads in languages other than English. **Second**, I could replace ResNet + DistilBERT with **CLIP** — a model pretrained on image-text pairs that already understands the relationship between vision and language. **Third**, I could add an **active learning loop** — let the model flag uncertain predictions and ask a human to label them, then retrain. **Fourth**, I could add **drift monitoring** — detect when the model's accuracy degrades over time as ad trends change, and trigger automatic retraining. **Fifth**, I could optimize the model for **mobile inference** using **ONNX** or **TensorRT** so it can run on phones without a server. Each of these would be a meaningful improvement on its own."

## What to Show on Screen

```
🔮 FUTURE ENHANCEMENTS

   1️⃣  MULTILINGUAL ENCODER
       Replace DistilBERT with XLM-RoBERTa
       → Support ads in any language

   2️⃣  CLIP BACKBONE
       Replace ResNet + DistilBERT with CLIP
       → Already understands image-text relationships

   3️⃣  ACTIVE LEARNING LOOP
       Model flags uncertain predictions
       → Human labels them → Retrain

   4️⃣  DRIFT MONITORING
       Detect accuracy degradation over time
       → Trigger automatic retraining

   5️⃣  MOBILE-FRIENDLY INFERENCE
       Convert to ONNX or TensorRT
       → Run on phones without a server

   🎯 VISION:
      From a research project to a production system
      that improves continuously.
```

## Visual Suggestion
- Use a **roadmap** style with arrows showing progression.
- Add **timeline** (e.g., "Q1: multilingual, Q2: CLIP").
- Use **upward arrows** to show growth.

## Key Talking Points
- **CLIP** is the natural next step — it's the state of the art for vision-language.
- **Active learning** is how production ML systems stay accurate.
- **Mobile inference** opens up new use cases (e.g., real-time ad analysis).
