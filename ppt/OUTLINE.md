# 📊 PPT Outline — Multi-Model Prediction System (FG_MFN)

> **Project:** Fine-Grained Multi-Modal Fusion Network for Ad-Creative Classification
> **Audience:** College faculty + peers
> **Suggested duration:** 18–22 minutes + 5 min Q&A
> **Total slides:** ~28 (target 25–30)

---

## 🎯 Presentation Goals

1. Show a **real, end-to-end deep-learning system** (not just a notebook demo).
2. Explain the **multi-modal fusion idea** (image + text → 9 attributes).
3. Demonstrate **clean architecture, training pipeline, evaluation, and a working web app**.
4. Highlight **engineering rigor** (tests, metrics, visualizations, deployment).

---

## 🗂️ Slide-by-Slide Outline

### **SLIDE 1 — Title Slide**
- Project title: **"Multi-Model Prediction System for Ad-Creative Classification using FG_MFN"**
- Subtitle: *A Fine-Grained Multi-Modal Fusion Network combining Vision + Text*
- Your name, roll no., branch, college name, guide name
- Date

---

### **SLIDE 2 — Table of Contents**
1. Problem Statement
2. Motivation & Use Cases
3. Objectives
4. Literature / Existing Approaches
5. Proposed System — FG_MFN
6. Dataset
7. Architecture (high-level)
8. Model Details (visual + text + fusion)
9. 9 Output Attributes
10. Training Pipeline
11. Loss, Optimizer, Regularisation
12. Evaluation Metrics
13. Results & Visualizations
14. Web Application (Frontend)
15. REST API (Backend)
16. OCR Engine
17. Project Structure
18. Testing & Quality
19. Deployment
20. Challenges Faced
21. Key Learnings
22. Future Enhancements
23. Conclusion
24. References
25. Thank You + Q&A

---

### **SLIDE 3 — Problem Statement**
- Ad agencies & marketers receive **thousands of creatives** daily.
- Manually tagging each ad with **theme, sentiment, emotion, colour, audience, CTR, etc.** is slow, expensive, and inconsistent.
- Need an **automated, multi-attribute classifier** that looks at both the **image** and the **text inside the image** (OCR).
- Output must be **structured** (9 labels per ad) so it can plug into analytics dashboards.

---

### **SLIDE 4 — Motivation & Real-World Use Cases**
- 🎯 Programmatic ad buying — auto-tagging creatives before auction.
- 📊 Marketing analytics — sentiment + emotion trends across campaigns.
- 🛡️ Brand safety — flag *unsafe / questionable* ads automatically.
- 👥 Audience targeting — predict age bucket from creative style.
- 💰 CTR / share prediction — pre-launch creative scoring.
- 🧪 A/B testing — pick the strongest creative variant.

---

### **SLIDE 5 — Objectives**
- Build a **single model** that predicts **9 attributes simultaneously** (multi-task learning).
- Fuse **two modalities**: visual (CNN) + textual (Transformer via OCR).
- Achieve **production-grade accuracy** with proper validation, metrics, and visualizations.
- Expose the model through a **FastAPI REST endpoint**.
- Build a **Next.js dashboard** for non-technical users.
- Maintain **clean architecture, tests, and documentation**.

---

### **SLIDE 6 — Existing Approaches (Literature)**
- Single-task CNN classifiers (one model per attribute) — costly, ignores correlation.
- CLIP-style contrastive models — strong but require huge data.
- Late-fusion vs. early-fusion — trade-offs.
- Our approach: **mid-level cross-modal attention fusion** with multi-task heads.
- *Cite 2–3 papers (CLIP, VisualBERT, MMBT, ALBEF).*

---

### **SLIDE 7 — Proposed System Overview**
- One-liner: *"An image goes in → 9 structured attributes come out."*
- Pipeline diagram (use boxes + arrows):
  `Image → OCR → Text  ┐`
  `Image → CNN        ─┴→ Cross-Modal Attention → Shared FC → 9 Heads → Predictions`
- Highlight: **single forward pass, 9 outputs**.

---

### **SLIDE 8 — Dataset**
- Source: Kaggle *Ads Dataset with Images 2025-2026 v1*.
- ~4,860 training samples, validation split.
- Each row = (image, 9 labels).
- Augmentation: horizontal flip, rotation, color jitter, random crop.
- Show 2–3 sample images with their labels (visual).

---

### **SLIDE 9 — System Architecture (Layered)**
- 4-layer clean architecture diagram:
  - `scripts/` (CLI entry points)
  - `app/` (FastAPI HTTP layer)
  - `use_cases/` (orchestration)
  - `lib/` (core domain — models, OCR, preprocessing, services, utils)
- Dependency rule: **inward only**.
- Design patterns used: **Factory, Strategy, Dependency Injection**.

---

### **SLIDE 10 — FG_MFN Model (Visual Branch)**
- Backbone: **ResNet-50** (pretrained on ImageNet).
- Optional freeze (`FREEZE_BACKBONE=true`) — combats overfitting on small dataset.
- Output: 2048-d feature vector per image.
- Image input: 224×224×3, normalized.

---

### **SLIDE 11 — FG_MFN Model (Text Branch)**
- Encoder: **DistilBERT (base, uncased)** — HuggingFace.
- Pooling: **attention-weighted** (uses encoder's own attention scores).
- Tokenization: max length 256.
- Output: 768-d feature vector per text.

---

### **SLIDE 12 — Cross-Modal Fusion**
- Strategy: **Cross-Modal Multi-Head Attention** (8 heads, dim 768).
- Visual attends to text **AND** text attends to visual (bidirectional).
- Projection → concat → Linear → `HIDDEN_DIM = 512`.
- Why attention > concat/add? (parameter efficiency + learns inter-modal alignment).

---

### **SLIDE 13 — Shared Layer + 9 Attribute Heads**
- Shared FC: `Linear(512→512) → ReLU → Dropout(0.5)` (deep variant).
- 9 independent classification heads (one `Linear` each):
  | # | Attribute | Classes |
  |---|-----------|---------|
  | 1 | theme | 9 |
  | 2 | sentiment | 3 |
  | 3 | emotion | 5 |
  | 4 | dominant_colour | 10 |
  | 5 | attention_score | 3 |
  | 6 | trust_safety | 3 |
  | 7 | target_audience | 6 |
  | 8 | predicted_ctr | 3 |
  | 9 | likelihood_shares | 3 |

---

### **SLIDE 14 — OCR Engine**
- Two engines supported via **Strategy pattern**: **EasyOCR** (default), **PaddleOCR** (optional).
- Why OCR? — text in ads is part of the creative, not metadata.
- Factory: `create_ocr_engine("easyocr" | "paddleocr")`.
- Cached models in `local/ocr/`.

---

### **SLIDE 15 — Preprocessing Pipeline**
- **Image:** resize → normalize (ImageNet stats) → augment (train only).
- **Text:** clean OCR artifacts → tokenize → attention mask.
- **Dataset:** custom PyTorch `Dataset` returning `(image, input_ids, attention_mask, labels)`.
- DataLoader: `batch_size=64`, `num_workers=8`, `drop_last=True`.

---

### **SLIDE 16 — Training Pipeline**
- Optimizer: **AdamW** with **differential learning rates**:
  - Heads / shared FC: `2e-4`
  - Text encoder: `1.5e-5` (10–25× lower to avoid catastrophic forgetting).
- Scheduler: **Linear warmup (5 epochs) → Cosine annealing**.
- Regularization: **label smoothing 0.2**, **dropout 0.5**, **weight decay 0.01**.
- Early stopping: patience 10.
- Per-attribute **loss weights** (engagement heads down-weighted to 0.1).

---

### **SLIDE 17 — Loss Function**
- Sum of 9 `CrossEntropyLoss` (with label smoothing) weighted by `ATTRIBUTE_LOSS_WEIGHTS`.
- Class imbalance handled by **inverse-frequency class weights** (auto-computed).
- Why multi-task? — shared representation, regularization effect, single forward pass.

---

### **SLIDE 18 — Evaluation Metrics**
- **Overall:** accuracy_macro, accuracy_weighted, total samples.
- **Per-attribute (×9):** accuracy, precision (macro/weighted), recall (macro/weighted), F1 (macro/weighted), confusion matrix, per-class breakdown.
- **17+ distinct metric types** logged to `results/results.json`.

---

### **SLIDE 19 — Training Visualizations**
- `training_curves.png` — 2×2 grid: train/val loss, train/val accuracy, best epoch marker.
- `per_attribute_training.png` — per-attribute accuracy curves.
- Helps diagnose overfitting per head.

---

### **SLIDE 20 — Evaluation Visualizations**
- `confusion_matrices.png` — heatmap per attribute.
- `per_attribute_metrics.png` — accuracy / precision-recall / F1 bar charts.
- `macro_weighted_comparison.png` — macro vs weighted comparison.

---

### **SLIDE 21 — Results Snapshot**
- Insert your actual numbers here (e.g., macro accuracy ~0.82, weighted ~0.81).
- Per-attribute accuracy table (theme 0.92, sentiment 0.87, …).
- Best epoch, total training time, GPU used (T4 × 2 via DataParallel).

---

### **SLIDE 22 — REST API (Backend)**
- **FastAPI** server, port 8000.
- Endpoints:
  - `GET /health` — health check.
  - `GET /model/info` — model metadata.
  - `POST /predict` — multipart upload, returns 9 attributes per image.
- Constraints: max 10 MB, allowed extensions (png/jpg/jpeg/gif/bmp/tiff/webp).
- Lifespan pattern for startup/shutdown.

---

### **SLIDE 23 — Web Application (Frontend)**
- **Next.js 16 + React 19 + TypeScript + Tailwind v4**.
- State: **Zustand**.
- Charts: **Recharts**.
- Backend client: **Supabase JS + Axios**.
- Features: upload ad → see 9 predicted attributes visualized as cards/bars.
- Theme: dark/light via `next-themes`.

---

### **SLIDE 24 — Project Structure**
- Tree view of `multi-model/`:
  ```
  app/  configs/  lib/  scripts/  tests/  use_cases/
  ```
- Plus `frontend/` (Next.js) and `docs/` (13 markdown guides).
- Show how clean separation enables independent testing.

---

### **SLIDE 25 — Testing & Quality**
- `pytest` suite under `tests/unit/`.
- Coverage of: config loading, dataset, model factory, OCR factory, predictor, postprocessor.
- `run_tests.sh` for one-shot execution.
- Linting: ESLint (frontend), PEP8-friendly Python.

---

### **SLIDE 26 — Deployment**
- Training: Kaggle (2× T4 GPUs, DataParallel).
- Inference: local FastAPI server / cloud VM.
- Frontend: Vercel-ready Next.js.
- Models + logs versioned in `saved_models/` and `local/logs/`.

---

### **SLIDE 27 — Challenges Faced**
- Small dataset (~4,860 samples) → overfitting → solved via backbone freezing, dropout, label smoothing.
- Multi-task label conflict → Mixup disabled.
- Engagement metrics (CTR/shares) not learnable from image+text → loss weights reduced.
- OCR noise → text cleaner + attention-weighted pooling.
- 28 documented bugs fixed during development (see `docs/`).

---

### **SLIDE 28 — Key Learnings**
- Multi-modal fusion > single modality for ad understanding.
- Cross-modal attention beats naive concat/add.
- Differential LR is critical when fine-tuning transformers.
- Engineering hygiene (tests, metrics, viz) matters as much as the model.
- Clean architecture pays off when the project grows.

---

### **SLIDE 29 — Future Enhancements**
- Replace DistilBERT with **multilingual** encoder for regional ads.
- Add **CLIP** as visual backbone for zero-shot transfer.
- **Active learning** loop — flag low-confidence ads for human review.
- **Drift monitoring** in production.
- Mobile-friendly inference (ONNX / TensorRT).

---

### **SLIDE 30 — Conclusion**
- Successfully built an **end-to-end multi-modal classification system** predicting 9 attributes per ad.
- Demonstrated **clean architecture, rigorous training, comprehensive evaluation, and a working web app**.
- Project is **production-ready** and well-documented (13 guides).
- One-line takeaway: *"One image in, nine structured insights out."*

---

### **SLIDE 31 — References**
- Devlin et al., **BERT** (2018).
- Sanh et al., **DistilBERT** (2019).
- He et al., **ResNet** (2016).
- Radford et al., **CLIP** (2021).
- EasyOCR / PaddleOCR docs.
- PyTorch, HuggingFace Transformers, FastAPI, Next.js docs.

---

### **SLIDE 32 — Thank You + Q&A**
- "Thank you" + your contact info.
- Invite questions.
- Optional: live demo of `/predict` endpoint or the web UI.

---

## 🎨 Design Tips for the Slides

- **Theme:** dark navy + accent teal/orange (matches your Next.js UI).
- **Font:** Inter / Poppins (clean, modern).
- **Diagrams:** use **Excalidraw** or **draw.io** for the architecture slide.
- **Code blocks:** keep ≤ 8 lines per slide; use syntax highlighting.
- **Numbers:** always show **baseline vs. improved** (e.g., dropout 0.4 → 0.5).
- **One idea per slide** — don't overload.

---

## 🛠️ Suggested Tools to Build the PPT

| Tool | Why |
|------|-----|
| **Canva** | Fastest, free templates, good for college presentations |
| **Google Slides** | Easy collaboration, free |
| **PowerPoint** | Best for animations + offline |
| **Figma** | If you want pixel-perfect custom slides |
| **revealjs** | If you want an HTML/web-based deck |

---

## 📁 Recommended File Layout Inside `ppt/`

```
ppt/
├── OUTLINE.md                 ← this file
├── slides/                    ← individual slide source (if using reveal.js / markdown)
│   ├── slide-01-title.md
│   ├── slide-02-toc.md
│   └── ...
├── images/                    ← diagrams, screenshots, charts to embed
│   ├── architecture.png
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   └── ui-screenshot.png
└── README.md                  ← how to assemble the deck
```

---

## ✅ Next Steps

1. Confirm the outline above (or tell me which slides to add/remove).
2. I'll generate **per-slide markdown content** in `ppt/slides/` so you can paste into any tool.
3. I'll also generate **Mermaid diagrams** (architecture, dataflow) you can export as PNG.
4. Optionally, I can produce an **HTML reveal.js deck** directly in `ppt/`.

**Tell me which format you want:**
- (a) Markdown slides (paste into Google Slides / PowerPoint)
- (b) reveal.js HTML deck (open in browser, present directly)
- (c) Both
