# Slide 26 — Deployment

## What to Say (Speaker Notes)
"Here's how the system is **deployed**. Training happens on **Kaggle** using **2× T4 GPUs** with PyTorch **DataParallel** for multi-GPU acceleration. The trained model is saved to `saved_models/` and the training logs go to `local/logs/`. For inference, the model can run on a **local machine** or a **cloud VM** — anywhere with a GPU. The FastAPI server is lightweight and can be deployed with **Uvicorn** or **Gunicorn**. The Next.js frontend is **Vercel-ready** — it can be deployed with one click. The whole system is **containerizable** with Docker, though I haven't included a Dockerfile in this version. The key insight is that training and inference have **different requirements** — training needs GPUs, inference just needs a CPU or single GPU."

## What to Show on Screen

```
🚀 DEPLOYMENT

   TRAINING (Heavy):
   ┌────────────────────┬──────────────────────────┐
   │ Platform           │ Kaggle Notebooks         │
   │ Hardware           │ 2× T4 GPUs               │
   │ Parallelism        │ PyTorch DataParallel     │
   │ Duration           │ ~4 hours                 │
   │ Output             │ saved_models/, logs/     │
   └────────────────────┴──────────────────────────┘

   INFERENCE (Light):
   ┌────────────────────┬──────────────────────────┐
   │ Backend            │ FastAPI + Uvicorn        │
   │ Frontend           │ Next.js (Vercel-ready)   │
   │ Hardware           │ CPU or single GPU        │
   │ Container          │ Docker-ready             │
   └────────────────────┴──────────────────────────┘

   📦 ARTIFACTS:
      • saved_models/fg_mfn_best.pt
      • local/logs/training_*.log
      • local/logs/*.png (visualizations)
```

## Visual Suggestion
- Show a **flow diagram**: Kaggle → saved_models → API → Web.
- Add **icons** for Kaggle, Docker, Vercel.
- Use **different colors** for training vs inference.

## Key Talking Points
- **Training and inference are decoupled** — different hardware needs.
- **DataParallel** lets us use multiple GPUs without code changes.
- **Vercel** makes frontend deployment trivial.
