# Slide 22 — REST API (Backend)

## What to Say (Speaker Notes)
"The trained model is exposed through a **REST API** built with **FastAPI**, running on port 8000. There are **3 endpoints**. `GET /health` is a simple health check that returns `{'status': 'ok'}` if the server is running. `GET /model/info` returns metadata about the model — its name, version, the 9 attributes it predicts, and the number of classes per attribute. `POST /predict` is the main endpoint — the client uploads an image as multipart form data, and the server returns the 9 predicted attributes as JSON. There are constraints: max file size 10 MB, and allowed extensions are png, jpg, jpeg, gif, bmp, tiff, and webp. I use FastAPI's **lifespan pattern** for clean startup and shutdown — the model is loaded once at startup and reused for all requests. This makes inference fast."

## What to Show on Screen

```
🌐 REST API (FastAPI) — Port 8000

   ┌──────────────────────────────────────────────────┐
   │  GET  /health                                    │
   │  →  {"status": "ok"}                             │
   │                                                  │
   │  GET  /model/info                                │
   │  →  {                                            │
   │       "name": "FG_MFN",                          │
   │       "version": "1.0",                          │
   │       "attributes": ["theme", "sentiment", ...], │
   │       "classes_per_attribute": [9, 3, 5, ...]    │
   │     }                                            │
   │                                                  │
   │  POST /predict                                   │
   │  →  multipart upload (image file)                │
   │  ←  {                                            │
   │       "theme": "Sale",                           │
   │       "sentiment": "Positive",                   │
   │       "emotion": "Excited",                      │
   │       ... (9 attributes total)                   │
   │     }                                            │
   └──────────────────────────────────────────────────┘

   ⚙️  CONSTRAINTS:
      • Max file size: 10 MB
      • Allowed: png, jpg, jpeg, gif, bmp, tiff, webp

   🔄 LIFESPAN:
      • Model loaded ONCE at startup
      • Reused for all requests (fast inference)
```

## Visual Suggestion
- Show **3 code blocks** (or styled boxes) for the 3 endpoints.
- Add a **small screenshot** of the API in action (using Swagger UI or curl).
- Use **green** for GET endpoints and **blue** for POST.

## Key Talking Points
- **FastAPI** is modern, fast, and auto-generates documentation (Swagger UI at `/docs`).
- The **lifespan pattern** is the recommended way to manage resources in FastAPI.
- The API is **stateless** — each request is independent, which makes it scalable.
