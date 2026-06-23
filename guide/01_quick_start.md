# 01 — Quick Start

Get the Multi-Model Prediction System up and running in 5 minutes.

## Prerequisites

- Python 3.9+
- pip or conda for package management
- (Optional) CUDA-capable GPU for faster inference/training
- (Optional) Node.js 18+ for the frontend

## Installation

### 1. Clone and enter the project

```bash
cd multi-model/
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:

| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | Model training and inference |
| `transformers` | HuggingFace text encoder (DistilBERT) |
| `fastapi` + `uvicorn` | REST API server |
| `easyocr` | OCR text extraction from images |
| `paddleocr` | Alternative OCR engine (optional) |
| `pandas` | Dataset CSV loading |
| `pillow` | Image I/O |
| `numpy` | Array operations |

### 3. Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Running the Prediction Server

### Option A: Using the startup script

```bash
python -m scripts.predict_server --config configs/model/model_config.json --port 8000
```

### Option B: Using the app module directly

```bash
CONFIG_PATH=configs/model/model_config.json python -m app.app
```

### Option C: With uvicorn directly

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

The server starts on `http://0.0.0.0:8000`. On startup, it:

1. Creates required directories (`uploads/`, `saved_models/`, `local/`)
2. Loads the model configuration
3. Initializes the FG_MFN model
4. Starts the EasyOCR engine
5. Wires everything into the prediction endpoint

### Verify the server is running

```bash
curl http://localhost:8000/health
# {"status": "healthy", "version": "1.0.0"}
```

## Making Predictions

### Single image prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "files=@path/to/your/image.jpg"
```

### Multiple images

```bash
curl -X POST http://localhost:8000/predict \
  -F "files=@image1.jpg" \
  -F "files=@image2.png"
```

### Response format

```json
{
  "predictions": [
    {
      "theme": "Food",
      "sentiment": "Positive",
      "emotion": "Joy",
      "dominant_colour": "Red",
      "attention_score": "High",
      "trust_safety": "Safe",
      "target_audience": "Food Lovers",
      "predicted_ctr": "High",
      "likelihood_shares": "Medium",
      "predicted_label": "Food",
      "ocr_text": "Fresh Pizza Deal",
      "keywords": "deal, free",
      "monetary_mention": "$9.99",
      "call_to_action": "buy now",
      "object_detected": "",
      "filename": "image.jpg"
    }
  ],
  "total_images": 1,
  "processing_time_ms": 342
}
```

## Running the Frontend

```bash
cd frontend/
pnpm install
pnpm dev
```

The Next.js frontend connects to `http://localhost:8000/predict` and provides an image upload UI with results display.

## Training a Model

```bash
python -m scripts.train_model \
  --config configs/model/model_config.json \
  --epochs 50 \
  --warmup-epochs 3 \
  --log-dir local/logs
```

Training outputs:
- Console: epoch-by-epoch loss and accuracy
- `local/logs/training_log.csv`: per-epoch metrics
- `saved_models/best_model_epoch_N.pt`: best checkpoint (by validation accuracy)

See [07 — Training Guide](07_training_guide.md) for full details.

## Evaluating a Model

```bash
python -m scripts.evaluate \
  --config configs/model/model_config.json \
  --checkpoint saved_models/best_model_epoch_10.pt \
  --output results/
```

Outputs `results/results.json` with accuracy metrics.

See [08 — Evaluation Guide](08_evaluation_guide.md) for full details.

## Running Tests

```bash
cd multi-model/
python -m pytest tests/ -v
```

See [12 — Testing](12_testing.md) for the full test suite documentation.

## Next Steps

- [02 — Architecture](02_architecture.md): Understand how the modules fit together
- [03 — Model Reference](03_model_reference.md): Deep dive into FG_MFN internals
- [10 — Configuration](10_configuration.md): All configuration options explained
