# 09 — CLI Scripts

Command-line tools for training, evaluation, serving, and model analysis.

All scripts are located in [`scripts/`](../multi-model/scripts/) and run as Python modules from the `multi-model/` directory:

```bash
cd multi-model/
python -m scripts.<script_name> [args]
```

---

## Training — [`scripts/train_model.py`](../multi-model/scripts/train_model.py)

Trains the FG_MFN model with configurable epochs, learning rate, and warmup.

### Usage

```bash
python -m scripts.train_model \
  --config configs/model/model_config.json \
  --epochs 50 \
  --warmup-epochs 3 \
  --log-dir local/logs
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to JSON training configuration |
| `--epochs` | No | Config value or 100 | Number of training epochs |
| `--warmup-epochs` | No | 3 | Warmup epochs for linear LR schedule |
| `--log-dir` | No | `local/logs` | Directory for training log CSV |

### Output

- **Console:** Epoch-by-epoch loss and accuracy
- **`local/logs/training_log.csv`:** Per-epoch metrics (train_loss, train_accuracy, val_loss, val_accuracy)
- **`saved_models/best_model_epoch_N.pt`:** Best model checkpoint (saved when validation accuracy improves)

### Example

```
Epoch 1/50 | Train Loss: 2.3421 | Train Acc: 0.1234 | Val Loss: 2.1500 | Val Acc: 0.1890
Epoch 2/50 | Train Loss: 1.8900 | Train Acc: 0.2500 | Val Loss: 1.7800 | Val Acc: 0.3010
 -> Saved best model to saved_models/best_model_epoch_2.pt
...
Training complete. Best validation accuracy: 0.8234
```

See [07 — Training Guide](07_training_guide.md) for full details.

---

## Evaluation — [`scripts/evaluate.py`](../multi-model/scripts/evaluate.py)

Evaluates a trained model checkpoint on the test dataset.

### Usage

```bash
python -m scripts.evaluate \
  --config configs/model/model_config.json \
  --checkpoint saved_models/best_model_epoch_10.pt \
  --output results/ \
  --batch-size 32
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to JSON model configuration |
| `--checkpoint` | Yes | — | Path to trained model checkpoint (.pt) |
| `--output` | No | `results` | Output directory for results.json |
| `--batch-size` | No | 32 | Batch size for evaluation |

### Output

- **Console:** `Evaluation complete. Accuracy: 0.8234`
- **`results/results.json`:** `{"accuracy": 0.8234, "total": 1000}`

See [08 — Evaluation Guide](08_evaluation_guide.md) for full details.

---

## Prediction Server — [`scripts/predict_server.py`](../multi-model/scripts/predict_server.py)

Starts the FastAPI prediction server.

### Usage

```bash
python -m scripts.predict_server \
  --config configs/model/model_config.json \
  --host 0.0.0.0 \
  --port 8000
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | No | `configs/model/model_config.json` | Path to configuration file |
| `--host` | No | `0.0.0.0` | Host to bind the server |
| `--port` | No | `8000` | Port to run the server on |

### Startup Sequence

1. [`setup_directories()`](../multi-model/lib/utils/lifecycle.py:7) — create required directories
2. [`load_config()`](../multi-model/lib/utils/config.py:9) — read configuration
3. `uvicorn.run(app, host, port)` — start the FastAPI app

The FastAPI [`lifespan()`](../multi-model/app/app.py:36) handler then loads the model, OCR engine, and configures the predictor before accepting requests.

See [04 — API Reference](04_api_reference.md) for endpoint documentation.

---

## Model Analysis — [`scripts/analyze_model.py`](../multi-model/scripts/analyze_model.py)

Analyzes model architecture, counts parameters, and benchmarks inference latency.

### Usage

```bash
python -m scripts.analyze_model \
  --config configs/model/model_config.json \
  --num_samples 100
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to model configuration JSON |
| `--num_samples` | No | 100 | Number of forward passes for latency benchmarking |

### Output

```
  total_parameters: 67234560
  trainable_parameters: 24185600
  frozen_parameters: 43048960
  num_attribute_heads: 9
  attribute_names: ['theme', 'sentiment', 'emotion', 'dominant_colour', 'attention_score', 'trust_safety', 'target_audience', 'predicted_ctr', 'likelihood_shares']
  mean_latency_ms: 45.23
  throughput_samples_per_sec: 22.11
```

### Functions

#### [`count_parameters()`](../multi-model/scripts/analyze_model.py:35)

Counts trainable parameters only:

```python
trainable = count_parameters(model)  # int
```

#### [`count_total_parameters()`](../multi-model/scripts/analyze_model.py:48)

Counts all parameters (including frozen):

```python
total = count_total_parameters(model)  # int
```

#### [`benchmark_inference()`](../multi-model/scripts/analyze_model.py:61)

Measures inference latency with random dummy inputs:

```python
results = benchmark_inference(model, num_samples=100, batch_size=1)
# {"mean_latency_ms": 45.23, "throughput_samples_per_sec": 22.11}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 100 | Number of forward passes to time |
| `batch_size` | 1 | Batch size per forward pass |
| `image_height` | 224 | Dummy image height |
| `image_width` | 224 | Dummy image width |
| `seq_length` | 128 | Dummy text sequence length |

Uses `torch.no_grad()` and `time.perf_counter()` for accurate measurement.

#### [`analyze_model()`](../multi-model/scripts/analyze_model.py:114)

Combines parameter counting and benchmarking:

```python
result = analyze_model(cfg, num_samples=100)
```

---

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | [`app/app.py`](../multi-model/app/app.py) | Override default config path (default: `configs/model/model_config.json`) |
| `ENV` | [`app/app.py`](../multi-model/app/app.py) | Set to `"dev"` to enable uvicorn auto-reload |
