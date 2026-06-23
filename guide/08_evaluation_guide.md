# 08 — Evaluation Guide

Model evaluation, metrics computation, and result persistence.

## Quick Start

```bash
cd multi-model/
python -m scripts.evaluate \
  --config configs/model/model_config.json \
  --checkpoint saved_models/best_model_epoch_10.pt \
  --output results/ \
  --batch-size 32
```

## Evaluation Functions

### [`evaluate_model()`](../multi-model/use_cases/training/evaluate.py:21)

Runs inference on the entire dataset and collects predictions:

```python
results = evaluate_model(model, dataloader, device)
# results = {
#     "all_preds": [3, 7, 0, 1, ...],   # predicted class indices
#     "all_labels": [3, 5, 0, 2, ...],   # ground truth class indices
# }
```

**Flow per batch:**

1. Set `model.eval()`
2. Enter `torch.no_grad()` context
3. Move images to device
4. Convert labels to tensor (handles pandas Series, dicts, lists)
5. Prepare text inputs (from dataset or zero tensors if not available)
6. `model(images, input_ids, attention_mask)` → Dict of per-attribute logits
7. Use **first attribute head's** logits for top-level evaluation
8. `logits.max(1)` → predicted class indices
9. Collect predictions and labels into flat lists

**Key design choice:** Only the first attribute head is used for top-level accuracy. For per-attribute evaluation, extend this function to iterate all heads.

---

### [`compute_metrics()`](../multi-model/use_cases/training/evaluate.py:101)

Computes accuracy from prediction/label lists:

```python
metrics = compute_metrics(all_preds=[3, 7, 0], all_labels=[3, 5, 0])
# metrics = {"accuracy": 0.6667, "total": 3}
```

| Metric | Formula | Description |
|--------|---------|-------------|
| `accuracy` | `correct / total` | Fraction of correct predictions |
| `total` | `len(all_labels)` | Number of samples evaluated |

Returns `{"accuracy": 0.0, "total": 0}` if no labels are provided.

---

### [`save_results()`](../multi-model/use_cases/training/evaluate.py:128)

Persists evaluation results as JSON:

```python
save_results(metrics, "results/")
# Creates: results/results.json
```

**Output file** (`results/results.json`):

```json
{
  "accuracy": 0.8234,
  "total": 1000
}
```

- Creates the output directory if it doesn't exist
- Uses `indent=2` for human-readable formatting

---

## Evaluation Script — [`scripts/evaluate.py`](../multi-model/scripts/evaluate.py)

### [`main()`](../multi-model/scripts/evaluate.py:28)

Full evaluation orchestration:

```bash
python -m scripts.evaluate \
  --config configs/model/model_config.json \
  --checkpoint saved_models/best_model_epoch_10.pt \
  --output results/ \
  --batch-size 32
```

**CLI arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to JSON config |
| `--checkpoint` | Yes | — | Path to model checkpoint (.pt) |
| `--output` | No | `results` | Output directory for results.json |
| `--batch-size` | No | 32 | Batch size for evaluation |

**Internal flow:**

```
1. load_config(args.config)
2. load_model(config["model"], device, args.checkpoint)
3. load_dataset("test", augment=False)
4. DataLoader(dataset, batch_size, shuffle=False)
5. evaluate_model(model, dataloader, device)
6. compute_metrics(all_preds, all_labels)
7. save_results(metrics, args.output)
```

**Output:**

```
Evaluation complete. Accuracy: 0.8234
Results saved to results/
```

---

## Per-Attribute Evaluation (Advanced)

The default evaluation uses only the first attribute head. To evaluate all 9 attributes individually, modify `evaluate_model()`:

```python
# Instead of using only the first head:
if isinstance(outputs, dict):
    first_key = next(iter(outputs))
    logits = outputs[first_key]

# Evaluate all heads:
if isinstance(outputs, dict):
    all_preds_per_attr = {}
    for attr_name, logits in outputs.items():
        _, preds = logits.max(1)
        all_preds_per_attr[attr_name] = preds.cpu().tolist()
```

This requires corresponding per-attribute ground truth labels in the dataset.

---

## Test Dataset Requirements

The evaluation script expects a test dataset at:

```
dataset/test/
├── test.csv
└── images/
    ├── sample_001.jpg
    ├── sample_002.jpg
    └── ...
```

The CSV must have an `image_filename` column and at least one label column matching the model's first attribute head classes.
