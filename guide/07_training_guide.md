# 07 — Training Guide

Training pipeline, Mixup augmentation, warmup scheduling, epoch loops, and checkpointing.

## Quick Start

```bash
cd multi-model/
python -m scripts.train_model \
  --config configs/model/model_config.json \
  --epochs 50 \
  --warmup-epochs 3 \
  --log-dir local/logs
```

## Training Pipeline

### [`build_training_pipeline()`](../multi-model/use_cases/training/pipeline.py:133)

Assembles all training dependencies in one call:

```python
pipeline = build_training_pipeline("configs/model/model_config.json")
# pipeline = {
#     "model": FG_MFN,
#     "train_loader": DataLoader,
#     "val_loader": DataLoader,
#     "optimizer": Adam,
#     "criterion": CrossEntropyLoss,
#     "scheduler": StepLR,
# }
```

**Internal steps:**

| Step | Function | Output |
|------|----------|--------|
| 1 | [`load_config()`](../multi-model/lib/utils/config.py:9) | Config dict |
| 2 | [`load_datasets()`](../multi-model/use_cases/training/pipeline.py:39) | (train_dataset, val_dataset) |
| 3 | [`create_data_loaders()`](../multi-model/use_cases/training/pipeline.py:64) | (train_loader, val_loader) |
| 4 | [`create_model()`](../multi-model/lib/models/factory.py:60) | FG_MFN on device |
| 5 | [`setup_training_components()`](../multi-model/use_cases/training/pipeline.py:97) | {optimizer, criterion, scheduler} |

---

### [`load_datasets()`](../multi-model/use_cases/training/pipeline.py:39)

Loads train and validation datasets:

```python
train_ds, val_ds = load_datasets(config)
# train_ds: augment=True (random horizontal flip)
# val_ds:   augment=False
```

Expects the directory structure `dataset/train/train.csv` and `dataset/val/val.csv`.

### [`create_data_loaders()`](../multi-model/use_cases/training/pipeline.py:64)

Creates PyTorch DataLoaders:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Samples per batch |
| `num_workers` | 4 | Data loading workers |

Training loader uses `shuffle=True`; validation loader uses `shuffle=False`.

### [`setup_training_components()`](../multi-model/use_cases/training/pipeline.py:97)

| Component | Default | Config Override |
|-----------|---------|-----------------|
| Optimizer | `Adam(lr=0.001)` | `config["learning_rate"]` |
| Scheduler | `StepLR(step_size=10, gamma=0.1)` | `config["scheduler_step_size"]`, `config["scheduler_gamma"]` |
| Criterion | `CrossEntropyLoss()` | None (fixed) |

---

## Training Loop

### [`train_epoch()`](../multi-model/use_cases/training/train_model.py:103)

Runs one training epoch:

```python
avg_loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
```

**Per-batch flow:**

1. Move images to device
2. Convert labels to tensor (handles pandas Series, dicts, lists)
3. Prepare text inputs:
   - If dataset provides 4+ items: use `batch[2]` (input_ids) and `batch[3]` (attention_mask)
   - Otherwise: create **zero tensors** of shape `(B, 128)` as placeholder text
4. `optimizer.zero_grad()`
5. `model(images, input_ids, attention_mask)` → Dict of per-attribute logits
6. Compute loss: **sum of CrossEntropyLoss across all attribute heads**
7. `loss.backward()` → `optimizer.step()`
8. Track running loss and accuracy (using first attribute head's predictions)

**Returns:** `(average_loss: float, accuracy: float)`

### [`validate_epoch()`](../multi-model/use_cases/training/train_model.py:192)

Same flow as `train_epoch()` but:
- `model.eval()` instead of `model.train()`
- `torch.no_grad()` context (no gradient computation)
- No optimizer step

---

## Advanced Training Features

### Mixup Data Augmentation — [`mixup_data()`](../multi-model/use_cases/training/train_model.py:20)

Creates convex combinations of image pairs for regularization:

```python
mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `images` | (required) | Batch tensor `(B, C, H, W)` |
| `labels` | (required) | Label tensor `(B,)` |
| `alpha` | 1.0 | Beta distribution parameter. 0 = no mixing. |

**How it works:**
1. Sample `λ ~ Beta(alpha, alpha)`
2. Generate random permutation `index`
3. `mixed = λ * images + (1 - λ) * images[index]`
4. Return `(mixed, original_labels, shuffled_labels, λ)`

### Mixup Loss — [`mixup_criterion()`](../multi-model/use_cases/training/train_model.py:51)

Computes the interpolated loss for mixed samples:

```python
loss = mixup_criterion(criterion, preds, labels_a, labels_b, lam)
# = lam * criterion(preds, labels_a) + (1 - lam) * criterion(preds, labels_b)
```

> **Note:** Mixup is defined in the codebase but not currently wired into the default training loop. To enable it, modify `train_epoch()` to call `mixup_data()` before the forward pass and `mixup_criterion()` instead of direct loss computation.

### Warmup Learning Rate — [`apply_warmup()`](../multi-model/use_cases/training/train_model.py:76)

Linearly increases learning rate from 0 to `base_lr` over the first `warmup_epochs`:

```python
apply_warmup(optimizer, epoch=0, warmup_epochs=3, base_lr=0.001)
# epoch 0: lr = 0.001 * 1/3 = 0.000333
# epoch 1: lr = 0.001 * 2/3 = 0.000667
# epoch 2: lr = 0.001 * 3/3 = 0.001
```

**Important:** Warmup and `StepLR` scheduler are **mutually exclusive** in the default training script. Warmup is only applied when `scheduler is None`. After warmup completes, the base learning rate takes effect.

---

## Training Script — [`scripts/train_model.py`](../multi-model/scripts/train_model.py)

### [`main()`](../multi-model/scripts/train_model.py:32)

Full training orchestration:

```bash
python -m scripts.train_model \
  --config configs/model/model_config.json \
  --epochs 50 \
  --warmup-epochs 3 \
  --log-dir local/logs
```

**CLI arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to JSON config |
| `--epochs` | No | Config value or 100 | Number of training epochs |
| `--warmup-epochs` | No | 3 | Warmup epochs for LR schedule |
| `--log-dir` | No | `local/logs` | Directory for training logs |

**Training loop:**

```
for epoch in range(num_epochs):
    if epoch < warmup_epochs and scheduler is None:
        apply_warmup(optimizer, epoch, warmup_epochs, base_lr)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

    if scheduler is not None:
        scheduler.step()

    training_logger.log_metrics(metrics, epoch + 1)

    if val_acc > best_val_acc:
        # Atomic checkpoint write
        torch.save(model.state_dict(), f"{checkpoint_path}.tmp")
        os.replace(f"{checkpoint_path}.tmp", checkpoint_path)
```

### Checkpoint Strategy

- **Best model only:** Only saves when validation accuracy improves
- **Atomic writes:** Writes to `.tmp` file first, then `os.replace()` to prevent corruption
- **Location:** `saved_models/best_model_epoch_{N}.pt`
- **Format:** PyTorch `state_dict` (recommended for portability)

---

## Training Log

### [`TrainingLogger`](../multi-model/lib/utils/logging.py:11)

Persists per-epoch metrics to a CSV file:

```python
logger = TrainingLogger("local/logs")
logger.log_metrics({"train_loss": 0.5, "train_accuracy": 0.8, "val_loss": 0.6, "val_accuracy": 0.75}, epoch=1)
logger.close()
```

**CSV output** (`local/logs/training_log.csv`):

```csv
epoch,train_accuracy,train_loss,val_accuracy,val_loss
1,0.45,1.2,0.40,1.3
2,0.55,0.9,0.50,1.0
3,0.65,0.7,0.60,0.8
```

- First call creates the file with headers
- Subsequent calls append rows
- Flushes after each write to prevent data loss on crash
- Preserves existing logs (opens in append mode)

---

## Multi-Head Loss Computation

FG_MFN returns a `Dict[str, Tensor]` of per-attribute logits. During training, the loss is computed as the **sum** of CrossEntropyLoss across all 9 heads:

```python
if isinstance(outputs, dict):
    loss = sum(criterion(logits, labels) for logits in outputs.values())
```

This means each attribute head contributes equally to the gradient. For weighted attribute importance, modify this to include per-head loss weights.

**Accuracy tracking** uses only the first attribute head's predictions for simplicity. For per-attribute accuracy, extend the tracking loop to iterate all heads.

---

## GPU vs CPU

The training script automatically detects CUDA availability:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For forced CPU training:

```python
# Override in config or modify the script
device = torch.device("cpu")
```

**Expected performance:**

| Device | ResNet-18 + DistilBERT | ResNet-50 + DistilBERT |
|--------|----------------------|----------------------|
| CPU | ~5 min/epoch | ~12 min/epoch |
| GPU (V100) | ~30 sec/epoch | ~1 min/epoch |

> Times are approximate for a dataset of 10K samples with batch_size=32.
