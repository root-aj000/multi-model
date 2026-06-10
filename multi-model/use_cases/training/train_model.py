"""
Training utilities for the FG_MFN model.

BUG-01 FIX: Removed apply_warmup — it conflicted with SequentialLR and was
            never called. Warmup is handled entirely by pipeline.py.
BUG-04 FIX: Training accuracy is now computed against labels_a (the dominant
            label when lam >= 0.5) rather than the original labels dict, so
            the reported metric is meaningful during Mixup training.
BUG-06 FIX: _compute_accuracy now returns (correct_count, total_count) so
            the caller can accumulate true sample-level accuracy instead of
            the previous macro-average-across-heads-and-batches metric.
"""

import logging
from typing import Any, Dict, Tuple
from typing import Optional
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Default Mixup alpha parameter for Beta distribution
DEFAULT_MIXUP_ALPHA = 1.0


def mixup_data(
    images: torch.Tensor,
    labels: Any,
    alpha: float = DEFAULT_MIXUP_ALPHA,
) -> Tuple[torch.Tensor, Any, Any, float]:
    """
    Apply Mixup data augmentation to a batch.

    Creates convex combinations of pairs of samples and their labels,
    as described in Zhang et al. (2018).

    Args:
        images: (B, C, H, W) image batch.
        labels: Batch labels — tensor or dict of tensors.
        alpha:  Beta distribution parameter. 0 disables mixing.

    Returns:
        (mixed_images, labels_a, labels_b, lambda_value)
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index, :]

    if isinstance(labels, dict):
        labels_a = {k: v for k, v in labels.items()}
        labels_b = {k: v[index] for k, v in labels.items()}
    else:
        labels_a = labels
        labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(
    criterion: Any,
    preds: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute the Mixup loss: lam * loss(preds, a) + (1-lam) * loss(preds, b).
    """
    return lam * criterion(preds, targets_a) + (1 - lam) * criterion(preds, targets_b)


def _get_criterion(criterion: Any, attr_name: str) -> Any:
    """Return the criterion for a specific attribute, or the global one."""
    if isinstance(criterion, dict):
        return criterion.get(attr_name, torch.nn.CrossEntropyLoss())
    return criterion


def _compute_accuracy(
    outputs: Any,
    labels: Any,
) -> Tuple[int, int]:
    """
    Compute correct prediction count and total sample count.

    Returns (correct_count, total_count) for true sample-level
    micro-average accuracy across all attribute heads.

    Issue 4 note: during Mixup, this is called with labels_a (the dominant
    label). The micro-average across heads is the right metric here — it
    weights each head by its sample count, so heads with more classes don't
    artificially drag down the number.

    Args:
        outputs: Model outputs — tensor or dict of tensors.
        labels:  Ground-truth labels — tensor or dict of tensors.

    Returns:
        (correct_count, total_count)
    """
    if isinstance(outputs, dict):
        correct_total = 0
        sample_total  = 0
        for attr_name, logits in outputs.items():
            if isinstance(labels, dict) and attr_name in labels:
                target = labels[attr_name]
            elif not isinstance(labels, dict):
                target = labels
            else:
                continue
            _, predicted = logits.max(1)
            correct_total += predicted.eq(target).sum().item()
            sample_total  += target.size(0)
        return correct_total, sample_total

    _, predicted = outputs.max(1)
    if isinstance(labels, dict):
        first_target = next(iter(labels.values()))
        return predicted.eq(first_target).sum().item(), first_target.size(0)
    return predicted.eq(labels).sum().item(), labels.size(0)


def _compute_per_attribute_accuracy(
    outputs: Any,
    labels: Any,
) -> Dict[str, float]:
    """
    Compute per-attribute accuracy for a single batch.

    Issue 4 fix: the micro-average in _compute_accuracy can mask a single
    attribute performing poorly. This function returns a per-attribute
    breakdown so the training loop can log it at epoch end.

    Args:
        outputs: Dict[attr_name, logit_tensor] or plain tensor.
        labels:  Dict[attr_name, label_tensor] or plain tensor.

    Returns:
        Dict mapping attr_name → accuracy for this batch.
        Empty dict if outputs is not a dict.
    """
    if not isinstance(outputs, dict):
        return {}

    per_attr: Dict[str, float] = {}
    for attr_name, logits in outputs.items():
        if isinstance(labels, dict) and attr_name in labels:
            target = labels[attr_name]
        elif not isinstance(labels, dict):
            target = labels
        else:
            continue
        _, predicted = logits.max(1)
        per_attr[attr_name] = predicted.eq(target).float().mean().item()
    return per_attr


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = DEFAULT_MIXUP_ALPHA,
    text_max_length: int = 256,
    attribute_loss_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Args:
        model:           The model to train.
        loader:          DataLoader for training data.
        criterion:       Loss function or dict of per-attribute losses.
        optimizer:       Optimizer.
        device:          Compute device.
        mixup_alpha:     Mixup alpha. 0 disables Mixup.
        text_max_length: Fallback sequence length when batch has no text.
                         Must match text_max_length in config.
        attribute_loss_weights: Optional dict mapping attribute name to a
                         scalar multiplier applied to that head's loss before
                         summing. Downweight noisy heads (predicted_ctr,
                         likelihood_shares, attention_score) to prevent their
                         gradients from corrupting the shared representation.

    Returns:
        (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_total = 0
    sample_total  = 0
    # Issue 4: track per-attribute correct/total to expose per-head accuracy
    per_attr_correct: Dict[str, int] = {}
    per_attr_total:   Dict[str, int] = {}

    for batch in loader:
        inputs = batch[0].to(device)
        labels = _move_labels_to_device(batch[1], device)

        batch_size = inputs.size(0)
        if len(batch) >= 4:
            input_ids = batch[2].to(device)
            attention_mask = batch[3].to(device)
        else:
            # No text in batch — use zero-padded tensors
            # text_max_length must match the value used during training
            input_ids = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
            attention_mask = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)

        if mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
        else:
            labels_a = labels
            labels_b = labels
            lam = 1.0

        optimizer.zero_grad()
        outputs = model(inputs, input_ids, attention_mask)

        if isinstance(outputs, dict) and isinstance(labels_a, dict):
            loss = sum(
                (attribute_loss_weights or {}).get(attr, 1.0) *
                mixup_criterion(
                    _get_criterion(criterion, attr),
                    outputs[attr],
                    labels_a[attr],
                    labels_b[attr],
                    lam,
                )
                for attr in outputs if attr in labels_a
            )
        elif isinstance(outputs, dict):
            loss = sum(
                (attribute_loss_weights or {}).get(attr, 1.0) *
                mixup_criterion(
                    _get_criterion(criterion, attr),
                    logits, labels_a, labels_b, lam,
                )
                for attr, logits in outputs.items()
            )
        else:
            loss = mixup_criterion(
                _get_criterion(criterion, ""), outputs, labels_a, labels_b, lam,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss  += loss.item() * batch_size
        total_samples += batch_size

        # Micro-average accuracy (BUG-04 FIX: use labels_a, the dominant label)
        c, n = _compute_accuracy(outputs, labels_a)
        correct_total += c
        sample_total  += n

        # Issue 4: accumulate per-attribute accuracy
        for attr_name, attr_acc in _compute_per_attribute_accuracy(outputs, labels_a).items():
            batch_n = labels_a[attr_name].size(0) if isinstance(labels_a, dict) else batch_size
            per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * batch_n)
            per_attr_total[attr_name]   = per_attr_total.get(attr_name, 0)   + batch_n

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_total / sample_total if sample_total > 0 else 0.0

    # Log per-attribute accuracy so training issues are visible immediately
    per_attr_acc_log = {
        attr: round(per_attr_correct[attr] / per_attr_total[attr], 4)
        for attr in per_attr_correct if per_attr_total.get(attr, 0) > 0
    }
    logger.info("Train epoch: loss=%.4f, accuracy=%.4f", avg_loss, accuracy)
    if per_attr_acc_log:
        logger.info("Train per-attribute accuracy: %s", per_attr_acc_log)
    return avg_loss, accuracy


def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    device: torch.device,
    text_max_length: int = 256,
) -> Tuple[float, float]:
    """
    Run one validation epoch.

    Args:
        model:           The model to evaluate.
        loader:          DataLoader for validation data.
        criterion:       Loss function or dict of per-attribute losses.
        device:          Compute device.
        text_max_length: Fallback sequence length when batch has no text.

    Returns:
        (average_loss, accuracy) for the epoch.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_total = 0
    sample_total  = 0
    per_attr_correct: Dict[str, int] = {}
    per_attr_total:   Dict[str, int] = {}

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            labels = _move_labels_to_device(batch[1], device)

            batch_size = inputs.size(0)
            if len(batch) >= 4:
                input_ids = batch[2].to(device)
                attention_mask = batch[3].to(device)
            else:
                input_ids = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
                attention_mask = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)

            outputs = model(inputs, input_ids, attention_mask)

            if isinstance(outputs, dict) and isinstance(labels, dict):
                loss = sum(
                    _get_criterion(criterion, attr)(outputs[attr], labels[attr])
                    for attr in outputs if attr in labels
                )
            elif isinstance(outputs, dict):
                loss = sum(
                    _get_criterion(criterion, attr)(logits, labels)
                    for attr, logits in outputs.items()
                )
            else:
                loss = _get_criterion(criterion, "")(outputs, labels)

            running_loss  += loss.item() * batch_size
            total_samples += batch_size

            c, n = _compute_accuracy(outputs, labels)
            correct_total += c
            sample_total  += n

            for attr_name, attr_acc in _compute_per_attribute_accuracy(outputs, labels).items():
                batch_n = labels[attr_name].size(0) if isinstance(labels, dict) else batch_size
                per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * batch_n)
                per_attr_total[attr_name]   = per_attr_total.get(attr_name, 0)   + batch_n

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_total / sample_total if sample_total > 0 else 0.0

    per_attr_acc_log = {
        attr: round(per_attr_correct[attr] / per_attr_total[attr], 4)
        for attr in per_attr_correct if per_attr_total.get(attr, 0) > 0
    }
    logger.info("Val epoch: loss=%.4f, accuracy=%.4f", avg_loss, accuracy)
    if per_attr_acc_log:
        logger.info("Val per-attribute accuracy: %s", per_attr_acc_log)
    return avg_loss, accuracy


def _move_labels_to_device(labels: Any, device: torch.device) -> Any:
    """Move labels (tensor or dict of tensors) to the target device."""
    if isinstance(labels, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor)
            else torch.tensor(v, dtype=torch.long, device=device)
            for k, v in labels.items()
        }
    if isinstance(labels, torch.Tensor):
        return labels.to(device)
    try:
        return torch.tensor(labels, dtype=torch.long, device=device)
    except (TypeError, ValueError):
        values = list(labels.values()) if hasattr(labels, "values") else list(labels)
        return torch.tensor(values, dtype=torch.long, device=device)
