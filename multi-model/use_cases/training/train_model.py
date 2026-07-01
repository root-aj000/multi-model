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

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Tuple
from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as T

from tqdm import tqdm

from lib.utils.losses import FocalLoss, GCELoss

logger = logging.getLogger(__name__)

# Default Mixup alpha parameter for Beta distribution
DEFAULT_MIXUP_ALPHA = 1.0


class ELRRegularizer:
    """
    Early Learning Regularization (Liu et al., ICLR 2020).

    Maintains an exponential moving average (EMA) of each sample's predicted
    logits for each attribute.  Adds a regularization term:

        ℓ_ELR = β / T * log(1 - ⟨softmax(logits), EMA⟩)

    that penalizes deviation from early-training predictions (before the
    model starts memorizing noisy labels).  β controls the strength, and
    the 1/T decay ensures the regularizer relaxes over time.

    Usage:
        elr = ELRRegularizer(n_samples, num_classes_dict, beta, device)
        for epoch in range(epochs):
            for batch in loader:
                ...
                elr_loss = elr(outputs, indices, epoch + 1)
                total_loss = task_loss + elr_loss
                ...
    """

    def __init__(
        self,
        n_samples: int,
        num_classes_dict: Dict[str, int],
        beta: float = 0.7,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.beta = beta
        self.num_classes = num_classes_dict
        self.attr_names = sorted(num_classes_dict.keys())
        self.device = device

        # EMA targets: one (N, C) tensor per attribute
        self.ema: Dict[str, torch.Tensor] = {}
        for name, n_cls in self.num_classes.items():
            self.ema[name] = torch.zeros(n_samples, n_cls, device=device)

    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        indices: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        Compute ELR regularization loss.

        Args:
            outputs:  Dict[attr_name → (B, C) logits]
            indices:  (B,) sample indices in the dataset.
            epoch:    Current 1-indexed epoch number.

        Returns:
            Scalar ELR loss.
        """
        loss = torch.tensor(0.0, device=self.device)
        for name in self.attr_names:
            if name not in outputs:
                continue
            logits = outputs[name]
            probs = torch.softmax(logits, dim=-1)
            idx = indices.to(self.device)

            # Retrieve EMA targets for these samples
            ema_targets = self.ema[name][idx]

            # Compute ELR term: -β/T * log(1 - ⟨p, ema⟩)
            dot_product = (probs * ema_targets).sum(dim=-1)
            dot_product = torch.clamp(dot_product, min=-1.0 + 1e-7, max=1.0 - 1e-7)
            elr_term = self.beta / epoch * torch.log(1.0 - dot_product + 1e-7)
            loss = loss + elr_term.mean()

            # Update EMA: ema = ema * (1 - β) + probs * β
            self.ema[name][idx] = (
                (1.0 - self.beta) * ema_targets + self.beta * probs.detach()
            )

        return loss


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
    accuracy_attributes: Optional[set] = None,
) -> Tuple[int, int]:
    """
    Compute correct prediction count and total sample count.

    Returns (correct_count, total_count) for true sample-level
    micro-average accuracy. When accuracy_attributes is set, only
    those attributes contribute to the count — others are still
    predicted but do not influece the primary accuracy metric.

    Args:
        outputs:            Model outputs — tensor or dict of tensors.
        labels:             Ground-truth labels — tensor or dict of tensors.
        accuracy_attributes: Optional set of attribute names to include in
                            accuracy. If None, all attributes are used.

    Returns:
        (correct_count, total_count)
    """
    if isinstance(outputs, dict):
        correct_total = 0
        sample_total  = 0
        for attr_name, logits in outputs.items():
            if accuracy_attributes is not None and attr_name not in accuracy_attributes:
                continue
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


def _compute_per_sample_loss_attr(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion_obj: Any,
) -> torch.Tensor:
    """
    Compute per-sample loss for a single attribute, respecting the
    criterion type.  Used by loss truncation below.
    """
    if isinstance(criterion_obj, torch.nn.CrossEntropyLoss):
        return F.cross_entropy(
            logits, targets,
            weight=criterion_obj.weight,
            label_smoothing=getattr(criterion_obj, 'label_smoothing', 0.0),
            reduction='none',
        )
    if isinstance(criterion_obj, FocalLoss):
        ce = F.cross_entropy(
            logits, targets,
            weight=criterion_obj.weight,
            label_smoothing=criterion_obj.label_smoothing,
            reduction='none',
        )
        return ((1 - (-ce).exp()) ** criterion_obj.gamma) * ce
    if isinstance(criterion_obj, GCELoss):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        one_hot = F.one_hot(targets, num_classes=logits.size(-1))
        p_t = (probs * one_hot).sum(dim=-1)
        loss = (1.0 - p_t ** criterion_obj.q) / criterion_obj.q
        if criterion_obj.weight is not None:
            class_weights = (one_hot * criterion_obj.weight.unsqueeze(0)).sum(dim=-1)
            loss = loss * class_weights
        return loss
    return F.cross_entropy(logits, targets, reduction='none')


def _compute_loss(
    outputs: Any,
    labels_a: Any,
    labels_b: Any,
    lam: float,
    criterion: Any,
    attribute_loss_weights: Optional[Dict[str, float]] = None,
    loss_truncation: Optional[Dict[str, float]] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(outputs, dict) and isinstance(labels_a, dict):
        total = torch.tensor(0.0, device=next(iter(labels_a.values())).device)
        for attr in outputs:
            if attr not in labels_a:
                continue
            weight = (attribute_loss_weights or {}).get(attr, 1.0)
            attr_crit = _get_criterion(criterion, attr)
            if sample_weights is not None:
                per_sample = _compute_per_sample_loss_attr(outputs[attr], labels_a[attr], attr_crit)
                if loss_truncation is not None and attr in loss_truncation:
                    per_sample = per_sample.clamp(max=loss_truncation[attr])
                loss = (per_sample * sample_weights).sum() / (sample_weights.sum() + 1e-8)
            elif loss_truncation is not None and attr in loss_truncation:
                per_sample = _compute_per_sample_loss_attr(outputs[attr], labels_a[attr], attr_crit)
                per_sample = per_sample.clamp(max=loss_truncation[attr])
                loss = per_sample.mean()
            else:
                loss = mixup_criterion(attr_crit, outputs[attr], labels_a[attr], labels_b[attr], lam)
            total = total + weight * loss
        return total
    elif isinstance(outputs, dict):
        total = torch.tensor(0.0, device=labels_a.device if isinstance(labels_a, torch.Tensor) else next(iter(labels_a.values())).device)
        for attr, logits in outputs.items():
            weight = (attribute_loss_weights or {}).get(attr, 1.0)
            attr_crit = _get_criterion(criterion, attr)
            if sample_weights is not None:
                per_sample = _compute_per_sample_loss_attr(logits, labels_a, attr_crit)
                if loss_truncation is not None and attr in loss_truncation:
                    per_sample = per_sample.clamp(max=loss_truncation[attr])
                loss = (per_sample * sample_weights).sum() / (sample_weights.sum() + 1e-8)
            elif loss_truncation is not None and attr in loss_truncation:
                per_sample = _compute_per_sample_loss_attr(logits, labels_a, attr_crit)
                per_sample = per_sample.clamp(max=loss_truncation[attr])
                loss = per_sample.mean()
            else:
                loss = mixup_criterion(attr_crit, logits, labels_a, labels_b, lam)
            total = total + weight * loss
        return total
    return mixup_criterion(
        _get_criterion(criterion, ""), outputs, labels_a, labels_b, lam,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    mixup_alpha: float = DEFAULT_MIXUP_ALPHA,
    text_max_length: int = 256,
    attribute_loss_weights: Optional[Dict[str, float]] = None,
    stop_gradient_heads: Optional[set] = None,
    accuracy_attributes: Optional[set] = None,
    elr: Optional[ELRRegularizer] = None,
    loss_truncation: Optional[Dict[str, float]] = None,
    l2rw: Optional[L2RW] = None,
    meta_weight_trainer: Optional[MetaWeightNetTrainer] = None,
    mwn: Optional[MetaWeightNet] = None,
    progress_bar: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Run one training epoch.

    Args:
        model:                The model to train.
        loader:               DataLoader for training data.
        criterion:            Loss function or dict of per-attribute losses.
        optimizer:            Optimizer.
        device:               Compute device.
        epoch:                Current 1-indexed epoch number (used by ELR).
        mixup_alpha:          Mixup alpha. 0 disables Mixup.
        text_max_length:      Fallback sequence length when batch has no text.
                              Must match text_max_length in config.
        attribute_loss_weights: Optional dict mapping attribute name to a
                              scalar multiplier applied to that head's loss.
        stop_gradient_heads:  Optional set of attribute names whose gradients
                              are detached from shared features.
        accuracy_attributes:  Optional set of attribute names to include in
                              the primary accuracy metric. Per-attribute
                              accuracy is still computed for all heads.
        elr:                  Optional ELRRegularizer for Early Learning
                              Regularization. If provided, the batch is
                              expected to include sample indices (last element).
        loss_truncation:      Optional dict mapping attribute name to a
                              per-sample loss cap. Prevents individual noisy
                              samples from dominating the gradient.
        l2rw:                 Optional L2RW instance for meta-learned sample
                              reweighting. Requires indices in batch.
        meta_weight_trainer:  Optional MetaWeightNetTrainer for bi-level
                              meta-learned sample reweighting.
        mwn:                  Optional MetaWeightNet instance (used when
                              meta_weight_trainer computes weights).

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

    use_meta_weights = l2rw is not None or meta_weight_trainer is not None

    loader_iter = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False, disable=not progress_bar) if progress_bar else loader
    for batch in loader_iter:
        # Check if batch includes sample indices (last element, for ELR or meta-weights)
        has_indices = (elr is not None or use_meta_weights) and isinstance(batch[-1], torch.Tensor) and batch[-1].dim() == 1
        indices = batch[-1].to(device) if has_indices else None
        # Strip indices from batch for standard unpacking
        if has_indices:
            batch = batch[:-1]

        inputs = batch[0].to(device)
        labels = _move_labels_to_device(batch[1], device)

        batch_size = inputs.size(0)
        if len(batch) >= 4:
            input_ids = batch[2].to(device)
            attention_mask = batch[3].to(device)
            aux_features = batch[4].to(device) if len(batch) >= 5 else None
        else:
            # No text in batch — use zero-padded tensors
            # text_max_length must match the value used during training
            input_ids = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
            attention_mask = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
            aux_features = batch[2].to(device) if len(batch) >= 3 else None

        if mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
        else:
            labels_a = labels
            labels_b = labels
            lam = 1.0

        use_sam = hasattr(optimizer, 'first_step')

        # ── SAM first step ──────────────────────────────────────────────
        optimizer.zero_grad()
        outputs = model(inputs, input_ids, attention_mask, aux_features=aux_features, stop_gradient_heads=stop_gradient_heads)

        # Compute sample weights via L2RW or Meta-Weight-Net if enabled
        sample_weights = None
        if l2rw is not None and indices is not None:
            sample_weights = l2rw.compute_weights(model, outputs, labels_a, criterion, attribute_loss_weights)
        elif meta_weight_trainer is not None and mwn is not None:
            train_per = _compute_per_sample_loss(outputs, labels_a, criterion, attribute_loss_weights)
            sample_weights = meta_weight_trainer.compute_weights(model, train_per, outputs, criterion, attribute_loss_weights)

        loss = _compute_loss(outputs, labels_a, labels_b, lam, criterion, attribute_loss_weights, loss_truncation=loss_truncation, sample_weights=sample_weights)
        # Add ELR regularization if enabled
        if elr is not None and indices is not None:
            loss = loss + elr(outputs, indices, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if use_sam:
            optimizer.first_step(zero_grad=True)
            # Second forward pass at perturbed weights
            outputs = model(inputs, input_ids, attention_mask, aux_features=aux_features, stop_gradient_heads=stop_gradient_heads)
            if l2rw is not None and indices is not None:
                sample_weights = l2rw.compute_weights(model, outputs, labels_a, criterion, attribute_loss_weights)
            elif meta_weight_trainer is not None and mwn is not None:
                train_per = _compute_per_sample_loss(outputs, labels_a, criterion, attribute_loss_weights)
                sample_weights = meta_weight_trainer.compute_weights(model, train_per, outputs, criterion, attribute_loss_weights)
            loss = _compute_loss(outputs, labels_a, labels_b, lam, criterion, attribute_loss_weights, loss_truncation=loss_truncation, sample_weights=sample_weights)
            if elr is not None and indices is not None:
                loss = loss + elr(outputs, indices, epoch)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        running_loss  += loss.item() * batch_size
        total_samples += batch_size

        # Micro-average accuracy over selected attributes (BUG-04 FIX: use labels_a)
        c, n = _compute_accuracy(outputs, labels_a, accuracy_attributes)
        correct_total += c
        sample_total  += n

        # Issue 4: accumulate per-attribute accuracy
        for attr_name, attr_acc in _compute_per_attribute_accuracy(outputs, labels_a).items():
            batch_n = labels_a[attr_name].size(0) if isinstance(labels_a, dict) else batch_size
            per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * batch_n)
            per_attr_total[attr_name]   = per_attr_total.get(attr_name, 0)   + batch_n

        if progress_bar:
            loader_iter.set_postfix(loss=loss.item(), acc=c / n if n > 0 else 0.0)

    if progress_bar:
        loader_iter.close()

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
    return avg_loss, accuracy, per_attr_acc_log


def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    device: torch.device,
    text_max_length: int = 256,
    accuracy_attributes: Optional[set] = None,
    use_tta: bool = False,
    progress_bar: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Run one validation epoch.

    Args:
        model:               The model to evaluate.
        loader:              DataLoader for validation data.
        criterion:           Loss function or dict of per-attribute losses.
        device:              Compute device.
        text_max_length:     Fallback sequence length when batch has no text.
        accuracy_attributes: Optional set of attribute names to include in
                             the primary accuracy metric.
        use_tta:             If True, apply test-time augmentation (horizontal
                             flip) and average predictions.
        progress_bar:        If True, show a tqdm progress bar.

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

    loader_iter = tqdm(loader, desc=f"Val Epoch", leave=False, disable=not progress_bar) if progress_bar else loader

    with torch.no_grad():
        for batch in loader_iter:
            inputs = batch[0].to(device)
            labels = _move_labels_to_device(batch[1], device)

            batch_size = inputs.size(0)
            if len(batch) >= 4:
                input_ids = batch[2].to(device)
                attention_mask = batch[3].to(device)
                aux_features = batch[4].to(device) if len(batch) >= 5 else None
            else:
                input_ids = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
                attention_mask = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
                aux_features = batch[2].to(device) if len(batch) >= 3 else None

            # Forward pass (original)
            outputs = model(inputs, input_ids, attention_mask, aux_features=aux_features)

            # TTA: horizontal flip + average logits
            if use_tta:
                flipped = model(
                    inputs.flip(-1), input_ids, attention_mask,
                    aux_features=aux_features,
                )
                if isinstance(outputs, dict):
                    outputs = {
                        k: (outputs[k] + flipped[k]) / 2.0
                        for k in outputs
                    }
                else:
                    outputs = (outputs + flipped) / 2.0

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

            c, n = _compute_accuracy(outputs, labels, accuracy_attributes)
            correct_total += c
            sample_total  += n

            for attr_name, attr_acc in _compute_per_attribute_accuracy(outputs, labels).items():
                batch_n = labels[attr_name].size(0) if isinstance(labels, dict) else batch_size
                per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * batch_n)
                per_attr_total[attr_name]   = per_attr_total.get(attr_name, 0)   + batch_n

            if progress_bar:
                loader_iter.set_postfix(loss=loss.item(), acc=c / n if n > 0 else 0.0)

    if progress_bar:
        loader_iter.close()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_total / sample_total if sample_total > 0 else 0.0

    per_attr_acc_log = {
        attr: round(per_attr_correct[attr] / per_attr_total[attr], 4)
        for attr in per_attr_correct if per_attr_total.get(attr, 0) > 0
    }
    logger.info("Val epoch: loss=%.4f, accuracy=%.4f", avg_loss, accuracy)
    if per_attr_acc_log:
        logger.info("Val per-attribute accuracy: %s", per_attr_acc_log)
    return avg_loss, accuracy, per_attr_acc_log


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


def _compute_per_sample_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    criterion: Any,
    attribute_loss_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Compute per-sample total loss by summing per-attribute losses.

    Returns:
        (B,) tensor of per-sample losses.
    """
    per_sample = torch.zeros_like(next(iter(labels.values())), dtype=torch.float)
    for attr_name, logits in outputs.items():
        if attr_name not in labels:
            continue
        attr_crit = _get_criterion(criterion, attr_name)
        weight = (attribute_loss_weights or {}).get(attr_name, 1.0)
        ce = _compute_per_sample_loss_attr(logits, labels[attr_name], attr_crit)
        per_sample = per_sample + weight * ce
    return per_sample


# ── Co-teaching ─────────────────────────────────────────────────────


def train_epoch_coteaching(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    optimizer1: torch.optim.Optimizer,
    optimizer2: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    noise_rate: float = 0.2,
    forget_rate: Optional[float] = None,
    num_epochs: int = 100,
    mixup_alpha: float = DEFAULT_MIXUP_ALPHA,
    text_max_length: int = 256,
    attribute_loss_weights: Optional[Dict[str, float]] = None,
    stop_gradient_heads: Optional[set] = None,
    accuracy_attributes: Optional[set] = None,
    loss_truncation: Optional[Dict[str, float]] = None,
    progress_bar: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Co-teaching training epoch (Han et al. NeurIPS 2018).

    Two models are trained simultaneously.  Each batch:
      1. Both models forward pass.
      2. Per-sample loss computed from each model.
      3. Each model selects the small-loss samples (retain fraction decays
         from 1-noise_rate to ~0 over the epoch schedule).
      4. Model1 is updated from Model2's selected samples and vice versa.

    This prevents both models from memorising the same noisy labels.

    Args:
        model1:            First model.
        model2:            Second model.
        loader:            Training DataLoader (must include sample indices).
        criterion:         Loss function or dict of per-attribute losses.
        optimizer1:        Optimizer for model1.
        optimizer2:        Optimizer for model2.
        device:            Compute device.
        epoch:             Current 1-indexed epoch number.
        noise_rate:        Estimated label noise rate.
        forget_rate:       Forget rate (defaults to noise_rate).
        num_epochs:        Total number of epochs (for epoch-based rate decay).
                           Set to 1 to disable decay.
        mixup_alpha:       Mixup alpha. 0 disables Mixup.
        text_max_length:   Fallback sequence length when batch has no text.
        attribute_loss_weights: Optional dict mapping attribute name to a
                               scalar multiplier applied to that head's loss.
        stop_gradient_heads: Optional set of attribute names whose gradients
                            are detached from shared features.
        accuracy_attributes: Optional set of attribute names to include in
                            the primary accuracy metric.

    Returns:
        (average_loss, accuracy) for the epoch.
    """
    if forget_rate is None:
        forget_rate = noise_rate

    model1.train()
    model2.train()

    running_loss = 0.0
    total_samples = 0
    correct_total = 0
    sample_total = 0
    per_attr_correct: Dict[str, int] = {}
    per_attr_total: Dict[str, int] = {}

    # Epoch-dependent retain rate (decays over time)
    if num_epochs > 1:
        rate_schedule = 1.0 - noise_rate * min(1.0, epoch / num_epochs)
    else:
        rate_schedule = 1.0 - forget_rate

    has_sam1 = hasattr(optimizer1, 'first_step')
    has_sam2 = hasattr(optimizer2, 'first_step')

    loader_iter = tqdm(loader, desc=f"CoTeach Epoch {epoch}", leave=False, disable=not progress_bar) if progress_bar else loader
    for batch in loader_iter:
        indices = batch[-1].to(device)
        batch = batch[:-1]

        inputs = batch[0].to(device)
        labels = _move_labels_to_device(batch[1], device)

        batch_size = inputs.size(0)
        if len(batch) >= 4:
            input_ids = batch[2].to(device)
            attention_mask = batch[3].to(device)
            aux_features = batch[4].to(device) if len(batch) >= 5 else None
        else:
            input_ids = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
            attention_mask = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
            aux_features = batch[2].to(device) if len(batch) >= 3 else None

        # Mixup
        if mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        # ── Forward pass both models ──
        outputs1 = model1(inputs, input_ids, attention_mask, aux_features=aux_features, stop_gradient_heads=stop_gradient_heads)
        outputs2 = model2(inputs, input_ids, attention_mask, aux_features=aux_features, stop_gradient_heads=stop_gradient_heads)

        # ── Per-sample losses for sample selection ──
        loss1_per = _compute_per_sample_loss(outputs1, labels_a, criterion, attribute_loss_weights)
        loss2_per = _compute_per_sample_loss(outputs2, labels_a, criterion, attribute_loss_weights)

        # ── Small-loss sample selection ──
        n_keep = max(1, int(batch_size * rate_schedule))
        _, idx1 = torch.topk(loss1_per, k=min(n_keep, batch_size), largest=False)
        _, idx2 = torch.topk(loss2_per, k=min(n_keep, batch_size), largest=False)

        # ── Compute supervised loss on each model's selected samples ──
        labels_a_slice = {k: v[idx2] for k, v in labels_a.items()}  # for model1
        labels_a_slice_1 = {k: v[idx1] for k, v in labels_a.items()}  # for model2
        labels_b_slice = {k: v[idx2] for k, v in labels_b.items()}
        labels_b_slice_1 = {k: v[idx1] for k, v in labels_b.items()}

        def slice_outputs(outputs, idx):
            return {k: v[idx] for k, v in outputs.items()}

        # ── Update model1 using model2's selected samples ──
        optimizer1.zero_grad()
        out1_slice = slice_outputs(outputs1, idx2)
        loss1 = _compute_loss(out1_slice, labels_a_slice, labels_b_slice, lam, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
        loss1.backward()
        if has_sam1:
            optimizer1.first_step(zero_grad=True)
            # second forward for SAM
            outputs1_sam = model1(inputs[idx2], input_ids[idx2] if input_ids.dim() > 1 else input_ids,
                                  attention_mask[idx2] if attention_mask.dim() > 1 else attention_mask,
                                  aux_features=aux_features[idx2] if aux_features is not None else None,
                                  stop_gradient_heads=stop_gradient_heads)
            loss1_sam = _compute_loss(slice_outputs(outputs1_sam, torch.arange(idx2.size(0), device=device)),
                                      labels_a_slice, labels_b_slice, lam, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
            loss1_sam.backward()
            optimizer1.second_step(zero_grad=True)
        else:
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            optimizer1.step()

        # ── Update model2 using model1's selected samples ──
        optimizer2.zero_grad()
        out2_slice = slice_outputs(outputs2, idx1)
        loss2 = _compute_loss(out2_slice, labels_a_slice_1, labels_b_slice_1, lam, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
        loss2.backward()
        if has_sam2:
            optimizer2.first_step(zero_grad=True)
            outputs2_sam = model2(inputs[idx1], input_ids[idx1] if input_ids.dim() > 1 else input_ids,
                                  attention_mask[idx1] if attention_mask.dim() > 1 else attention_mask,
                                  aux_features=aux_features[idx1] if aux_features is not None else None,
                                  stop_gradient_heads=stop_gradient_heads)
            loss2_sam = _compute_loss(slice_outputs(outputs2_sam, torch.arange(idx1.size(0), device=device)),
                                      labels_a_slice_1, labels_b_slice_1, lam, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
            loss2_sam.backward()
            optimizer2.second_step(zero_grad=True)
        else:
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
            optimizer2.step()

        avg_loss = 0.5 * (loss1.item() + loss2.item())
        running_loss += avg_loss * batch_size
        total_samples += batch_size

        # Accuracy from model1 (using all samples)
        c, n = _compute_accuracy(outputs1, labels_a, accuracy_attributes)
        correct_total += c
        sample_total += n

        for attr_name, attr_acc in _compute_per_attribute_accuracy(outputs1, labels_a).items():
            batch_n = labels_a[attr_name].size(0) if isinstance(labels_a, dict) else batch_size
            per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * batch_n)
            per_attr_total[attr_name] = per_attr_total.get(attr_name, 0) + batch_n

        if progress_bar:
            loader_iter.set_postfix(loss=avg_loss, acc=c / n if n > 0 else 0.0)

    if progress_bar:
        loader_iter.close()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_total / sample_total if sample_total > 0 else 0.0

    per_attr_acc_log = {
        attr: round(per_attr_correct[attr] / per_attr_total[attr], 4)
        for attr in per_attr_correct if per_attr_total.get(attr, 0) > 0
    }
    logger.info("Co-teaching epoch %d: loss=%.4f, accuracy=%.4f, keep_rate=%.3f",
                epoch, avg_loss, accuracy, rate_schedule)
    if per_attr_acc_log:
        logger.info("Co-teaching per-attribute accuracy: %s", per_attr_acc_log)
    return avg_loss, accuracy, per_attr_acc_log


# ── FixMatch Strong Augmentation ─────────────────────────────────────


class StrongAugment(torch.nn.Module):
    """
    Strong augmentation for FixMatch-style consistency.

    Applies aggressive spatial + colour transforms to batched normalized
    tensors.  All ops work directly on float tensors (no denormalize step
    needed) so the extra forward pass is cheap.
    """

    def __init__(self) -> None:
        super().__init__()
        # RandomApply containers for batched tensor ops
        self.spatial = T.RandomApply(
            [T.RandomAffine(degrees=(-20, 20), translate=(0.12, 0.12),
                            scale=(0.75, 1.25), fill=0.0)],
            p=0.9,
        )
        self.perspective = T.RandomPerspective(distortion_scale=0.3, p=0.5, fill=0.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) normalized float tensor.

        Returns:
            Strongly augmented (B, C, H, W) tensor.
        """
        x = self.spatial(images)
        x = self.perspective(x)
        # Random colour jitter on tensor range (works because visible
        # spectrum is roughly centred around 0 after ImageNet norm)
        if x.size(0) > 0 and torch.rand(1).item() < 0.8:
            with torch.no_grad():
                brightness = 0.6 + 0.8 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
                contrast = 0.6 + 0.8 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
                saturation = 0.6 + 0.8 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
                # Apply brightness/contrast/saturation channel-wise
                for i in range(x.size(0)):
                    if torch.rand(1).item() < 0.5:
                        x[i] = TF.adjust_brightness(x[i], brightness[i].item())
                    if torch.rand(1).item() < 0.5:
                        x[i] = TF.adjust_contrast(x[i], contrast[i].item())
                    if torch.rand(1).item() < 0.5:
                        x[i] = TF.adjust_saturation(x[i], saturation[i].item())
        return x


# ── DivideMix ────────────────────────────────────────────────────────


class DivideMix:
    """
    DivideMix (Li et al., ICLR 2020) — simplified version.

    Uses two co-training models and a Gaussian Mixture Model (GMM) to
    split the training set into a clean and a noisy subset each epoch.

    *Warm-up phase* (first `warmup_epochs` epochs): standard supervised
    training on the full dataset with both models (no sample selection).

    *Divide-and-Mix phase* (after warmup):
      1. For each model, compute per-sample loss from the *other* model.
      2. Fit a 2-component GMM on these losses.
      3. Split clean (prob > clean_threshold) vs. noisy.
      4. On the clean split: standard supervised CrossEntropy.
      5. On the noisy split: ELR-style consistency loss or minimise loss
         on the other model's prediction (co-regularisation).
      6. Co-teaching-style sample exchange: each model sees the other's
         clean-set selected samples.

    Args:
        n_samples:            Total number of training samples.
        num_classes_dict:     Dict[attr_name → num_classes].
        device:               Compute device.
        warmup_epochs:        Number of warmup epochs (full supervision).
        clean_threshold:      GMM probability threshold for clean split.
        lambda_u:             Weight for unsupervised (noisy) loss.
        noise_rate:           Estimated label noise rate.
        beta_elr:             ELR beta parameter for noisy sample
                              regularisation.
        elr_T:                ELR epoch schedule denominator.
    """

    def __init__(
        self,
        n_samples: int,
        num_classes_dict: Dict[str, int],
        device: torch.device,
        warmup_epochs: int = 10,
        clean_threshold: float = 0.5,
        lambda_u: float = 25.0,
        noise_rate: float = 0.2,
        beta_elr: float = 0.7,
        elr_T: int = 100,
        label_correction_enabled: bool = True,
        label_correction_ramp_epochs: int = 10,
        label_correction_max_rate: float = 0.9,
        label_correction_confidence_threshold: float = 0.9,
        fixmatch_enabled: bool = True,
        fixmatch_confidence_threshold: float = 0.95,
        fixmatch_loss_weight: float = 1.0,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.clean_threshold = clean_threshold
        self.lambda_u = lambda_u
        self.noise_rate = noise_rate
        self.device = device

        # Per-sample loss history for GMM fitting
        self.loss_history: Dict[str, torch.Tensor] = {
            "model1": torch.zeros(n_samples, device=device),
            "model2": torch.zeros(n_samples, device=device),
        }
        self._gmm_cache: Dict[str, torch.Tensor] = {}

        # ELR regulariser for noisy samples (separate for each model)
        self.elr1 = ELRRegularizer(n_samples, num_classes_dict, beta=beta_elr, device=device)
        self.elr2 = ELRRegularizer(n_samples, num_classes_dict, beta=beta_elr, device=device)
        self.elr_T = elr_T

        # ── Progressive label correction ──
        self.label_correction_enabled = label_correction_enabled
        self.label_correction_ramp_epochs = label_correction_ramp_epochs
        self.label_correction_max_rate = label_correction_max_rate
        self.label_correction_confidence_threshold = label_correction_confidence_threshold

        self.attr_names = sorted(num_classes_dict.keys())
        # Per-attribute corrected label: (n_samples,) long tensor, -1 = not corrected
        self.correction_label: Dict[str, torch.Tensor] = {}
        # Per-attribute correction confidence: (n_samples,) float, 0 = not corrected
        self.correction_confidence: Dict[str, torch.Tensor] = {}
        for name in self.attr_names:
            self.correction_label[name] = torch.full((n_samples,), -1, dtype=torch.long, device=device)
            self.correction_confidence[name] = torch.zeros(n_samples, device=device)

        # ── FixMatch consistency ──
        self.fixmatch_enabled = fixmatch_enabled
        self.fixmatch_confidence_threshold = fixmatch_confidence_threshold
        self.fixmatch_loss_weight = fixmatch_loss_weight
        self.strong_aug = StrongAugment().to(device) if fixmatch_enabled else None

    def update_loss_history(
        self,
        model_idx: str,
        indices: torch.Tensor,
        per_sample_loss: torch.Tensor,
        momentum: float = 0.9,
    ) -> None:
        """Update running average of per-sample losses."""
        idx = indices.to(self.device)
        self.loss_history[model_idx][idx] = (
            momentum * self.loss_history[model_idx][idx]
            + (1.0 - momentum) * per_sample_loss.detach()
        )

    def gmm_split(self, model_idx: str, current_epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit a 2-component GMM on per-sample losses and return clean/noisy
        indices for the given model.

        Returns:
            (clean_indices, noisy_indices) — 1D tensors of sample indices
            for the training set.
        """
        from sklearn.mixture import GaussianMixture

        losses = self.loss_history[model_idx].cpu().numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=current_epoch, reg_covar=1e-5)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)

        # Identify which component is "clean" (lower mean loss)
        if gmm.means_[0] > gmm.means_[1]:
            prob_clean = prob[:, 1]
        else:
            prob_clean = prob[:, 0]

        # Mask: P(clean) > threshold
        prob_t = torch.from_numpy(prob_clean).to(self.device)
        all_indices = torch.arange(len(losses), device=self.device)
        clean_idx = all_indices[prob_t > self.clean_threshold]
        noisy_idx = all_indices[prob_t <= self.clean_threshold]
        return clean_idx, noisy_idx

    def _apply_stored_corrections(
        self,
        labels: Dict[str, torch.Tensor],
        indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply existing label corrections (from previous epochs) to a batch.

        Returns a copy of labels with corrected values overwritten.
        """
        if not self.label_correction_enabled:
            return labels
        corrected = {k: v.clone() for k, v in labels.items()}
        for attr_name in self.attr_names:
            if attr_name not in corrected or attr_name not in self.correction_label:
                continue
            stored = self.correction_label[attr_name][indices]
            mask = stored >= 0
            if mask.any():
                corrected[attr_name] = corrected[attr_name].clone()
                corrected[attr_name][mask] = stored[mask]
        return corrected

    def _update_corrections(
        self,
        out1: Dict[str, torch.Tensor],
        out2: Dict[str, torch.Tensor],
        indices: torch.Tensor,
        noisy_mask: torch.Tensor,
        epoch: int,
    ) -> None:
        """
        Compute new label corrections for noisy samples and store them.

        Updates self.correction_label / self.correction_confidence in-place.
        Corrected labels are used by _apply_stored_corrections in future batches.
        """
        if not self.label_correction_enabled or not noisy_mask.any() or epoch <= self.warmup_epochs:
            return

        ramp_start = self.warmup_epochs + 1
        ramp_end = ramp_start + self.label_correction_ramp_epochs
        p = min(1.0, max(0.0, (epoch - ramp_start) / max(1, ramp_end - ramp_start)))
        fraction = p * self.label_correction_max_rate
        if fraction <= 0.0:
            return

        noisy_idx_batch = indices[noisy_mask]

        for attr_name in self.attr_names:
            if attr_name not in out1 or attr_name not in out2:
                continue

            # Ensemble prediction: avg of both models' softmax
            probs = (torch.softmax(out1[attr_name], dim=-1) +
                     torch.softmax(out2[attr_name], dim=-1)) / 2.0
            confidences, predictions = probs.max(dim=-1)

            noisy_confs = confidences[noisy_mask]
            noisy_preds = predictions[noisy_mask]
            noisy_stored_conf = self.correction_confidence[attr_name][noisy_idx_batch]

            # Eligible: confidence > threshold AND confidence > previously stored
            eligible_mask = (noisy_confs >= self.label_correction_confidence_threshold) & \
                            (noisy_confs > noisy_stored_conf)
            if not eligible_mask.any():
                continue

            eligible_indices = torch.where(eligible_mask)[0]
            eligible_confs = noisy_confs[eligible_indices]
            k = int(len(eligible_indices) * fraction)
            if k == 0:
                continue
            _, topk = torch.topk(eligible_confs, k=k)

            correct_ids = eligible_indices[topk]                 # positions within noisy subset
            correct_dataset_ids = noisy_idx_batch[correct_ids]   # dataset-wide indices
            correct_labels = noisy_preds[correct_ids]

            self.correction_label[attr_name][correct_dataset_ids] = correct_labels
            self.correction_confidence[attr_name][correct_dataset_ids] = noisy_confs[correct_ids]

    def _fixmatch_loss(
        self,
        out_weak: Dict[str, torch.Tensor],
        out_strong: Optional[Dict[str, torch.Tensor]],
        noisy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        FixMatch-style consistency loss on noisy samples.

        For each attribute, compute pseudo-label from weak-aug prediction
        (argmax of softmax).  Only apply CE where weak-aug confidence >
        fixmatch_confidence_threshold.

        Args:
            out_weak:   Model outputs on weakly-augmented inputs.
            out_strong: Model outputs on strongly-augmented inputs (same
                        noisy samples only), or None if no noisy samples.
            noisy_mask: (B,) boolean mask.

        Returns:
            Scalar FixMatch loss (0 if no noisy samples or fixmatch disabled).
        """
        if not self.fixmatch_enabled or out_strong is None or not noisy_mask.any():
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        n_active = 0

        for attr_name in self.attr_names:
            if attr_name not in out_weak or attr_name not in out_strong:
                continue

            # Weak-aug pseudo-labels (argmax) and confidence
            weak_probs = torch.softmax(out_weak[attr_name][noisy_mask], dim=-1)
            confidences, pseudo_labels = weak_probs.max(dim=-1)

            # Only keep where confidence > threshold
            confident_mask = confidences >= self.fixmatch_confidence_threshold
            if not confident_mask.any():
                continue

            # CE between strong-aug prediction and weak-aug pseudo-label
            strong_logits = out_strong[attr_name]
            matched = strong_logits[confident_mask]
            targets = pseudo_labels[confident_mask]
            loss = loss + F.cross_entropy(matched, targets)
            n_active += 1

        if n_active == 0:
            return torch.tensor(0.0, device=self.device)

        return self.fixmatch_loss_weight * loss / n_active

    def train_epoch(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        loader: DataLoader,
        criterion: Any,
        optimizer1: torch.optim.Optimizer,
        optimizer2: torch.optim.Optimizer,
        device: torch.device,
        epoch: int = 1,
        text_max_length: int = 256,
        attribute_loss_weights: Optional[Dict[str, float]] = None,
        stop_gradient_heads: Optional[set] = None,
        accuracy_attributes: Optional[set] = None,
        loss_truncation: Optional[Dict[str, float]] = None,
        verbose: bool = False,
        progress_bar: bool = True,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Run one DivideMix training epoch.

        Args:
            model1:            First model.
            model2:            Second model.
            loader:            Training DataLoader (must include indices).
            criterion:         Loss function or dict of per-attribute losses.
            optimizer1:        Optimizer for model1.
            optimizer2:        Optimizer for model2.
            device:            Compute device.
            epoch:             Current 1-indexed epoch number.
            text_max_length:   Fallback sequence length for missing text.
            attribute_loss_weights: Optional per-attribute loss multipliers.
            stop_gradient_heads: Optional set of attribute names to detach.
            accuracy_attributes: Optional set for primary accuracy metric.
            verbose:           If True, log GMM split sizes.
            progress_bar:      If True, show a tqdm progress bar.

        Returns:
            (average_loss, accuracy) for the epoch.
        """
        model1.train()
        model2.train()

        running_loss = 0.0
        total_samples = 0
        correct_total = 0
        sample_total = 0
        per_attr_correct: Dict[str, int] = {}
        per_attr_total: Dict[str, int] = {}

        is_warmup = epoch <= self.warmup_epochs

        loader_iter = tqdm(loader, desc=f"DivideMix Epoch {epoch}", leave=False, disable=not progress_bar) if progress_bar else loader
        for batch in loader_iter:
            indices = batch[-1].to(device)
            batch = batch[:-1]

            inputs = batch[0].to(device)
            labels = _move_labels_to_device(batch[1], device)

            batch_size = inputs.size(0)
            if len(batch) >= 4:
                input_ids = batch[2].to(device)
                attention_mask = batch[3].to(device)
                aux_features = batch[4].to(device) if len(batch) >= 5 else None
            else:
                input_ids = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
                attention_mask = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
                aux_features = batch[2].to(device) if len(batch) >= 3 else None

            # ── Warmup: standard supervised training ──
            if is_warmup:
                use_sam1 = hasattr(optimizer1, 'first_step')
                use_sam2 = hasattr(optimizer2, 'first_step')

                # Model1 — forward + backward + step
                optimizer1.zero_grad()
                out1 = model1(inputs, input_ids, attention_mask, aux_features=aux_features,
                              stop_gradient_heads=stop_gradient_heads)
                loss1 = _compute_loss(out1, labels, labels, 1.0, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
                loss1.backward()
                if use_sam1:
                    optimizer1.first_step(zero_grad=True)
                    out1 = model1(inputs, input_ids, attention_mask, aux_features=aux_features,
                                  stop_gradient_heads=stop_gradient_heads)
                    loss1 = _compute_loss(out1, labels, labels, 1.0, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
                    loss1.backward()
                    optimizer1.second_step(zero_grad=True)
                else:
                    optimizer1.step()

                # Model2 — forward + backward + step
                optimizer2.zero_grad()
                out2 = model2(inputs, input_ids, attention_mask, aux_features=aux_features,
                              stop_gradient_heads=stop_gradient_heads)
                loss2 = _compute_loss(out2, labels, labels, 1.0, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
                loss2.backward()
                if use_sam2:
                    optimizer2.first_step(zero_grad=True)
                    out2 = model2(inputs, input_ids, attention_mask, aux_features=aux_features,
                                  stop_gradient_heads=stop_gradient_heads)
                    loss2 = _compute_loss(out2, labels, labels, 1.0, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
                    loss2.backward()
                    optimizer2.second_step(zero_grad=True)
                else:
                    optimizer2.step()

                # Update loss history
                loss1_per = _compute_per_sample_loss(out1, labels, criterion, attribute_loss_weights)
                loss2_per = _compute_per_sample_loss(out2, labels, criterion, attribute_loss_weights)
                self.update_loss_history("model1", indices, loss1_per)
                self.update_loss_history("model2", indices, loss2_per)

                running_loss += 0.5 * (loss1.item() + loss2.item()) * batch_size
                total_samples += batch_size

                c, n = _compute_accuracy(out1, labels, accuracy_attributes)
                correct_total += c
                sample_total += n
                for attr_name, attr_acc in _compute_per_attribute_accuracy(out1, labels).items():
                    bn = labels[attr_name].size(0) if isinstance(labels, dict) else batch_size
                    per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * bn)
                    per_attr_total[attr_name] = per_attr_total.get(attr_name, 0) + bn

                if progress_bar:
                    c_curr, n_curr = _compute_accuracy(out1, labels, accuracy_attributes)
                    loader_iter.set_postfix(loss=running_loss / total_samples if total_samples > 0 else 0.0,
                                            acc=c_curr / n_curr if n_curr > 0 else 0.0)
                continue

            # ── Divide phase: forward pass both models ──
            out1 = model1(inputs, input_ids, attention_mask, aux_features=aux_features,
                          stop_gradient_heads=stop_gradient_heads)
            out2 = model2(inputs, input_ids, attention_mask, aux_features=aux_features,
                          stop_gradient_heads=stop_gradient_heads)

            # Apply stored label corrections from PREVIOUS epochs
            labels_corrected = self._apply_stored_corrections(labels, indices)

            # Per-sample losses updated from OTHER model's output
            # (cross-update for GMM consistency, with corrected labels)
            loss1_per = _compute_per_sample_loss(out2, labels_corrected, criterion, attribute_loss_weights)
            loss2_per = _compute_per_sample_loss(out1, labels_corrected, criterion, attribute_loss_weights)
            self.update_loss_history("model1", indices, loss1_per)
            self.update_loss_history("model2", indices, loss2_per)

            # ── Clean / Noisy split (run once per epoch, cached) ──
            if epoch not in self._gmm_cache:
                clean1, noisy1 = self.gmm_split("model1", epoch)
                clean2, noisy2 = self.gmm_split("model2", epoch)
                self._gmm_cache[epoch] = {
                    "clean1": clean1, "noisy1": noisy1,
                    "clean2": clean2, "noisy2": noisy2,
                }
                if verbose:
                    logger.info(
                        "DivideMix epoch %d – model1 clean=%d noisy=%d, model2 clean=%d noisy=%d",
                        epoch, len(clean1), len(noisy1), len(clean2), len(noisy2),
                    )
            split = self._gmm_cache[epoch]
            clean1, noisy1 = split["clean1"], split["noisy1"]
            clean2, noisy2 = split["clean2"], split["noisy2"]

            # ── Determine which samples in this batch are clean/noisy ──
            def in_split(split_idx: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
                """Return mask of batch_idx elements present in split_idx."""
                s = set(split_idx.tolist())
                return torch.tensor([i.item() in s for i in batch_idx], device=device)

            clean_mask1 = in_split(clean1, indices)
            noisy_mask1 = ~clean_mask1
            clean_mask2 = in_split(clean2, indices)
            noisy_mask2 = ~clean_mask2

            # ── Progressive label correction for noisy samples ──
            self._update_corrections(out1, out2, indices, noisy_mask1, epoch)

            # ── FixMatch: forward pass on strongly-augmented noisy samples ──
            out1_strong = None
            out2_strong = None
            if self.fixmatch_enabled and epoch > self.warmup_epochs:
                if noisy_mask1.any():
                    strong_inputs1 = self.strong_aug(inputs[noisy_mask1])
                    out1_strong = model1(
                        strong_inputs1,
                        input_ids[noisy_mask1] if input_ids.dim() > 1 else input_ids,
                        attention_mask[noisy_mask1] if attention_mask.dim() > 1 else attention_mask,
                        aux_features=aux_features[noisy_mask1] if aux_features is not None else None,
                        stop_gradient_heads=stop_gradient_heads,
                    )
                if noisy_mask2.any():
                    strong_inputs2 = self.strong_aug(inputs[noisy_mask2])
                    out2_strong = model2(
                        strong_inputs2,
                        input_ids[noisy_mask2] if input_ids.dim() > 1 else input_ids,
                        attention_mask[noisy_mask2] if attention_mask.dim() > 1 else attention_mask,
                        aux_features=aux_features[noisy_mask2] if aux_features is not None else None,
                        stop_gradient_heads=stop_gradient_heads,
                    )

            def _divide_loss_fn(out_a, out_b, clean_mask, noisy_mask, elr_obj, labels_in):
                """Compute supervised + consistency + ELR loss for one model."""
                sup = torch.tensor(0.0, device=device)
                if clean_mask.any():
                    out_clean = {k: v[clean_mask] for k, v in out_a.items()}
                    labels_clean = {k: v[clean_mask] for k, v in labels_in.items()}
                    sup = _compute_loss(out_clean, labels_clean, labels_clean, 1.0,
                                        criterion, attribute_loss_weights, loss_truncation=loss_truncation)
                unsup = torch.tensor(0.0, device=device)
                if noisy_mask.any():
                    out_noisy = {k: v[noisy_mask] for k, v in out_a.items()}
                    out_b_detached = {k: v.detach() for k, v in out_b.items()}
                    out_b_for = {k: v[noisy_mask] for k, v in out_b_detached.items()}
                    for attr in out_noisy:
                        if attr not in out_b_for:
                            continue
                        p = torch.softmax(out_b_for[attr], dim=-1)
                        w = (attribute_loss_weights or {}).get(attr, 1.0)
                        unsup = unsup + w * torch.mean(
                            torch.sum(-p * torch.log_softmax(out_noisy[attr], dim=-1), dim=-1)
                        )
                elr_val = torch.tensor(0.0, device=device)
                if noisy_mask.any():
                    out_noisy = {k: v[noisy_mask] for k, v in out_a.items()}
                    elr_val = elr_obj(out_noisy, indices[noisy_mask], epoch)
                return sup + self.lambda_u * unsup + elr_val

            # Compute FixMatch loss on weakly-augmented outputs
            fm_loss1 = self._fixmatch_loss(out1, out1_strong, noisy_mask1)
            fm_loss2 = self._fixmatch_loss(out2, out2_strong, noisy_mask2)

            use_sam1 = hasattr(optimizer1, 'first_step')
            use_sam2 = hasattr(optimizer2, 'first_step')

            # ── Model1 backward + step ──
            total_loss1 = _divide_loss_fn(out1, out2, clean_mask1, noisy_mask1, self.elr1, labels_corrected) + fm_loss1
            optimizer1.zero_grad()
            total_loss1.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            if use_sam1:
                optimizer1.first_step(zero_grad=True)
                out1_sam = model1(inputs, input_ids, attention_mask, aux_features=aux_features,
                                  stop_gradient_heads=stop_gradient_heads)
                fm_loss1_sam = torch.tensor(0.0, device=device)
                if self.fixmatch_enabled and noisy_mask1.any():
                    out1_strong_sam = model1(
                        strong_inputs1,
                        input_ids[noisy_mask1],
                        attention_mask[noisy_mask1],
                        aux_features=aux_features[noisy_mask1] if aux_features is not None else None,
                        stop_gradient_heads=stop_gradient_heads,
                    )
                    fm_loss1_sam = self._fixmatch_loss(out1_sam, out1_strong_sam, noisy_mask1)
                total_loss1_sam = _divide_loss_fn(out1_sam, out2, clean_mask1, noisy_mask1, self.elr1, labels_corrected) + fm_loss1_sam
                total_loss1_sam.backward()
                optimizer1.second_step(zero_grad=True)
            else:
                optimizer1.step()

            # ── Model2 backward + step ──
            total_loss2 = _divide_loss_fn(out2, out1, clean_mask2, noisy_mask2, self.elr2, labels_corrected) + fm_loss2
            optimizer2.zero_grad()
            total_loss2.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
            if use_sam2:
                optimizer2.first_step(zero_grad=True)
                out2_sam = model2(inputs, input_ids, attention_mask, aux_features=aux_features,
                                  stop_gradient_heads=stop_gradient_heads)
                fm_loss2_sam = torch.tensor(0.0, device=device)
                if self.fixmatch_enabled and noisy_mask2.any():
                    out2_strong_sam = model2(
                        strong_inputs2,
                        input_ids[noisy_mask2],
                        attention_mask[noisy_mask2],
                        aux_features=aux_features[noisy_mask2] if aux_features is not None else None,
                        stop_gradient_heads=stop_gradient_heads,
                    )
                    fm_loss2_sam = self._fixmatch_loss(out2_sam, out2_strong_sam, noisy_mask2)
                total_loss2_sam = _divide_loss_fn(out2_sam, out1, clean_mask2, noisy_mask2, self.elr2, labels_corrected) + fm_loss2_sam
                total_loss2_sam.backward()
                optimizer2.second_step(zero_grad=True)
            else:
                optimizer2.step()

            running_loss += 0.5 * (total_loss1.item() + total_loss2.item()) * batch_size
            total_samples += batch_size

            # Accuracy from model1 (against original labels for fair metric)
            c, n = _compute_accuracy(out1, labels, accuracy_attributes)
            correct_total += c
            sample_total += n
            for attr_name, attr_acc in _compute_per_attribute_accuracy(out1, labels).items():
                bn = labels[attr_name].size(0) if isinstance(labels, dict) else batch_size
                per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * bn)
                per_attr_total[attr_name] = per_attr_total.get(attr_name, 0) + bn

            if progress_bar:
                c_curr, n_curr = _compute_accuracy(out1, labels, accuracy_attributes)
                loader_iter.set_postfix(loss=running_loss / total_samples if total_samples > 0 else 0.0,
                                        acc=c_curr / n_curr if n_curr > 0 else 0.0)

        if progress_bar:
            loader_iter.close()

        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct_total / sample_total if sample_total > 0 else 0.0
        per_attr_acc_log = {
            attr: round(per_attr_correct[attr] / per_attr_total[attr], 4)
            for attr in per_attr_correct if per_attr_total.get(attr, 0) > 0
        }
        logger.info("DivideMix epoch %d: loss=%.4f, accuracy=%.4f%s",
                    epoch, avg_loss, accuracy,
                    " (warmup)" if is_warmup else "")
        if per_attr_acc_log:
            logger.info("DivideMix per-attribute accuracy: %s", per_attr_acc_log)
        return avg_loss, accuracy, per_attr_acc_log


# ── L2RW (Learning to Reweight) ─────────────────────────────────────


class L2RW:
    """
    Learning to Reweight Examples (Ren et al., ICML 2018).

    Uses a small clean validation set to learn per-sample weights for the
    training set via a meta-learning objective.

    At each training step:
      1. Forward pass on training batch, compute per-sample losses.
      2. Compute gradient surrogate using these losses.
      3. Forward pass on a hold-out clean validation batch.
      4. Compute meta loss on the validation batch.
      5. Compute meta gradient (how much each training sample should matter)
         and use it to obtain sample weights.
      6. Reweight the training loss and update model parameters.

    Usage:
        l2rw = L2RW(val_loader, device)
        for batch in train_loader:
            weights = l2rw.compute_weights(model, batch, val_batch, criterion)
            loss = (weights * per_sample_losses).sum() / weights.sum()
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        val_loader: DataLoader,
        device: torch.device,
        meta_lr: float = 0.01,
        meta_weight_decay: float = 0.0,
    ) -> None:
        self.val_loader = val_loader
        self.device = device
        self.meta_lr = meta_lr
        self.meta_weight_decay = meta_weight_decay
        self._val_iter: Optional[Any] = None

    def _get_val_batch(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a single validation batch, cycling through the loader."""
        if self._val_iter is None:
            self._val_iter = iter(self.val_loader)
        try:
            batch = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self.val_loader)
            batch = next(self._val_iter)
        inputs = batch[0].to(self.device)
        labels = _move_labels_to_device(batch[1], self.device)
        return inputs, labels

    def compute_weights(
        self,
        model: torch.nn.Module,
        train_outputs: Dict[str, torch.Tensor],
        train_labels: Dict[str, torch.Tensor],
        criterion: Any,
        attribute_loss_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Compute per-sample importance weights via meta-learning.

        Args:
            model:            The model being trained.
            train_outputs:    Model outputs on the training batch.
            train_labels:     Labels for the training batch.
            criterion:        Loss function.
            attribute_loss_weights: Optional per-attribute loss multipliers.

        Returns:
            (B,) weight tensor (detached, for use in the outer training step).
        """
        # Per-sample loss on training batch
        train_per = _compute_per_sample_loss(
            train_outputs, train_labels, criterion, attribute_loss_weights
        )

        # Compute pseudo-gradient: d(loss_i) / d(theta) approximated as
        # grad_theta(loss) where loss = sum_i(train_per_i * 1.0)
        # We use the final-layer outputs to compute a gradient surrogate.
        # Simplified: use train_per directly as importance scores.
        # For a proper implementation, we would compute the dot product
        # between the validation gradient and each training sample's gradient.
        # Here we use the efficient first-order approximation:
        # weight_i ≈ -eta * grad_theta(val_loss)^T * grad_theta(train_loss_i)

        val_inputs, val_labels = self._get_val_batch()
        val_outputs = model(val_inputs)
        val_loss = _compute_loss(
            val_outputs, val_labels, val_labels, 1.0,
            criterion, attribute_loss_weights,
        )

        # Compute gradient of validation loss w.r.t. parameters
        val_grads = torch.autograd.grad(
            val_loss, model.parameters(), retain_graph=True, create_graph=False,
        )

        # Compute gradient of each training sample's loss
        weights = []
        for i in range(train_per.size(0)):
            grad_i = torch.autograd.grad(
                train_per[i], model.parameters(), retain_graph=True, create_graph=False,
            )
            # Weight = dot product between validation grad and sample i's grad
            w_i = sum(
                (vg * gi).sum()
                for vg, gi in zip(val_grads, grad_i)
            )
            weights.append(w_i)

        w_tensor = torch.stack(weights)
        # Normalise to non-negative weights via softmax-like transform
        w_tensor = torch.softmax(w_tensor / 0.1, dim=0) * w_tensor.size(0)
        return w_tensor.detach()


# ── Meta-Weight-Net ─────────────────────────────────────────────────


class MetaWeightNet(torch.nn.Module):
    """
    Meta-Weight-Net (Shu et al., NeurIPS 2019).

    A small MLP that maps per-sample loss → sample weight.
    Trained via a bi-level meta-learning objective: the weights produced
    should minimise the loss on a clean validation set.

    Architecture:
        input(1) → Linear(1, 16) → BN → ReLU → Linear(16, 1) → Sigmoid * C

    The output is scaled by C so that the maximum possible weight is C.
    This prevents the meta-network from collapsing to zero weights.
    """

    def __init__(self, max_weight: float = 10.0, hidden_dim: int = 16) -> None:
        super().__init__()
        self.max_weight = max_weight
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            losses: (B,) per-sample loss values.

        Returns:
            (B,) sample weights in [0, max_weight].
        """
        B = losses.size(0)
        if B == 0:
            return losses
        w = self.net(losses.unsqueeze(1))
        return w.squeeze(1) * self.max_weight


class MetaWeightNetTrainer:
    """
    Training procedure for Meta-Weight-Net.

    Uses a bi-level optimisation:
        Inner loop:  train model with meta-learned weights.
        Outer loop:  update MetaWeightNet to minimise validation loss.

    Usage:
        mwn = MetaWeightNet(device)
        trainer = MetaWeightNetTrainer(mwn, val_loader, device)
        for batch in train_loader:
            weights = trainer.compute_weights(model, batch, val_batch, criterion)
            ...
    """

    def __init__(
        self,
        meta_weight_net: MetaWeightNet,
        val_loader: DataLoader,
        device: torch.device,
        meta_lr: float = 1e-4,
        meta_weight_decay: float = 0.0,
    ) -> None:
        self.mwn = meta_weight_net
        self.val_loader = val_loader
        self.device = device
        self.meta_lr = meta_lr
        self.meta_weight_decay = meta_weight_decay
        self.meta_optimizer = torch.optim.Adam(
            self.mwn.parameters(), lr=meta_lr, weight_decay=meta_weight_decay,
        )
        self._val_iter: Optional[Any] = None

    def _get_val_batch(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a single validation batch, cycling through the loader."""
        if self._val_iter is None:
            self._val_iter = iter(self.val_loader)
        try:
            batch = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self.val_loader)
            batch = next(self._val_iter)
        inputs = batch[0].to(self.device)
        labels = _move_labels_to_device(batch[1], self.device)
        return inputs, labels

    def compute_weights(
        self,
        model: torch.nn.Module,
        train_per: torch.Tensor,
        train_outputs: Dict[str, torch.Tensor],
        criterion: Any,
        attribute_loss_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Compute Meta-Weight-Net weights and perform one meta-update step.

        Args:
            model:            The model being trained.
            train_per:        (B,) per-sample losses from the training batch.
            train_outputs:    Model outputs on the training batch.
            criterion:        Loss function.
            attribute_loss_weights: Optional per-attribute loss multipliers.

        Returns:
            (B,) weight tensor (detached, for use in the training step.grad).
        """
        # ── Forward through Meta-Weight-Net ──
        weights = self.mwn(train_per.detach())  # (B,)

        # ── Compute weighted training loss ──
        weighted_loss = (weights * train_per.detach()).sum() / (weights.sum() + 1e-8)

        # ── Meta-gradient step ──
        # Compute gradient of weighted loss w.r.t. model parameters
        model_grads = torch.autograd.grad(
            weighted_loss, model.parameters(), retain_graph=True, create_graph=False,
        )

        # Store original parameters
        orig_params = [p.data.clone() for p in model.parameters()]

        # Simulate one step of gradient descent on the model
        lr = 1.0  # virtual learning rate (cancels out in the meta-gradient)
        with torch.no_grad():
            for p, g in zip(model.parameters(), model_grads):
                if g is not None:
                    p.data = p.data - lr * g

        # Forward pass on validation batch with updated parameters
        val_inputs, val_labels = self._get_val_batch()
        val_outputs = model(val_inputs)
        val_loss = _compute_loss(
            val_outputs, val_labels, val_labels, 1.0,
            criterion, attribute_loss_weights,
        )

        # Compute meta-gradient: d(val_loss) / d(MWN_params)
        self.meta_optimizer.zero_grad()
        val_loss.backward()
        self.meta_optimizer.step()

        # Restore original model parameters
        with torch.no_grad():
            for p, orig in zip(model.parameters(), orig_params):
                p.data = orig

        # Recompute weights with updated Meta-Weight-Net
        with torch.no_grad():
            weights = self.mwn(train_per.detach())

        return weights.detach()


# ── Multi-Model Co-Training ─────────────────────────────────────────


class MultiModelCoTrain:
    """
    Generalised N-model co-training for DivideMix-style training.

    Supports an arbitrary number of models (N >= 2).  Each model:
      - Maintains its own loss history and GMM split.
      - Uses the *other* models' clean samples for supervised training
        (sample exchange).
      - For label correction: ensemble of all N models > majority-vote
        confidence threshold.

    This framework subsumes DivideMix (N=2) and Co-teaching (N=2 with
    small-loss selection instead of GMM) and allows scaling beyond 2
    models for additional robustness.

    Args:
        n_samples:          Total number of training samples.
        num_classes_dict:   Dict[attr_name -> num_classes].
        device:             Compute device.
        n_models:           Number of co-training models (default 3).
        warmup_epochs:      Epochs of full supervision before divide phase.
        clean_threshold:    GMM probability threshold for clean split.
        lambda_u:           Weight for unsupervised (noisy) loss.
        noise_rate:         Estimated label noise rate.
        beta_elr:           ELR beta parameter.
        use_gmm:            If True, use GMM for clean/noisy split.
                            If False, use small-loss selection (Co-teaching).
    """

    def __init__(
        self,
        n_samples: int,
        num_classes_dict: Dict[str, int],
        device: torch.device,
        n_models: int = 3,
        warmup_epochs: int = 10,
        clean_threshold: float = 0.5,
        lambda_u: float = 1.0,
        noise_rate: float = 0.2,
        beta_elr: float = 0.7,
        use_gmm: bool = True,
    ) -> None:
        self.n_models = n_models
        self.warmup_epochs = warmup_epochs
        self.clean_threshold = clean_threshold
        self.lambda_u = lambda_u
        self.noise_rate = noise_rate
        self.device = device
        self.use_gmm = use_gmm
        self.attr_names = sorted(num_classes_dict.keys())

        # Per-model loss histories
        self.loss_histories: Dict[str, torch.Tensor] = {}
        self.elr_objs: Dict[str, ELRRegularizer] = {}
        for i in range(n_models):
            key = f"model{i}"
            self.loss_histories[key] = torch.zeros(n_samples, device=device)
            self.elr_objs[key] = ELRRegularizer(
                n_samples, num_classes_dict, beta=beta_elr, device=device,
            )

        self._gmm_cache: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def update_loss_history(
        self,
        model_idx: int,
        indices: torch.Tensor,
        per_sample_loss: torch.Tensor,
        momentum: float = 0.9,
    ) -> None:
        key = f"model{model_idx}"
        idx = indices.to(self.device)
        self.loss_histories[key][idx] = (
            momentum * self.loss_histories[key][idx]
            + (1.0 - momentum) * per_sample_loss.detach()
        )

    def gmm_split(
        self, model_idx: int, current_epoch: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit GMM on model_idx's loss history. Returns (clean_idx, noisy_idx)."""
        from sklearn.mixture import GaussianMixture
        key = f"model{model_idx}"
        losses = self.loss_histories[key].cpu().numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=current_epoch, reg_covar=1e-5)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        if gmm.means_[0] > gmm.means_[1]:
            prob_clean = prob[:, 1]
        else:
            prob_clean = prob[:, 0]
        prob_t = torch.from_numpy(prob_clean).to(self.device)
        all_indices = torch.arange(len(losses), device=self.device)
        clean_idx = all_indices[prob_t > self.clean_threshold]
        noisy_idx = all_indices[prob_t <= self.clean_threshold]
        return clean_idx, noisy_idx

    def small_loss_split(
        self, model_idx: int, epoch: int, num_epochs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Co-teaching-style small-loss selection. Returns (clean_idx, noisy_idx)."""
        rate = 1.0 - self.noise_rate * min(1.0, epoch / max(1, num_epochs))
        n_keep = max(1, int(self.loss_histories[f"model{model_idx}"].size(0) * rate))
        losses = self.loss_histories[f"model{model_idx}"]
        _, keep_idx = torch.topk(losses, k=n_keep, largest=False)
        all_idx = torch.arange(losses.size(0), device=self.device)
        clean_mask = torch.zeros(losses.size(0), dtype=torch.bool, device=self.device)
        clean_mask[keep_idx] = True
        return all_idx[clean_mask], all_idx[~clean_mask]

    def train_epoch(
        self,
        models: Tuple[torch.nn.Module, ...],
        loaders: Tuple[DataLoader, ...],
        optimizers: Tuple[torch.optim.Optimizer, ...],
        criterion: Any,
        device: torch.device,
        epoch: int = 1,
        num_epochs: int = 100,
        text_max_length: int = 256,
        attribute_loss_weights: Optional[Dict[str, float]] = None,
        stop_gradient_heads: Optional[set] = None,
        accuracy_attributes: Optional[set] = None,
        loss_truncation: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Run one multi-model co-training epoch.

        Args:
            models:    Tuple of N models.
            loaders:   Tuple of N DataLoaders (one per model, or same for all).
            optimizers: Tuple of N optimizers.
            epoch:      Current 1-indexed epoch number.
            num_epochs: Total epochs (for rate decay in small-loss selection).
            progress_bar: If True, show a tqdm progress bar.

        Returns:
            (avg_loss, accuracy, per_attr_acc_log)
        """
        N = self.n_models
        for m in models:
            m.train()

        running_loss = 0.0
        total_samples = 0
        correct_total = 0
        sample_total = 0
        per_attr_correct: Dict[str, int] = {}
        per_attr_total: Dict[str, int] = {}

        is_warmup = epoch <= self.warmup_epochs

        # Zip iterators over all model loaders
        loaders_iters = [iter(loader) for loader in loaders]
        pbar = tqdm(desc=f"MCT Epoch {epoch}", leave=False, disable=not progress_bar)

        while True:
            # Fetch one batch per model (or same batch if same loader)
            batch_data: list = []
            for it in loaders_iters:
                try:
                    batch_data.append(next(it))
                except StopIteration:
                    # End of epoch
                    if total_samples == 0:
                        continue
                    break
            if len(batch_data) < N:
                break

            # Use the first model's batch for everything (shared data source)
            batch = batch_data[0]
            indices = batch[-1].to(device)
            batch = batch[:-1]

            inputs = batch[0].to(device)
            labels = _move_labels_to_device(batch[1], device)
            batch_size = inputs.size(0)
            if len(batch) >= 4:
                input_ids = batch[2].to(device)
                attention_mask = batch[3].to(device)
                aux_features = batch[4].to(device) if len(batch) >= 5 else None
            else:
                input_ids = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
                attention_mask = torch.zeros(batch_size, text_max_length, dtype=torch.long, device=device)
                aux_features = batch[2].to(device) if len(batch) >= 3 else None

            # ── Forward all models ──
            all_outputs: list = []
            for m in models:
                out = m(inputs, input_ids, attention_mask, aux_features=aux_features,
                        stop_gradient_heads=stop_gradient_heads)
                all_outputs.append(out)

            # ── Per-sample losses ──
            all_per = []
            for i, out in enumerate(all_outputs):
                per = _compute_per_sample_loss(out, labels, criterion, attribute_loss_weights)
                all_per.append(per)
                self.update_loss_history(i, indices, per)

            # ── Warmup: standard supervised training ──
            if is_warmup:
                batch_loss = 0.0
                for i in range(N):
                    opt = optimizers[i]
                    use_sam = hasattr(opt, 'first_step')
                    opt.zero_grad()
                    loss_i = _compute_loss(all_outputs[i], labels, labels, 1.0,
                                           criterion, attribute_loss_weights,
                                           loss_truncation=loss_truncation)
                    loss_i.backward()
                    if use_sam:
                        opt.first_step(zero_grad=True)
                        out_i_sam = models[i](inputs, input_ids, attention_mask,
                                              aux_features=aux_features,
                                              stop_gradient_heads=stop_gradient_heads)
                        loss_i = _compute_loss(out_i_sam, labels, labels, 1.0,
                                               criterion, attribute_loss_weights,
                                               loss_truncation=loss_truncation)
                        loss_i.backward()
                        opt.second_step(zero_grad=True)
                    else:
                        opt.step()
                    batch_loss += loss_i.item()
                running_loss += (batch_loss / N) * batch_size
                total_samples += batch_size
                c, n = _compute_accuracy(all_outputs[0], labels, accuracy_attributes)
                correct_total += c
                sample_total += n
                for attr_name, attr_acc in _compute_per_attribute_accuracy(all_outputs[0], labels).items():
                    bn = labels[attr_name].size(0) if isinstance(labels, dict) else batch_size
                    per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * bn)
                    per_attr_total[attr_name] = per_attr_total.get(attr_name, 0) + bn
                continue

            # ── Divide phase: GMM or small-loss split ──
            if epoch not in self._gmm_cache:
                self._gmm_cache[epoch] = {}
                for i in range(N):
                    if self.use_gmm:
                        clean_i, noisy_i = self.gmm_split(i, epoch)
                    else:
                        clean_i, noisy_i = self.small_loss_split(i, epoch, num_epochs)
                    self._gmm_cache[epoch][f"clean{i}"] = clean_i
                    self._gmm_cache[epoch][f"noisy{i}"] = noisy_i
                    if verbose:
                        logger.info(
                            "MCT epoch %d model %d: clean=%d noisy=%d",
                            epoch, i, len(clean_i), len(noisy_i),
                        )
            split = self._gmm_cache[epoch]

            # ── For each model: supervised on other models' clean set ──
            def in_split(split_idx: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
                s = set(split_idx.tolist())
                return torch.tensor([i.item() in s for i in batch_idx], device=device)

            batch_loss = 0.0
            for i in range(N):
                # This model uses clean samples from ALL OTHER models
                other_clean_mask_list = []
                other_noisy_mask_list = []
                for j in range(N):
                    if j == i:
                        continue
                    clean_j = split[f"clean{j}"]
                    noisy_j = split[f"noisy{j}"]
                    other_clean_mask_list.append(in_split(clean_j, indices))
                    other_noisy_mask_list.append(in_split(noisy_j, indices))

                # Union of other models' clean masks
                clean_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                noisy_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                for cm in other_clean_mask_list:
                    clean_mask = clean_mask | cm
                for nm in other_noisy_mask_list:
                    noisy_mask = noisy_mask | nm
                # A sample can't be both clean and noisy; clean takes precedence
                noisy_mask = noisy_mask & ~clean_mask

                # Supervised loss on clean samples (using the union of others' clean)
                sup = torch.tensor(0.0, device=device)
                if clean_mask.any():
                    out_clean = {k: v[clean_mask] for k, v in all_outputs[i].items()}
                    labels_clean = {k: v[clean_mask] for k, v in labels.items()}
                    sup = _compute_loss(out_clean, labels_clean, labels_clean, 1.0,
                                        criterion, attribute_loss_weights,
                                        loss_truncation=loss_truncation)

                # Consistency loss on noisy samples (use ensemble of others)
                unsup = torch.tensor(0.0, device=device)
                if noisy_mask.any():
                    out_noisy = {k: v[noisy_mask] for k, v in all_outputs[i].items()}
                    # Ensemble of other models' predictions on these samples
                    ensemble_probs = None
                    n_others = 0
                    for j in range(N):
                        if j == i:
                            continue
                        with torch.no_grad():
                            probs_j = torch.softmax(all_outputs[j][noisy_mask], dim=-1)
                        if ensemble_probs is None:
                            ensemble_probs = probs_j
                        else:
                            ensemble_probs = ensemble_probs + probs_j
                        n_others += 1
                    if ensemble_probs is not None:
                        ensemble_probs = ensemble_probs / n_others
                        for attr in out_noisy:
                            if attr not in ensemble_probs:
                                continue
                            w = (attribute_loss_weights or {}).get(attr, 1.0)
                            unsup = unsup + w * torch.mean(
                                torch.sum(
                                    -ensemble_probs[attr].detach()
                                    * torch.log_softmax(out_noisy[attr], dim=-1),
                                    dim=-1,
                                )
                            )

                # ELR on noisy samples
                elr_val = torch.tensor(0.0, device=device)
                if noisy_mask.any():
                    out_noisy_elr = {k: v[noisy_mask] for k, v in all_outputs[i].items()}
                    elr_val = self.elr_objs[f"model{i}"](
                        out_noisy_elr, indices[noisy_mask], epoch,
                    )

                total_loss_i = sup + self.lambda_u * unsup + elr_val

                opt = optimizers[i]
                use_sam = hasattr(opt, 'first_step')
                opt.zero_grad()
                total_loss_i.backward()
                torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm=1.0)
                if use_sam:
                    opt.first_step(zero_grad=True)
                    out_i_sam = models[i](inputs, input_ids, attention_mask,
                                          aux_features=aux_features,
                                          stop_gradient_heads=stop_gradient_heads)
                    sup_sam = torch.tensor(0.0, device=device)
                    if clean_mask.any():
                        out_sam_clean = {k: v[clean_mask] for k, v in out_i_sam.items()}
                        sup_sam = _compute_loss(out_sam_clean, labels_clean, labels_clean, 1.0,
                                                criterion, attribute_loss_weights,
                                                loss_truncation=loss_truncation)
                    unsup_sam = torch.tensor(0.0, device=device)
                    if noisy_mask.any():
                        out_sam_noisy = {k: v[noisy_mask] for k, v in out_i_sam.items()}
                        for attr in out_sam_noisy:
                            if attr not in ensemble_probs:
                                continue
                            w = (attribute_loss_weights or {}).get(attr, 1.0)
                            unsup_sam = unsup_sam + w * torch.mean(
                                torch.sum(
                                    -ensemble_probs[attr].detach()
                                    * torch.log_softmax(out_sam_noisy[attr], dim=-1),
                                    dim=-1,
                                )
                            )
                    elr_val_sam = torch.tensor(0.0, device=device)
                    if noisy_mask.any():
                        out_sam_noisy_elr = {k: v[noisy_mask] for k, v in out_i_sam.items()}
                        elr_val_sam = self.elr_objs[f"model{i}"](
                            out_sam_noisy_elr, indices[noisy_mask], epoch,
                        )
                    total_sam = sup_sam + self.lambda_u * unsup_sam + elr_val_sam
                    total_sam.backward()
                    opt.second_step(zero_grad=True)
                else:
                    opt.step()

                batch_loss += total_loss_i.item()

            running_loss += (batch_loss / N) * batch_size
            total_samples += batch_size
            c, n = _compute_accuracy(all_outputs[0], labels, accuracy_attributes)
            correct_total += c
            sample_total += n
            for attr_name, attr_acc in _compute_per_attribute_accuracy(all_outputs[0], labels).items():
                bn = labels[attr_name].size(0) if isinstance(labels, dict) else batch_size
                per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * bn)
                per_attr_total[attr_name] = per_attr_total.get(attr_name, 0) + bn

            if progress_bar:
                pbar.update(1)
                pbar.set_postfix(loss=running_loss / total_samples if total_samples > 0 else 0.0,
                                 acc=correct_total / sample_total if sample_total > 0 else 0.0)

        if progress_bar:
            pbar.close()

        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct_total / sample_total if sample_total > 0 else 0.0
        per_attr_acc_log = {
            attr: round(per_attr_correct[attr] / per_attr_total[attr], 4)
            for attr in per_attr_correct if per_attr_total.get(attr, 0) > 0
        }
        logger.info("MultiModelCoTrain epoch %d: loss=%.4f, accuracy=%.4f%s",
                    epoch, avg_loss, accuracy,
                    " (warmup)" if is_warmup else "")
        if per_attr_acc_log:
            logger.info("MCT per-attribute accuracy: %s", per_attr_acc_log)
        return avg_loss, accuracy, per_attr_acc_log
