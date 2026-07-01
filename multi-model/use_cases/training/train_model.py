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
) -> torch.Tensor:
    if isinstance(outputs, dict) and isinstance(labels_a, dict):
        total = torch.tensor(0.0, device=next(iter(labels_a.values())).device)
        for attr in outputs:
            if attr not in labels_a:
                continue
            weight = (attribute_loss_weights or {}).get(attr, 1.0)
            attr_crit = _get_criterion(criterion, attr)
            if loss_truncation is not None and attr in loss_truncation:
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
            if loss_truncation is not None and attr in loss_truncation:
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
        # Check if batch includes sample indices (last element, for ELR)
        has_indices = elr is not None and isinstance(batch[-1], torch.Tensor) and batch[-1].dim() == 1
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
        loss = _compute_loss(outputs, labels_a, labels_b, lam, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
        # Add ELR regularization if enabled
        if elr is not None and indices is not None:
            loss = loss + elr(outputs, indices, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if use_sam:
            optimizer.first_step(zero_grad=True)
            # Second forward pass at perturbed weights
            outputs = model(inputs, input_ids, attention_mask, aux_features=aux_features, stop_gradient_heads=stop_gradient_heads)
            loss = _compute_loss(outputs, labels_a, labels_b, lam, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
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
        # CrossEntropyLoss reduction='none' gives per-sample loss
        if isinstance(attr_crit, torch.nn.CrossEntropyLoss):
            ce = torch.nn.functional.cross_entropy(logits, labels[attr_name], reduction='none')
        else:
            ce = attr_crit(logits, labels[attr_name])
            if ce.dim() == 0:
                ce = ce.expand_as(per_sample)
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

    for batch in loader:
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

        for batch in loader:
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
                out1 = model1(inputs, input_ids, attention_mask, aux_features=aux_features,
                              stop_gradient_heads=stop_gradient_heads)
                out2 = model2(inputs, input_ids, attention_mask, aux_features=aux_features,
                              stop_gradient_heads=stop_gradient_heads)

                loss1 = _compute_loss(out1, labels, labels, 1.0, criterion, attribute_loss_weights, loss_truncation=loss_truncation)
                loss2 = _compute_loss(out2, labels, labels, 1.0, criterion, attribute_loss_weights, loss_truncation=loss_truncation)

                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                loss2.backward()
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
                continue

            # ── Divide phase: forward pass both models ──
            out1 = model1(inputs, input_ids, attention_mask, aux_features=aux_features,
                          stop_gradient_heads=stop_gradient_heads)
            out2 = model2(inputs, input_ids, attention_mask, aux_features=aux_features,
                          stop_gradient_heads=stop_gradient_heads)

            # Per-sample losses updated from OTHER model's output
            # (cross-update for GMM consistency)
            loss1_per = _compute_per_sample_loss(out2, labels, criterion, attribute_loss_weights)
            loss2_per = _compute_per_sample_loss(out1, labels, criterion, attribute_loss_weights)
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

            # ── Supervised loss on clean samples ──
            loss_sup1 = torch.tensor(0.0, device=device)
            loss_sup2 = torch.tensor(0.0, device=device)
            if clean_mask1.any():
                out1_clean = {k: v[clean_mask1] for k, v in out1.items()}
                labels_clean = {k: v[clean_mask1] for k, v in labels.items()}
                loss_sup1 = _compute_loss(out1_clean, labels_clean, labels_clean, 1.0,
                                          criterion, attribute_loss_weights, loss_truncation=loss_truncation)
            if clean_mask2.any():
                out2_clean = {k: v[clean_mask2] for k, v in out2.items()}
                labels_clean2 = {k: v[clean_mask2] for k, v in labels.items()}
                loss_sup2 = _compute_loss(out2_clean, labels_clean2, labels_clean2, 1.0,
                                          criterion, attribute_loss_weights, loss_truncation=loss_truncation)

            # ── Consistency loss on noisy samples ──
            # Model1: minimise cross-entropy against Model2's prediction (soft pseudo-label)
            # Model2: minimise cross-entropy against Model1's prediction
            loss_unsup1 = torch.tensor(0.0, device=device)
            loss_unsup2 = torch.tensor(0.0, device=device)
            if noisy_mask1.any():
                out1_noisy = {k: v[noisy_mask1] for k, v in out1.items()}
                out2_for1 = {k: v[noisy_mask1] for k, v in out2.detach().items()}
                for attr in out1_noisy:
                    if attr not in out2_for1:
                        continue
                    p = torch.softmax(out2_for1[attr], dim=-1)
                    w = (attribute_loss_weights or {}).get(attr, 1.0)
                    loss_unsup1 = loss_unsup1 + w * torch.mean(
                        torch.sum(-p * torch.log_softmax(out1_noisy[attr], dim=-1), dim=-1)
                    )
            if noisy_mask2.any():
                out2_noisy = {k: v[noisy_mask2] for k, v in out2.items()}
                out1_for2 = {k: v[noisy_mask2] for k, v in out1.detach().items()}
                for attr in out2_noisy:
                    if attr not in out1_for2:
                        continue
                    p = torch.softmax(out1_for2[attr], dim=-1)
                    w = (attribute_loss_weights or {}).get(attr, 1.0)
                    loss_unsup2 = loss_unsup2 + w * torch.mean(
                        torch.sum(-p * torch.log_softmax(out2_noisy[attr], dim=-1), dim=-1)
                    )

            # ── ELR regularisation on noisy samples ──
            if noisy_mask1.any():
                out1_noisy = {k: v[noisy_mask1] for k, v in out1.items()}
                loss_elr1 = self.elr1(out1_noisy, indices[noisy_mask1], epoch)
            else:
                loss_elr1 = torch.tensor(0.0, device=device)
            if noisy_mask2.any():
                out2_noisy = {k: v[noisy_mask2] for k, v in out2.items()}
                loss_elr2 = self.elr2(out2_noisy, indices[noisy_mask2], epoch)
            else:
                loss_elr2 = torch.tensor(0.0, device=device)

            # ── Combined loss ──
            total_loss1 = loss_sup1 + self.lambda_u * loss_unsup1 + loss_elr1
            total_loss2 = loss_sup2 + self.lambda_u * loss_unsup2 + loss_elr2

            # ── Backward ──
            optimizer1.zero_grad()
            total_loss1.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            optimizer1.step()

            optimizer2.zero_grad()
            total_loss2.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
            optimizer2.step()

            running_loss += 0.5 * (total_loss1.item() + total_loss2.item()) * batch_size
            total_samples += batch_size

            # Accuracy from model1
            c, n = _compute_accuracy(out1, labels, accuracy_attributes)
            correct_total += c
            sample_total += n
            for attr_name, attr_acc in _compute_per_attribute_accuracy(out1, labels).items():
                bn = labels[attr_name].size(0) if isinstance(labels, dict) else batch_size
                per_attr_correct[attr_name] = per_attr_correct.get(attr_name, 0) + int(attr_acc * bn)
                per_attr_total[attr_name] = per_attr_total.get(attr_name, 0) + bn

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
