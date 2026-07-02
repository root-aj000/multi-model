"""
Training pipeline builder.

BUG-02 FIX: AdamW is called without a global weight_decay — each param group
            already has weight_decay set explicitly, so the global arg was
            redundant and could silently override no-decay groups.
BUG-07 FIX: train DataLoader now uses drop_last=True to prevent single-sample
            batches that crash BatchNorm layers.
BUG-12 FIX: _get_image_size logs a warning when the config value is invalid
            instead of silently falling back.
BUG-17 FIX: tokenizer_fn closure no longer has a redundant default argument.
BUG-25 FIX: build_training_pipeline now auto-computes inverse-frequency class
            weights from the training CSV when CLASS_WEIGHTS is absent from
            config.  Previously the model always trained without any weighting,
            causing it to over-predict majority classes.
BUG-26 FIX: StepLR constants are clearly documented as non-default scheduler
            options, not dead code.
BUG-28 FIX: Documented that CLASS_WEIGHTS and training hyperparameters must
            be at the top level of config, not nested under "model".
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from lib.models.factory import create_model
from lib.models.fg_mfn import ATTRIBUTE_NAMES
from lib.preprocessing.dataset import (
    AUX_FEATURE_COLUMNS, AUX_FEATURE_DIM, CustomDataset, load_dataset,
)
from lib.preprocessing.image.transforms import DEFAULT_IMAGE_SIZE, build_image_transform
from lib.preprocessing.text.cleaner import clean_text, clean_adcopy
from lib.preprocessing.text.pipeline import build_text_pipeline
from lib.preprocessing.text.tokenizer import load_tokenizer, tokenize_text
from lib.utils.class_weights import (
    compute_class_weights_from_csv,
    compute_cooccurrence_prior,
    log_class_distribution,
)
from lib.utils.config import get_dataset_paths, get_label_maps, load_config
from lib.utils.losses import FocalLoss, GCELoss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback defaults — all of these are defined in model_config.json.
# These values are only used if the key is somehow absent from config.
# ---------------------------------------------------------------------------

_FALLBACK_BATCH_SIZE = 32
_FALLBACK_NUM_WORKERS = min(8, os.cpu_count() or 1)
_FALLBACK_LEARNING_RATE = 5e-4
_FALLBACK_ENCODER_LEARNING_RATE = 2e-5
_FALLBACK_SCHEDULER_STEP_SIZE = 30
_FALLBACK_SCHEDULER_GAMMA = 0.1


def _get_image_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    Read image_size from config and return as a (width, height) tuple.

    BUG-12 FIX: Logs a warning when the value is invalid instead of
    silently falling back, so misconfigured runs are caught early.

    Accepts:
        [w, h]  — list or tuple of two ints
        N       — single int (square image)
        missing — falls back to DEFAULT_IMAGE_SIZE with a warning

    Args:
        config: Configuration dictionary.

    Returns:
        (width, height) tuple.
    """
    raw = config.get("image_size", DEFAULT_IMAGE_SIZE)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    if isinstance(raw, int):
        return (raw, raw)
    logger.warning(
        "Invalid image_size value %r in config; falling back to %s. "
        "Expected [width, height] or a single int.",
        raw, DEFAULT_IMAGE_SIZE,
    )
    return DEFAULT_IMAGE_SIZE


def _build_text_pipeline(config: Dict[str, Any]) -> Optional[Any]:
    """
    Build a text preprocessing pipeline if text features are enabled.

    BUG-17 FIX: tokenizer_fn no longer has a redundant default argument
    that shadowed the closure variable.

    Args:
        config: Configuration dictionary.

    Returns:
        Callable text pipeline or None if use_text is False.
    """
    if not config.get("use_text", False):
        return None

    tokenizer = load_tokenizer(config.get("TEXT_ENCODER", "distilbert-base-uncased"))
    max_length = config.get("text_max_length", 256)

    # Use the light ad-copy cleaner for CSV text columns (clean ad copy /
    # headlines / slogans). The aggressive OCR cleaner strips apostrophes
    # and most punctuation — fine for noisy OCR output, but lossy for
    # structured ad copy where "don't miss out!" should stay intact.
    cleaner = clean_adcopy

    def tokenizer_fn(text: str, max_length_arg: int = None) -> dict:
        actual_max_length = max_length_arg if max_length_arg is not None else max_length
        return tokenize_text(text, max_length=actual_max_length, tokenizer=tokenizer)

    return build_text_pipeline(cleaner, tokenizer_fn)


def load_datasets(
    config: Dict[str, Any],
    aux_feature_columns: Optional[List[str]] = None,
) -> Tuple[CustomDataset, CustomDataset]:
    """
    Load and return the train and validation datasets.

    Args:
        config: Configuration dictionary with dataset paths and ATTRIBUTES.
        aux_feature_columns: Optional list of CSV column names for handcrafted
            auxiliary feature extraction.

    Returns:
        (train_dataset, val_dataset)
    """
    label_maps = get_label_maps(config)
    image_size = _get_image_size(config)
    aug_cfg = config.get("augmentation", None)

    train_image_pipeline = build_image_transform(
        size=image_size, augment=True, aug_cfg=aug_cfg,
    )
    val_image_pipeline = build_image_transform(
        size=image_size, augment=False,
    )
    text_pipeline = _build_text_pipeline(config)

    dataset_root = config.get("DATASET_ROOT") or config.get("dataset_root")

    train_dataset = load_dataset(
        "train",
        label_maps=label_maps,
        image_pipeline=train_image_pipeline,
        text_pipeline=text_pipeline,
        dataset_root=dataset_root,
        aux_feature_columns=aux_feature_columns,
    )
    val_dataset = load_dataset(
        "val",
        label_maps=label_maps,
        image_pipeline=val_image_pipeline,
        text_pipeline=text_pipeline,
        dataset_root=dataset_root,
        aux_feature_columns=aux_feature_columns,
    )
    logger.info(
        "Datasets loaded: train=%d, val=%d (image_size=%s)",
        len(train_dataset), len(val_dataset), image_size,
    )
    return train_dataset, val_dataset


def create_data_loaders(
    train_dataset: Any,
    val_dataset: Any,
    batch_size: int = _FALLBACK_BATCH_SIZE,
    num_workers: int = _FALLBACK_NUM_WORKERS,
    sampler: Optional[WeightedRandomSampler] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation datasets.

    BUG-07 FIX: train_loader uses drop_last=True to prevent single-sample
    batches that crash ResNet's BatchNorm layers when the dataset size is
    not evenly divisible by batch_size.

    Args:
        train_dataset: Training dataset.
        val_dataset:   Validation dataset.
        batch_size:    Samples per batch.
        num_workers:   DataLoader worker processes.
        sampler:       Optional WeightedRandomSampler for class rebalancing.

    Returns:
        (train_loader, val_loader)
    """
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,   # BUG-07 FIX: prevents batch-size-1 BatchNorm crash
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        # drop_last=False for val so every sample is evaluated
    )
    return train_loader, val_loader


def _build_param_groups(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build parameter groups with per-group learning rates and weight_decay.

    The pretrained text encoder is fine-tuned at encoder_learning_rate
    (much lower than learning_rate for randomly initialised layers).
    Bias terms and normalization weights are excluded from weight decay
    following the standard BERT fine-tuning recipe.

    BUG-02 FIX: Each param group already has weight_decay set explicitly.

    DataParallel note: when the model is wrapped with nn.DataParallel,
    named_parameters() adds a "module." prefix to all names. This function
    strips that prefix before matching so encoder detection works correctly
    regardless of whether DataParallel is active.

    Args:
        model: The FG_MFN model.
        cfg:   Configuration dict with learning rate settings.

    Returns:
        List of parameter group dicts ready for an optimizer.
    """
    lr = cfg.get("learning_rate", _FALLBACK_LEARNING_RATE)
    encoder_lr = cfg.get("encoder_learning_rate", _FALLBACK_ENCODER_LEARNING_RATE)
    weight_decay = cfg.get("weight_decay", 1e-2)

    no_decay_suffixes = ("bias", "LayerNorm.weight", "layer_norm.weight")

    encoder_decay: List[torch.nn.Parameter] = []
    encoder_no_decay: List[torch.nn.Parameter] = []
    other_decay: List[torch.nn.Parameter] = []
    other_no_decay: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        bare_name = name[len("module."):] if name.startswith("module.") else name
        is_encoder = bare_name.startswith("text_module.encoder")
        no_decay = any(bare_name.endswith(sfx) for sfx in no_decay_suffixes)

        if is_encoder:
            (encoder_no_decay if no_decay else encoder_decay).append(param)
        else:
            (other_no_decay if no_decay else other_decay).append(param)

    param_groups = [
        {"params": other_decay,    "lr": lr,         "weight_decay": weight_decay},
        {"params": other_no_decay, "lr": lr,         "weight_decay": 0.0},
    ]
    if encoder_decay or encoder_no_decay:
        param_groups += [
            {"params": encoder_decay,    "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
        ]
        logger.info(
            "Param groups: encoder (decay=%d, no_decay=%d) @ lr=%.2e; "
            "other (decay=%d, no_decay=%d) @ lr=%.2e",
            len(encoder_decay), len(encoder_no_decay), encoder_lr,
            len(other_decay), len(other_no_decay), lr,
        )
    else:
        logger.info(
            "Param groups: %d params @ lr=%.2e (no encoder params found)",
            len(other_decay) + len(other_no_decay), lr,
        )

    return param_groups


def _build_optimizer(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
) -> AdamW:
    """Build an AdamW optimizer from model and config parameter groups."""
    param_groups = _build_param_groups(model, cfg)
    return AdamW(param_groups)


# ──────────────────────────────────────────────────────────────────────────────
# SAM (Sharpness-Aware Minimization) Optimizer Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SAM(torch.optim.Optimizer):
    """
    Wraps a base optimizer with Sharpness-Aware Minimization.

    SAM finds flatter minima by first taking a gradient ascent step
    (perturbing weights in the direction of sharpest loss increase),
    then computing the gradient at that perturbed point, and finally
    applying the base optimizer's update from the original position.

    Usage in training loop:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss = criterion(model(x), y)
        loss.backward()
        optimizer.second_step(zero_grad=True)

    Reference:
        Foret et al., "Sharpness-Aware Minimization for Efficiently
        Improving Generalization" (ICLR 2021)
    """

    def __init__(
        self,
        params: Any,
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            params:           Model parameters or parameter groups.
            base_optimizer:   Optimizer class (e.g. AdamW, SGD).
            rho:              Perturbation radius (neighborhood size).
            adaptive:         If True, use ASAM (adaptive SAM) where the
                              perturbation is scaled by |w| for each weight.
            kwargs:           Arguments passed to the base optimizer.
        """
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Perturb weights along the gradient direction (ascent step)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale
                if group["adaptive"]:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore original weights and apply base optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        """Compute the L2 norm of all gradients combined."""
        norm = torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norm += p.grad.norm() ** 2
        return norm ** 0.5

    def __repr__(self) -> str:
        return (
            f"SAM(base={self.base_optimizer.__class__.__name__}, "
            f"rho={self.param_groups[0]['rho']}, "
            f"adaptive={self.param_groups[0]['adaptive']})"
        )


def setup_training_components(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Set up optimizer, scheduler, criterion, and mixup_alpha.

    IMPORTANT (BUG-28 NOTE): Training hyperparameters — learning_rate,
    encoder_learning_rate, weight_decay, CLASS_WEIGHTS, epochs,
    warmup_epochs, mixup_alpha, scheduler_type — must be at the TOP LEVEL
    of the config dict, NOT nested under a "model" key. The model
    architecture keys (IMAGE_BACKBONE, TEXT_ENCODER, etc.) may be nested
    under "model", but training keys must not be.

    scheduler_t_max is auto-computed as (epochs - warmup_epochs) so it
    always covers the post-warmup training period correctly.

    Args:
        model:  The model to train.
        cfg:    Top-level configuration dict (not model_cfg).
        device: Compute device.

    Returns:
        Dict with keys: optimizer, scheduler, criterion, mixup_alpha.
    """
    param_groups = _build_param_groups(model, cfg)

    use_sam = cfg.get("use_sam", False)
    if use_sam:
        sam_rho = cfg.get("sam_rho", 0.05)
        optimizer = SAM(
            param_groups, AdamW,
            rho=sam_rho,
            lr=cfg.get("learning_rate", _FALLBACK_LEARNING_RATE),
            weight_decay=cfg.get("weight_decay", 1e-2),
        )
        logger.info("SAM optimizer enabled (rho=%.4f)", sam_rho)
    else:
        optimizer = AdamW(param_groups)

    epochs = cfg.get("epochs", 100)
    warmup_epochs = cfg.get("warmup_epochs", 0) or 0

    scheduler_type = cfg.get("scheduler_type", "cosine")
    if scheduler_type == "cosine":
        t_max = cfg.get("scheduler_t_max", max(1, epochs - warmup_epochs))
        eta_min = cfg.get("scheduler_eta_min", 1e-6)
        base_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        logger.info("CosineAnnealingLR: T_max=%d, eta_min=%.2e", t_max, eta_min)
    else:
        # StepLR — set scheduler_type = "step" in config to use this
        step_size = cfg.get("scheduler_step_size", _FALLBACK_SCHEDULER_STEP_SIZE)
        gamma = cfg.get("scheduler_gamma", _FALLBACK_SCHEDULER_GAMMA)
        base_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info("StepLR: step_size=%d, gamma=%.2f", step_size, gamma)

    if warmup_epochs > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=1e-5,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                base_scheduler,
            ],
            milestones=[warmup_epochs],
        )
        logger.info("Linear warmup for %d epochs, then %s", warmup_epochs, scheduler_type)
    else:
        scheduler = base_scheduler

    # Build criterion — supports per-attribute class weights, focal loss,
    # and per-attribute label smoothing.
    # CLASS_WEIGHTS must be at the top level of config (not under "model").
    class_weights = cfg.get("CLASS_WEIGHTS", {})
    label_smoothing_raw = cfg.get("label_smoothing", 0.0)
    # Allow label_smoothing to be a float (global) or a dict (per-attribute)
    global_label_smoothing = label_smoothing_raw if isinstance(label_smoothing_raw, (int, float)) else 0.0
    per_attr_label_smoothing = label_smoothing_raw if isinstance(label_smoothing_raw, dict) else {}
    focal_gamma = cfg.get("FOCAL_LOSS_GAMMA", {})
    if isinstance(focal_gamma, (int, float)):
        focal_gamma = {name: focal_gamma for name in class_weights}
    gce_q = cfg.get("GCE_Q", {})
    if isinstance(gce_q, (int, float)):
        gce_q = {name: gce_q for name in class_weights}
    # Loss selection priority per attribute: FocalLoss > GCELoss > CrossEntropyLoss

    if isinstance(class_weights, dict) and class_weights:
        criterion: Any = {}
        for attr_name, weights in class_weights.items():
            if not isinstance(weights, list):
                continue
            weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
            smoothing = per_attr_label_smoothing.get(attr_name, global_label_smoothing)
            gamma = focal_gamma.get(attr_name, 0.0)
            q = gce_q.get(attr_name, 0.0)
            if gamma > 0:
                criterion[attr_name] = FocalLoss(
                    gamma=gamma, weight=weight_t, label_smoothing=smoothing,
                )
            elif q > 0:
                criterion[attr_name] = GCELoss(
                    q=q, weight=weight_t, label_smoothing=smoothing,
                )
            else:
                criterion[attr_name] = torch.nn.CrossEntropyLoss(
                    weight=weight_t, label_smoothing=smoothing,
                )
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=global_label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=global_label_smoothing)

    mixup_alpha = cfg.get("mixup_alpha", 0.0)

    logger.info(
        "Training components: lr=%.2e, encoder_lr=%.2e, wd=%.2e, "
        "scheduler=%s, warmup=%d, mixup_alpha=%.2f",
        cfg.get("learning_rate", _FALLBACK_LEARNING_RATE),
        cfg.get("encoder_learning_rate", _FALLBACK_ENCODER_LEARNING_RATE),
        cfg.get("weight_decay", 1e-2),
        scheduler_type, warmup_epochs, mixup_alpha,
    )

    stop_gradient_heads = set(cfg.get("STOP_GRADIENT_HEADS", []))
    if stop_gradient_heads:
        logger.info("Stop-gradient heads: %s", sorted(stop_gradient_heads))

    # Build loss truncation thresholds for noisy attributes.
    # Prevents individual samples with extremely wrong labels from
    # producing unbounded gradients.
    noisy_attrs_trunc = cfg.get("NOISY_ATTRIBUTES", [])
    trunc_value = cfg.get("LOSS_TRUNCATION", 3.0)
    loss_truncation = {attr: trunc_value for attr in noisy_attrs_trunc} if noisy_attrs_trunc else None
    if loss_truncation:
        logger.info("Loss truncation enabled: %s", loss_truncation)

    return {
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "mixup_alpha": mixup_alpha,
        "loss_truncation": loss_truncation,
    }


def build_training_pipeline(config_source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build and return the complete training pipeline.

    Loads config, creates datasets and data loaders, initialises the model,
    and sets up training components.

    BUG-25 FIX: If CLASS_WEIGHTS is absent from config, class weights are
    computed automatically from the training CSV using inverse-frequency
    weighting.  This prevents the model from over-predicting majority classes
    on imbalanced datasets.  Set CLASS_WEIGHTS explicitly in config to
    override the auto-computed values, or set CLASS_WEIGHTS to {} to
    disable weighting entirely.

    Args:
        config_source: Path to a JSON config file, or a pre-loaded config dict.

    Returns:
        Dict with keys: model, train_loader, val_loader, optimizer,
        criterion, scheduler, mixup_alpha.
    """
    if isinstance(config_source, str):
        config = load_config(config_source)
    else:
        config = config_source

    # Model architecture keys may be nested under "model" or at the top level.
    # Training hyperparameters (lr, CLASS_WEIGHTS, etc.) must be at top level.
    model_cfg = config.get("model", config)

    # If USE_GRAPH_HEADS is enabled, compute label co-occurrence prior from the
    # training CSV and inject it into model_cfg so the GNN can use it as an
    # attention bias. This is done before model creation.
    if model_cfg.get("USE_GRAPH_HEADS", False):
        dataset_root_cooc = config.get("DATASET_ROOT") or config.get("dataset_root")
        train_paths_cooc = get_dataset_paths("train", dataset_root_cooc)
        train_csv_cooc = train_paths_cooc["csv"]
        attr_names_cooc = [
            n for n in ATTRIBUTE_NAMES
            if n in model_cfg.get("ATTRIBUTES", {})
        ]
        logger.info(
            "USE_GRAPH_HEADS enabled — computing co-occurrence prior from %s",
            train_csv_cooc,
        )
        cooc_prior = compute_cooccurrence_prior(train_csv_cooc, attr_names_cooc)
        model_cfg = {**model_cfg, "COOCCURRENCE_PRIOR": cooc_prior.tolist()}
        logger.info(
            "Co-occurrence prior computed for %d attributes (shape: %s)",
            len(attr_names_cooc), list(cooc_prior.shape),
        )
        # Log NMI values for each pair so we can see which attributes share signal
        for i, ai in enumerate(attr_names_cooc):
            for j, aj in enumerate(attr_names_cooc):
                if i < j:
                    nmi_val = cooc_prior[i, j].item()
                    if nmi_val > 0.05:
                        logger.info("  NMI %s ↔ %s: %.3f", ai, aj, nmi_val)

    # Enable handcrafted auxiliary features from CSV metadata columns
    use_aux_features = config.get("USE_AUX_FEATURES", False)
    aux_feature_columns = AUX_FEATURE_COLUMNS if use_aux_features else None
    if use_aux_features:
        model_cfg = {**model_cfg, "AUX_FEATURE_DIM": AUX_FEATURE_DIM}
        logger.info(
            "Handcrafted aux features enabled: %s (dim=%d)",
            AUX_FEATURE_COLUMNS, AUX_FEATURE_DIM,
        )

    train_dataset, val_dataset = load_datasets(config, aux_feature_columns)
    batch_size = config.get("batch_size", _FALLBACK_BATCH_SIZE)
    num_workers = min(
        config.get("num_workers", _FALLBACK_NUM_WORKERS),
        os.cpu_count() or 1,
    )

    # WeightedRandomSampler — oversamples minority classes for noisy attributes
    noisy_attrs = set(config.get("NOISY_ATTRIBUTES", []))
    sampler = None
    if noisy_attrs and "CLASS_WEIGHTS" in config:
        cw = config["CLASS_WEIGHTS"]
        if isinstance(cw, dict) and cw:
            # Per-sample weight = max inverse-frequency weight across noisy attrs
            sample_weights = []
            train_df = train_dataset.data
            for _, row in train_df.iterrows():
                max_w = 0.0
                for attr in noisy_attrs:
                    if attr in cw and isinstance(cw[attr], list):
                        label_val = row.get(attr)
                        if attr in train_dataset._label_indices and label_val in train_dataset._label_indices[attr]:
                            cls_idx = train_dataset._label_indices[attr][label_val]
                            if cls_idx < len(cw[attr]):
                                max_w = max(max_w, cw[attr][cls_idx])
                sample_weights.append(max(max_w, 1.0))
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            logger.info(
                "WeightedRandomSampler: %d samples, %.2f avg weight (noisy=%s)",
                len(sample_weights), sum(sample_weights) / len(sample_weights),
                sorted(noisy_attrs),
            )

    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
    )

    # BUG-25 FIX: auto-compute class weights from the training CSV when
    # CLASS_WEIGHTS is not set in config.  The caller can still override by
    # setting CLASS_WEIGHTS explicitly (including {} to disable).
    if "CLASS_WEIGHTS" not in config:
        label_maps = get_label_maps(config)
        dataset_root = config.get("DATASET_ROOT") or config.get("dataset_root")
        train_paths = get_dataset_paths("train", dataset_root)
        train_csv = train_paths["csv"]
        logger.info(
            "CLASS_WEIGHTS not set in config — computing from training CSV: %s",
            train_csv,
        )
        log_class_distribution(train_csv, label_maps)
        auto_weights = compute_class_weights_from_csv(train_csv, label_maps)
        if auto_weights:
            config = {**config, "CLASS_WEIGHTS": auto_weights}
            logger.info(
                "Auto-computed CLASS_WEIGHTS for %d attribute(s): %s",
                len(auto_weights),
                list(auto_weights.keys()),
            )
        else:
            logger.warning(
                "Could not compute class weights — no matching columns found in CSV. "
                "Training without class weights (BUG-25 active)."
            )
    else:
        logger.info("Using CLASS_WEIGHTS from config.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(model_cfg, device)

    # Pass the full config (not model_cfg) so training hyperparameters
    # at the top level are accessible (BUG-28 NOTE).
    components = setup_training_components(model, config, device)

    accuracy_attrs_list = config.get("ACCURACY_ATTRIBUTES", [])
    accuracy_attributes = set(accuracy_attrs_list) if accuracy_attrs_list else None
    if accuracy_attributes:
        logger.info("Accuracy attributes: %s", sorted(accuracy_attributes))

    use_tta = config.get("USE_TTA", False)
    logger.info("TTA: %s", "enabled (horizontal flip)" if use_tta else "disabled")

    # ── ELR setup ──
    elr_beta = config.get("ELR_BETA", 0.7)
    use_elr = elr_beta > 0 and not config.get("USE_DIVIDEMIX", False)
    elr = None
    if use_elr and elr_beta > 0:
        from use_cases.training.train_model import ELRRegularizer
        num_classes_dict = {
            name: attr_cfg["num_classes"]
            for name, attr_cfg in config.get("ATTRIBUTES", {}).items()
        }
        elr = ELRRegularizer(
            n_samples=len(train_dataset),
            num_classes_dict=num_classes_dict,
            beta=elr_beta,
            device=device,
        )
        logger.info("ELR regularizer created: beta=%.2f, %d samples", elr_beta, len(train_dataset))

    # ── Co-teaching / DivideMix dual-model setup ──
    use_coteaching = config.get("USE_COTEACHING", False)
    use_dividemix = config.get("USE_DIVIDEMIX", False)
    model2 = None
    optimizer2 = None
    divide_mix = None

    if use_dividemix:
        from use_cases.training.train_model import DivideMix
        model2 = create_model(model_cfg, device)
        num_classes_dict = {
            name: attr_cfg["num_classes"]
            for name, attr_cfg in config.get("ATTRIBUTES", {}).items()
        }
        divide_mix = DivideMix(
            n_samples=len(train_dataset),
            num_classes_dict=num_classes_dict,
            device=device,
            warmup_epochs=config.get("DIVIDEMIX_WARMUP_EPOCHS", 10),
            clean_threshold=config.get("DIVIDEMIX_CLEAN_THRESHOLD", 0.5),
            lambda_u=config.get("DIVIDEMIX_LAMBDA_U", 1.0),
            noise_rate=config.get("COTEACHING_NOISE_RATE", 0.2),
            beta_elr=elr_beta,
            elr_T=config.get("epochs", 100),
            label_correction_enabled=config.get("LABEL_CORRECTION_ENABLED", True),
            label_correction_ramp_epochs=config.get("LABEL_CORRECTION_RAMP_EPOCHS", 10),
            label_correction_max_rate=config.get("LABEL_CORRECTION_MAX_RATE", 0.9),
            label_correction_confidence_threshold=config.get("LABEL_CORRECTION_CONFIDENCE_THRESHOLD", 0.9),
            fixmatch_enabled=config.get("FIXMATCH_ENABLED", True),
            fixmatch_confidence_threshold=config.get("FIXMATCH_CONFIDENCE_THRESHOLD", 0.95),
            fixmatch_loss_weight=config.get("FIXMATCH_LOSS_WEIGHT", 1.0),
        )
        # Reload train dataset with return_index=True for DivideMix
        train_dataset_idx, _ = load_datasets(config, aux_feature_columns)
        train_dataset_idx.return_index = True
        train_loader_idx, _ = create_data_loaders(
            train_dataset_idx, val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
        )
        train_loader = train_loader_idx
        model2_components = setup_training_components(model2, config, device)
        optimizer2 = model2_components["optimizer"]
        logger.info("DivideMix enabled: warmup=%d, lambda_u=%.1f, clean_thresh=%.2f",
                    divide_mix.warmup_epochs, divide_mix.lambda_u, divide_mix.clean_threshold)

    elif use_coteaching:
        model2 = create_model(model_cfg, device)
        # Reload train dataset with return_index=True for sample selection
        train_dataset_idx, _ = load_datasets(config, aux_feature_columns)
        train_dataset_idx.return_index = True
        train_loader_idx, _ = create_data_loaders(
            train_dataset_idx, val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
        )
        train_loader = train_loader_idx
        model2_components = setup_training_components(model2, config, device)
        optimizer2 = model2_components["optimizer"]
        logger.info("Co-teaching enabled: noise_rate=%.2f",
                    config.get("COTEACHING_NOISE_RATE", 0.2))

    else:
        # Standard mode — enable return_index only when ELR needs it
        if use_elr and elr_beta > 0:
            train_dataset_idx, _ = load_datasets(config, aux_feature_columns)
            train_dataset_idx.return_index = True
            train_loader_idx, _ = create_data_loaders(
                train_dataset_idx, val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
            )
            train_loader = train_loader_idx

    return {
        "model": model,
        "model2": model2,
        "train_loader": train_loader,
        "val_loader": val_loader,
        **components,
        "attribute_loss_weights": config.get("ATTRIBUTE_LOSS_WEIGHTS", {}),
        "stop_gradient_heads": set(config.get("STOP_GRADIENT_HEADS", [])),
        "accuracy_attributes": accuracy_attributes,
        "use_tta": use_tta,
        "elr": elr,
        "optimizer2": optimizer2,
        "divide_mix": divide_mix,
        "loss_truncation": components.get("loss_truncation"),
    }
