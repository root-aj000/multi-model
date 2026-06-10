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
from torch.utils.data import DataLoader

from lib.models.factory import create_model
from lib.preprocessing.dataset import CustomDataset, load_dataset
from lib.preprocessing.image.transforms import DEFAULT_IMAGE_SIZE, build_image_transform
from lib.preprocessing.text.cleaner import clean_text
from lib.preprocessing.text.pipeline import build_text_pipeline
from lib.preprocessing.text.tokenizer import load_tokenizer, tokenize_text
from lib.utils.class_weights import compute_class_weights_from_csv, log_class_distribution
from lib.utils.config import get_dataset_paths, get_label_maps, load_config

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

    # BUG-17 FIX: no redundant default arg — max_length is captured by closure
# AFTER
    def tokenizer_fn(text: str, max_length_arg: int = None) -> dict:  # ✅ Accepts 2 args
        actual_max_length = max_length_arg if max_length_arg is not None else max_length
        return tokenize_text(text, max_length=actual_max_length, tokenizer=tokenizer)


    return build_text_pipeline(clean_text, tokenizer_fn)


def load_datasets(config: Dict[str, Any]) -> Tuple[CustomDataset, CustomDataset]:
    """
    Load and return the train and validation datasets.

    Args:
        config: Configuration dictionary with dataset paths and ATTRIBUTES.

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
    )
    val_dataset = load_dataset(
        "val",
        label_maps=label_maps,
        image_pipeline=val_image_pipeline,
        text_pipeline=text_pipeline,
        dataset_root=dataset_root,
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

    Returns:
        (train_loader, val_loader)
    """
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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


def _build_optimizer(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
) -> AdamW:
    """
    Build an AdamW optimizer with separate learning-rate parameter groups.

    The pretrained text encoder is fine-tuned at encoder_learning_rate
    (much lower than learning_rate for randomly initialised layers).
    Bias terms and normalization weights are excluded from weight decay
    following the standard BERT fine-tuning recipe.

    BUG-02 FIX: AdamW is called WITHOUT a global weight_decay argument.
    Each param group already has weight_decay set explicitly. Passing a
    global value was redundant and could silently override the 0.0 entries
    in the no-decay groups if PyTorch's behaviour ever changes.

    DataParallel note: when the model is wrapped with nn.DataParallel,
    named_parameters() adds a "module." prefix to all names. This function
    strips that prefix before matching so encoder detection works correctly
    regardless of whether DataParallel is active.

    Args:
        model: The FG_MFN model.
        cfg:   Configuration dict with learning rate settings.

    Returns:
        Configured AdamW optimizer.
    """
    lr = cfg.get("learning_rate", _FALLBACK_LEARNING_RATE)
    encoder_lr = cfg.get("encoder_learning_rate", _FALLBACK_ENCODER_LEARNING_RATE)
    weight_decay = cfg.get("weight_decay", 1e-2)

    # Parameters that must NOT have weight decay (standard BERT recipe)
    no_decay_suffixes = ("bias", "LayerNorm.weight", "layer_norm.weight")

    encoder_decay: List[torch.nn.Parameter] = []
    encoder_no_decay: List[torch.nn.Parameter] = []
    other_decay: List[torch.nn.Parameter] = []
    other_no_decay: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Strip DataParallel's "module." prefix so name matching works
        # regardless of whether the model is wrapped or not.
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
            "Optimizer: encoder (decay=%d, no_decay=%d) @ lr=%.2e; "
            "other (decay=%d, no_decay=%d) @ lr=%.2e",
            len(encoder_decay), len(encoder_no_decay), encoder_lr,
            len(other_decay), len(other_no_decay), lr,
        )
    else:
        logger.info(
            "Optimizer: %d params @ lr=%.2e (no encoder params found)",
            len(other_decay) + len(other_no_decay), lr,
        )

    # BUG-02 FIX: no global weight_decay — each group already has it set
    return AdamW(param_groups)


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
    optimizer = _build_optimizer(model, cfg)

    epochs = cfg.get("epochs")
    warmup_epochs = cfg.get("warmup_epochs")

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

    # Build criterion — supports per-attribute class weights from config.
    # CLASS_WEIGHTS must be at the top level of config (not under "model").
    class_weights = cfg.get("CLASS_WEIGHTS", {})
    if isinstance(class_weights, dict) and class_weights:
        criterion: Any = {}
        for attr_name, weights in class_weights.items():
            if isinstance(weights, list):
                criterion[attr_name] = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(weights, dtype=torch.float32, device=device),
                )
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    mixup_alpha = cfg.get("mixup_alpha", 0.0)

    logger.info(
        "Training components: lr=%.2e, encoder_lr=%.2e, wd=%.2e, "
        "scheduler=%s, warmup=%d, mixup_alpha=%.2f",
        cfg.get("learning_rate", _FALLBACK_LEARNING_RATE),
        cfg.get("encoder_learning_rate", _FALLBACK_ENCODER_LEARNING_RATE),
        cfg.get("weight_decay", 1e-2),
        scheduler_type, warmup_epochs, mixup_alpha,
    )
    return {
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "mixup_alpha": mixup_alpha,
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

    train_dataset, val_dataset = load_datasets(config)
    batch_size = config.get("batch_size", _FALLBACK_BATCH_SIZE)
    num_workers = min(
        config.get("num_workers", _FALLBACK_NUM_WORKERS),
        os.cpu_count() or 1,
    )
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
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

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        **components,
        "attribute_loss_weights": config.get("ATTRIBUTE_LOSS_WEIGHTS", {}),
    }
