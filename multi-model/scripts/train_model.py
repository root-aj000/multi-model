"""
Training script for the FG_MFN model.

Runs the training loop with configurable epochs, learning rate,
and warmup schedule. Outputs training metrics to the console and
saves the best model checkpoint.
"""

import argparse
import logging
import math
import os
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from lib.models.factory import create_model
from lib.utils.config import load_config
from lib.utils.logging import TrainingLogger
from lib.utils.metrics_viz import generate_all_training_visualizations
from lib.utils.confident_learning import ConfidentLearning
from lib.preprocessing.text.back_translate import build_text_augmenter
from use_cases.training.pipeline import build_training_pipeline, setup_training_components
from use_cases.training.train_model import (
    train_epoch,
    train_epoch_coteaching,
    validate_epoch,
    DivideMix,
    L2RW,
    MetaWeightNet,
    MetaWeightNetTrainer,
    MultiModelCoTrain,
)

logger = logging.getLogger(__name__)

# Default number of training epochs
DEFAULT_EPOCHS = 100

# Default number of warmup epochs
DEFAULT_WARMUP_EPOCHS = 3

# Default base learning rate
DEFAULT_BASE_LR = 0.001

# Default configuration file path
DEFAULT_CONFIG_PATH = "configs/model/model_config.json"

BEST_CHECKPOINT_PATTERN = re.compile(
    r"^best_model_epoch_(\d+)(?:_acc_([0-9]+\.[0-9]+))?\.pt$"
)


def _find_best_checkpoints(save_dir: Path):
    checkpoints = []
    for path in save_dir.glob("best_model_epoch_*.pt"):
        match = BEST_CHECKPOINT_PATTERN.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        accuracy = float(match.group(2)) if match.group(2) else -1.0
        checkpoints.append((path, epoch, accuracy))
    return checkpoints


def _print_attr_table(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    val_loss: float,
    train_accuracy: float,
    val_accuracy: float,
    train_per_attr: Dict[str, float],
    val_per_attr: Dict[str, float],
    gap_history: List[Dict[str, float]],
) -> None:
    """
    Print a per-attribute train/val comparison table to the console.

    Args:
        train_loss / val_loss: Scalar losses.
        train_accuracy / val_accuracy: Overall accuracies.
        train_per_attr / val_per_attr: Per-attribute accuracies from the
            current epoch.
        gap_history: List of per-attribute gaps (train - val) from past
            epochs, most recent first. Used to show the rolling average
            gap over the previous N epochs.
    """
    all_attrs = sorted(
        set(list(train_per_attr.keys()) + list(val_per_attr.keys()))
    )

    # ── Summary line ──
    loss_gap = val_loss - train_loss
    acc_gap = train_accuracy - val_accuracy
    print(f"\nEpoch {epoch:>3}/{num_epochs} │ "
          f"Train Loss: {train_loss:.4f} │ Val Loss: {val_loss:.4f} │ ΔLoss: {loss_gap:+.4f}")
    print(f"{'':>13} │ "
          f"Train Acc: {train_accuracy:.4f} │ Val Acc:  {val_accuracy:.4f} │ ΔAcc: {acc_gap:+.4f}")

    # ── Column widths ──
    attr_w = max(len(a) for a in all_attrs) + 2
    num_w = 8
    gap_w = 9

    sep = "─" * (attr_w + 1 + num_w + 1 + num_w + 1 + gap_w + 1 + gap_w)

    hdr = (
        f"{'Attribute':>{attr_w}} │ "
        f"{'Train':>{num_w}} │ {'Val':>{num_w}} │ "
        f"{'Δ':>{gap_w}} │ {'Δ(prev5)':>{gap_w}}"
    )
    print(sep)
    print(hdr)
    print(sep)

    # Compute rolling average of gaps over the last N epochs
    n_prev = min(5, len(gap_history))
    if n_prev > 0:
        avg_prev_gaps: Dict[str, float] = {}
        for attr in all_attrs:
            vals = [h[attr] for h in gap_history[:n_prev] if attr in h]
            avg_prev_gaps[attr] = sum(vals) / len(vals) if vals else 0.0
    else:
        avg_prev_gaps = {}

    overall_gap = train_accuracy - val_accuracy
    overall_prev_gaps = [
        h.get("__overall__", 0.0) for h in gap_history[:n_prev]
    ]
    overall_avg_gap = (
        sum(overall_prev_gaps) / len(overall_prev_gaps)
        if overall_prev_gaps else 0.0
    )

    for attr in all_attrs:
        t = train_per_attr.get(attr, 0.0)
        v = val_per_attr.get(attr, 0.0)
        gap = t - v
        avg_gap = avg_prev_gaps.get(attr, 0.0)
        print(
            f"{attr:>{attr_w}} │ "
            f"{t:>{num_w}.4f} │ {v:>{num_w}.4f} │ "
            f"{gap:>+{gap_w}.4f} │ {avg_gap:>+{gap_w}.4f}"
        )

    print(sep)
    print(
        f"{'Overall':>{attr_w}} │ "
        f"{train_accuracy:>{num_w}.4f} │ {val_accuracy:>{num_w}.4f} │ "
        f"{overall_gap:>+{gap_w}.4f} │ {overall_avg_gap:>+{gap_w}.4f}"
    )
    print()


def _prune_best_checkpoints(save_dir: Path, keep: int = 2) -> None:
    checkpoints = _find_best_checkpoints(save_dir)
    if len(checkpoints) <= keep:
        return

    # Keep the top-N checkpoints by validation accuracy, breaking ties by epoch.
    sorted_checkpoints = sorted(
        checkpoints,
        key=lambda item: (item[2], item[1]),
        reverse=True,
    )
    to_remove = sorted_checkpoints[keep:]
    for path, _, _ in to_remove:
        try:
            path.unlink()
            logger.info("Removed old checkpoint: %s", path)
        except OSError as exc:
            logger.warning("Failed to remove checkpoint %s: %s", path, exc)


SAVE_DIR = Path("saved_models")
RESUME_CKPT = SAVE_DIR / "resume.pt"


def _collect_divide_mix_state(dm: Optional[DivideMix]) -> Optional[Dict[str, Any]]:
    if dm is None:
        return None
    return {
        "loss_history": {k: v.cpu() for k, v in dm.loss_history.items()},
        "correction_label": {k: v.cpu() for k, v in dm.correction_label.items()},
        "correction_confidence": {k: v.cpu() for k, v in dm.correction_confidence.items()},
        "elr1_ema": {k: v.cpu() for k, v in dm.elr1.ema.items()},
        "elr2_ema": {k: v.cpu() for k, v in dm.elr2.ema.items()},
        "_gmm_cache": {},  # Don't persist GMM cache — recomputed on resume
    }


def _restore_divide_mix_state(dm: DivideMix, state: Optional[Dict[str, Any]], device: torch.device) -> None:
    if dm is None or state is None:
        return
    for k, v in state["loss_history"].items():
        dm.loss_history[k] = v.to(device)
    for k, v in state["correction_label"].items():
        dm.correction_label[k] = v.to(device)
    for k, v in state["correction_confidence"].items():
        dm.correction_confidence[k] = v.to(device)
    for k, v in state["elr1_ema"].items():
        dm.elr1.ema[k] = v.to(device)
    for k, v in state["elr2_ema"].items():
        dm.elr2.ema[k] = v.to(device)
    dm._gmm_cache.clear()


def _collect_multico_state(
    mct: Optional[MultiModelCoTrain],
) -> Optional[Dict[str, Any]]:
    if mct is None:
        return None
    state: Dict[str, Any] = {
        "loss_histories": {k: v.cpu() for k, v in mct.loss_histories.items()},
    }
    elr_emas: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, elr_obj in mct.elr_objs.items():
        elr_emas[key] = {k: v.cpu() for k, v in elr_obj.ema.items()}
    state["elr_emas"] = elr_emas
    return state


def _restore_multico_state(
    mct: Optional[MultiModelCoTrain], state: Optional[Dict[str, Any]], device: torch.device,
) -> None:
    if mct is None or state is None:
        return
    for k, v in state["loss_histories"].items():
        mct.loss_histories[k] = v.to(device)
    for key, ema_dict in state["elr_emas"].items():
        if key in mct.elr_objs:
            for k2, v2 in ema_dict.items():
                mct.elr_objs[key].ema[k2] = v2.to(device)
    mct._gmm_cache.clear()


def _save_checkpoint(
    epoch: int,
    best_val_accuracy: float,
    epochs_without_improvement: int,
    gap_history: List[Dict[str, float]],
    model: torch.nn.Module,
    model2: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    optimizer2: Optional[torch.optim.Optimizer],
    scheduler: Any,
    divide_mix: Optional[DivideMix],
    multi_model_cotrain: Optional[MultiModelCoTrain],
    models_list: Optional[List[torch.nn.Module]],
    optimizers_list: Optional[List[torch.optim.Optimizer]],
    device: torch.device,
    path: Path = RESUME_CKPT,
) -> None:
    ckpt: Dict[str, Any] = {
        "epoch": epoch,
        "best_val_accuracy": best_val_accuracy,
        "epochs_without_improvement": epochs_without_improvement,
        "gap_history": gap_history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "divide_mix_state": _collect_divide_mix_state(divide_mix),
        "multico_state": _collect_multico_state(multi_model_cotrain),
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if model2 is not None:
        ckpt["model2_state_dict"] = model2.state_dict()
    if optimizer2 is not None:
        ckpt["optimizer2_state_dict"] = optimizer2.state_dict()
    if models_list is not None and len(models_list) > 2:
        ckpt["models_list_state"] = [m.state_dict() for m in models_list[2:]]
    if optimizers_list is not None and len(optimizers_list) > 2:
        ckpt["optimizers_list_state"] = [opt.state_dict() for opt in optimizers_list[2:]]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(str(tmp), str(path))
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch + 1)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model2: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    optimizer2: Optional[torch.optim.Optimizer],
    scheduler: Any,
    divide_mix: Optional[DivideMix],
    multi_model_cotrain: Optional[MultiModelCoTrain],
    models_list: Optional[List[torch.nn.Module]],
    optimizers_list: Optional[List[torch.optim.Optimizer]],
    device: torch.device,
) -> Tuple[int, float, int, List[Dict[str, float]]]:
    logger.info("Loading checkpoint from %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if model2 is not None and "model2_state_dict" in ckpt:
        model2.load_state_dict(ckpt["model2_state_dict"])
    if optimizer2 is not None and "optimizer2_state_dict" in ckpt:
        optimizer2.load_state_dict(ckpt["optimizer2_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if models_list is not None and "models_list_state" in ckpt:
        for m, sd in zip(models_list[2:], ckpt["models_list_state"]):
            m.load_state_dict(sd)
    if optimizers_list is not None and "optimizers_list_state" in ckpt:
        for opt, sd in zip(optimizers_list[2:], ckpt["optimizers_list_state"]):
            opt.load_state_dict(sd)
    _restore_divide_mix_state(divide_mix, ckpt.get("divide_mix_state"), device)
    _restore_multico_state(multi_model_cotrain, ckpt.get("multico_state"), device)
    epoch = ckpt.get("epoch", -1)
    best_val = ckpt.get("best_val_accuracy", -1.0)
    patience = ckpt.get("epochs_without_improvement", 0)
    gap_hist = ckpt.get("gap_history", [])
    logger.info("Resumed from epoch %d (best_val=%.4f)", epoch + 1, best_val)
    return epoch, best_val, patience, gap_hist


def main() -> None:
    """
    Wire the training use case and run training.

    Parses command-line arguments, builds the training pipeline,
    and executes the training loop with validation after each epoch.
    """
    parser = argparse.ArgumentParser(
        description="Train the FG_MFN model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Override the warmup epoch count from config",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=None,
        help="Mixup alpha value to override the config",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop training if validation accuracy does not improve for this many epochs (overrides config)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="local/logs",
        help="Directory to store training logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save a checkpoint every N epochs (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)
    if args.warmup_epochs is not None:
        config["warmup_epochs"] = args.warmup_epochs
    if args.mixup_alpha is not None:
        config["mixup_alpha"] = args.mixup_alpha

    pipeline = build_training_pipeline(config)

    model = pipeline["model"]
    model2 = pipeline.get("model2")
    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    optimizer = pipeline["optimizer"]
    optimizer2 = pipeline.get("optimizer2")
    criterion = pipeline["criterion"]
    scheduler = pipeline.get("scheduler")
    attribute_loss_weights = pipeline.get("attribute_loss_weights", {})
    stop_gradient_heads = pipeline.get("stop_gradient_heads", set())
    accuracy_attributes = pipeline.get("accuracy_attributes", None)
    elr = pipeline.get("elr")
    divide_mix = pipeline.get("divide_mix")
    loss_truncation = pipeline.get("loss_truncation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = args.epochs or config.get("epochs", DEFAULT_EPOCHS)

    use_dividemix = config.get("USE_DIVIDEMIX", False)
    use_coteaching = config.get("USE_COTEACHING", False) and not use_dividemix

    # ── SimCLR self-supervised pre-training ──────────────────────────────
    if config.get("SIMCLR_PRETRAIN", False):
        logger.info("SimCLR pre-training enabled — running before supervised training.")
        from lib.ssl.simclr import SimCLR as SimCLRModel
        unwrapped = model.module if hasattr(model, 'module') else model
        simclr_backbone = getattr(unwrapped, 'visual_module', None)
        if simclr_backbone is None:
            logger.warning("Cannot extract visual backbone for SimCLR — skipping pretraining.")
        else:
            simclr_cfg = config.get("SIMCLR_CONFIG", {})
            simclr_model = SimCLRModel(simclr_backbone,
                                        projection_dim=simclr_cfg.get("projection_dim", 128),
                                        temperature=simclr_cfg.get("temperature", 0.5))
            simclr_model.to(device)

            class _LARS(torch.optim.Optimizer):
                def __init__(self, params, lr, weight_decay=1e-4, eta=0.001):
                    super().__init__(params, dict(lr=lr, weight_decay=weight_decay, eta=eta))
                @torch.no_grad()
                def step(self, closure=None):
                    loss = None if closure is None else closure()
                    for g in self.param_groups:
                        lr, wd, eta = g['lr'], g['weight_decay'], g['eta']
                        for p in g['params']:
                            if p.grad is None: continue
                            grad = p.grad.add(p, alpha=wd) if wd > 0 else p.grad
                            trust = 1.0
                            pn, gn = p.norm(), grad.norm()
                            if pn > 0 and gn > 0:
                                trust = eta * pn / (gn + 1e-8)
                            p.add_(grad, alpha=-lr * trust)
                    return loss

            simclr_opt = _LARS(simclr_model.parameters(), lr=simclr_cfg.get("lr", 0.3))
            simclr_bs = simclr_cfg.get("batch_size", 256)
            simclr_ep = simclr_cfg.get("epochs", 200)
            from torchvision import transforms as T
            class _SimCLRTransform:
                def __init__(self, size=224):
                    self.t = T.Compose([
                        T.RandomResizedCrop(size, scale=(0.08, 1.0)),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                        T.RandomGrayscale(p=0.2),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                def __call__(self, x):
                    return self.t(x), self.t(x)
            from torchvision.datasets import ImageFolder
            from torch.utils.data import DataLoader
            dataset_root = config.get("DATASET_ROOT") or "dataset"
            simclr_ds = ImageFolder(root=f"{dataset_root}/images/train", transform=_SimCLRTransform())
            simclr_loader = DataLoader(simclr_ds, batch_size=simclr_bs, shuffle=True,
                                        num_workers=min(8, os.cpu_count() or 4), pin_memory=True, drop_last=True)
            for ep in range(simclr_ep):
                warmup = 10
                if ep < warmup:
                    lr = simclr_cfg.get("lr", 0.3) * (ep + 1) / warmup
                else:
                    progress = (ep - warmup) / max(1, simclr_ep - warmup)
                    lr = simclr_cfg.get("lr", 0.3) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in simclr_opt.param_groups:
                    pg['lr'] = lr
                simclr_model.train()
                total = 0.0
                for images, _ in simclr_loader:
                    x1, x2 = images
                    x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
                    loss, _, _ = simclr_model(x1, x2)
                    simclr_opt.zero_grad()
                    loss.backward()
                    simclr_opt.step()
                    total += loss.item() * x1.size(0)
                logger.info("SimCLR epoch %3d/%d: loss=%.4f lr=%.2e", ep + 1, simclr_ep, total / len(simclr_ds), lr)

    # ── Multi-model co-training (N >= 3) ─────────────────────────────────
    use_multico = use_dividemix and config.get("N_CO_TRAIN_MODELS", 2) > 2
    n_models = config.get("N_CO_TRAIN_MODELS", 2) if use_multico else 2
    multi_model_cotrain = None
    if use_multico:
        models_list = [model, model2]
        optimizers_list = [optimizer, optimizer2]
        for i in range(2, n_models):
            m_i = create_model(config.get("model", config), device)
            models_list.append(m_i)
            comps_i = setup_training_components(m_i, config, device)
            optimizers_list.append(comps_i["optimizer"])
        num_classes_dict = {
            name: attr_cfg["num_classes"]
            for name, attr_cfg in config.get("ATTRIBUTES", {}).items()
        }
        multi_model_cotrain = MultiModelCoTrain(
            n_samples=len(train_loader.dataset),
            num_classes_dict=num_classes_dict,
            device=device,
            n_models=n_models,
            warmup_epochs=config.get("DIVIDEMIX_WARMUP_EPOCHS", 10),
            clean_threshold=config.get("DIVIDEMIX_CLEAN_THRESHOLD", 0.5),
            lambda_u=config.get("DIVIDEMIX_LAMBDA_U", 1.0),
            noise_rate=config.get("COTEACHING_NOISE_RATE", 0.2),
            beta_elr=config.get("ELR_BETA", 0.7),
            use_gmm=True,
        )
        # Use same loader for all models
        loaders_tuple = tuple(train_loader for _ in range(n_models))
        logger.info("MultiModelCoTrain enabled: %d models", n_models)

    # ── Confidence Learning setup ────────────────────────────────────────
    confident_learning = None
    if config.get("USE_CONFIDENT_LEARNING", False):
        num_classes_dict = {
            name: attr_cfg["num_classes"]
            for name, attr_cfg in config.get("ATTRIBUTES", {}).items()
        }
        confident_learning = ConfidentLearning(
            num_classes_dict, threshold=config.get("CONFIDENT_LEARNING_THRESHOLD", "auto"),
        )
        logger.info("ConfidentLearning enabled: threshold=%s", confident_learning.threshold)

    # ── L2RW / Meta-Weight-Net setup ────────────────────────────────────
    l2rw = None
    meta_weight_trainer = None
    mwn = None
    if config.get("USE_L2RW", False) and not use_dividemix:
        l2rw = L2RW(val_loader, device, **config.get("L2RW_CONFIG", {}))
        logger.info("L2RW enabled: meta_lr=%.4f", l2rw.meta_lr)
    if config.get("USE_META_WEIGHT_NET", False) and not use_dividemix:
        mwn = MetaWeightNet(**config.get("META_WEIGHT_NET_CONFIG", {}))
        mwn.to(device)
        meta_weight_trainer = MetaWeightNetTrainer(
            mwn, val_loader, device,
            meta_lr=config.get("META_WEIGHT_NET_CONFIG", {}).get("meta_lr", 1e-4),
        )
        logger.info("Meta-Weight-Net enabled: hidden=%d, max_weight=%.1f",
                     mwn.net[0].out_features, mwn.max_weight)

    # ── UDA text augmentation setup ──────────────────────────────────────
    text_augmentor = None
    if config.get("USE_UDA_TEXT_AUG", False) and config.get("use_text", False):
        uda_cfg = config.get("UDA_TEXT_AUG_CONFIG", {})
        text_augmentor = build_text_augmenter(**uda_cfg)
        logger.info("UDA text augmentation enabled: method=%s", uda_cfg.get("method", "back_translate"))

    training_logger = TrainingLogger(args.log_dir)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    # text_max_length must match the value in config so fallback tensors
    # in train_epoch/validate_epoch have the correct sequence length.
    text_max_length = config.get("text_max_length", 256)
    early_stopping_patience = args.early_stopping_patience or config.get("early_stopping_patience", 10)
    checkpoint_interval = args.checkpoint_interval or config.get("checkpoint_interval", 5)
    epochs_without_improvement = 0
    best_val_accuracy = -1.0

    # Rolling history of per-attribute gaps (train - val), most recent first
    gap_history: List[Dict[str, float]] = []

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    resume_path = args.resume or (RESUME_CKPT if RESUME_CKPT.exists() else None)
    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            start_epoch, best_val_accuracy, epochs_without_improvement, gap_history = _load_checkpoint(
                rp, model, model2, optimizer, optimizer2, scheduler,
                divide_mix, multi_model_cotrain,
                models_list if use_multico else None,
                optimizers_list if use_multico else None,
                device,
            )
            start_epoch += 1  # resume from next epoch
            logger.info("Resuming training from epoch %d", start_epoch + 1)

    for epoch in range(start_epoch, num_epochs):
        # ── Confidence Learning pre-epoch analysis ─────────────────────
        if confident_learning is not None and epoch >= 1:
            with torch.no_grad():
                model.eval()
                all_probs: Dict[str, List[torch.Tensor]] = {}
                all_labels: Dict[str, List[torch.Tensor]] = {}
                for batch in val_loader:
                    inputs_v = batch[0].to(device)
                    labels_v = batch[1]
                    if isinstance(labels_v, dict):
                        pass
                    else:
                        labels_v = {k: v.to(device) for k, v in labels_v.items()} if isinstance(batch[1], dict) else {}
                    input_ids_v = batch[2].to(device) if len(batch) >= 4 else None
                    attn_v = batch[3].to(device) if len(batch) >= 4 else None
                    aux_v = batch[4].to(device) if len(batch) >= 5 else None
                    out_v = model(inputs_v, input_ids_v, attn_v, aux_features=aux_v)
                    for attr, logits in out_v.items():
                        all_probs.setdefault(attr, []).append(torch.softmax(logits, dim=-1))
                        if isinstance(batch[1], dict) and attr in batch[1]:
                            all_labels.setdefault(attr, []).append(batch[1][attr].to(device))
                stacked_probs = {a: torch.cat(v, dim=0) for a, v in all_probs.items()}
                stacked_labels = {a: torch.cat(v, dim=0) for a, v in all_labels.items()}
                issues = confident_learning.find_label_issues(stacked_probs, stacked_labels)
                logger.info("ConfidentLearning: %d potential label issues found in validation set", len(issues))

        if multi_model_cotrain is not None:
            train_loss, train_accuracy, train_per_attr = multi_model_cotrain.train_epoch(
                tuple(models_list), loaders_tuple, tuple(optimizers_list),
                criterion, device,
                epoch=epoch + 1,
                num_epochs=num_epochs,
                text_max_length=text_max_length,
                attribute_loss_weights=attribute_loss_weights,
                stop_gradient_heads=stop_gradient_heads,
                accuracy_attributes=accuracy_attributes,
                loss_truncation=loss_truncation,
                verbose=True,
            )
            val_loss, val_accuracy, val_per_attr = validate_epoch(
                models_list[0], val_loader, criterion, device,
                text_max_length=text_max_length,
                accuracy_attributes=accuracy_attributes,
                use_tta=pipeline.get("use_tta", False),
            )
        elif use_dividemix:
            train_loss, train_accuracy, train_per_attr = divide_mix.train_epoch(
                model, model2, train_loader, criterion,
                optimizer, optimizer2, device,
                epoch=epoch + 1,
                text_max_length=text_max_length,
                attribute_loss_weights=attribute_loss_weights,
                stop_gradient_heads=stop_gradient_heads,
                accuracy_attributes=accuracy_attributes,
                loss_truncation=loss_truncation,
                verbose=True,
            )
            # Use model1 for validation
            val_loss, val_accuracy, val_per_attr = validate_epoch(
                model, val_loader, criterion, device,
                text_max_length=text_max_length,
                accuracy_attributes=accuracy_attributes,
                use_tta=pipeline.get("use_tta", False),
            )
        elif use_coteaching:
            train_loss, train_accuracy, train_per_attr = train_epoch_coteaching(
                model, model2, train_loader, criterion,
                optimizer, optimizer2, device,
                epoch=epoch + 1,
                noise_rate=config.get("COTEACHING_NOISE_RATE", 0.2),
                num_epochs=num_epochs,
                mixup_alpha=mixup_alpha,
                text_max_length=text_max_length,
                attribute_loss_weights=attribute_loss_weights,
                stop_gradient_heads=stop_gradient_heads,
                accuracy_attributes=accuracy_attributes,
                loss_truncation=loss_truncation,
            )
            # Use model1 for validation
            val_loss, val_accuracy, val_per_attr = validate_epoch(
                model, val_loader, criterion, device,
                text_max_length=text_max_length,
                accuracy_attributes=accuracy_attributes,
                use_tta=pipeline.get("use_tta", False),
            )
        else:
            train_loss, train_accuracy, train_per_attr = train_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch=epoch + 1,
                mixup_alpha=mixup_alpha,
                text_max_length=text_max_length,
                attribute_loss_weights=attribute_loss_weights,
                stop_gradient_heads=stop_gradient_heads,
                accuracy_attributes=accuracy_attributes,
                elr=elr,
                loss_truncation=loss_truncation,
                l2rw=l2rw,
                meta_weight_trainer=meta_weight_trainer,
                mwn=mwn,
            )
            val_loss, val_accuracy, val_per_attr = validate_epoch(
                model, val_loader, criterion, device,
                text_max_length=text_max_length,
                accuracy_attributes=accuracy_attributes,
                use_tta=pipeline.get("use_tta", False),
            )

        if scheduler is not None:
            scheduler.step()

        # ── Update gap history ──
        epoch_gaps: Dict[str, float] = {"__overall__": train_accuracy - val_accuracy}
        if train_per_attr and val_per_attr:
            all_attrs = set(list(train_per_attr.keys()) + list(val_per_attr.keys()))
            for a in all_attrs:
                epoch_gaps[a] = train_per_attr.get(a, 0.0) - val_per_attr.get(a, 0.0)
        gap_history.insert(0, epoch_gaps)
        if len(gap_history) > 10:
            gap_history.pop()

        # ── Per-attribute table ──
        _print_attr_table(
            epoch + 1, num_epochs,
            train_loss, val_loss,
            train_accuracy, val_accuracy,
            train_per_attr, val_per_attr,
            gap_history[1:],  # exclude current epoch
        )

        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        training_logger.log_metrics(metrics, epoch + 1)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_dir = Path("saved_models")
            save_dir.mkdir(parents=True, exist_ok=True)
            best_path = save_dir / f"best_model_epoch_{epoch + 1}_acc_{val_accuracy:.4f}.pt"
            best_tmp = save_dir / f"{best_path.name}.tmp"
            torch.save(model.state_dict(), best_tmp)
            os.replace(best_tmp, best_path)
            _prune_best_checkpoints(save_dir, keep=2)
            print(f"  -> Saved best model to {best_path}")
        else:
            epochs_without_improvement += 1

        # ── Periodic checkpoint for resumability ──
        if (epoch + 1) % checkpoint_interval == 0:
            _save_checkpoint(
                epoch, best_val_accuracy, epochs_without_improvement, gap_history,
                model, model2, optimizer, optimizer2, scheduler,
                divide_mix, multi_model_cotrain,
                models_list if use_multico else None,
                optimizers_list if use_multico else None,
                device,
            )
        # Also save a lightweight resume checkpoint every epoch for crash recovery
        _save_checkpoint(
            epoch, best_val_accuracy, epochs_without_improvement, gap_history,
            model, model2, optimizer, optimizer2, scheduler,
            divide_mix, multi_model_cotrain,
            models_list if use_multico else None,
            optimizers_list if use_multico else None,
            device,
            path=SAVE_DIR / "resume_latest.pt",
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs "
                f"without improvement."
            )
            break

    os.makedirs("saved_models", exist_ok=True)
    last_path = "saved_models/last_model.pt"
    last_tmp = f"{last_path}.tmp"
    torch.save(model.state_dict(), last_tmp)
    os.replace(last_tmp, last_path)
    print(f"  -> Saved last model to {last_path}")
    # Final full checkpoint
    _save_checkpoint(
        epoch, best_val_accuracy, epochs_without_improvement, gap_history,
        model, model2, optimizer, optimizer2, scheduler,
        divide_mix, multi_model_cotrain,
        models_list if use_multico else None,
        optimizers_list if use_multico else None,
        device,
        path=SAVE_DIR / "final_resume.pt",
    )

    training_logger.close()
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")

    # Generate training visualizations
    print("\n" + "="*60)
    print("Generating training visualizations...")
    print("="*60)
    training_log_path = Path(args.log_dir) / "training_log.csv"
    viz_results = generate_all_training_visualizations(
        str(training_log_path),
        args.log_dir
    )
    for viz_name, success in viz_results.items():
        status = "✓ Generated" if success else "✗ Failed/Skipped"
        print(f"  {status}: {viz_name}")
    print("="*60)


if __name__ == "__main__":
    main()
