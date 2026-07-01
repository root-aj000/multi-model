"""
Training script for the FG_MFN model.

Runs the training loop with configurable epochs, learning rate,
and warmup schedule. Outputs training metrics to the console and
saves the best model checkpoint.
"""

import argparse
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from lib.utils.config import load_config
from lib.utils.logging import TrainingLogger
from lib.utils.metrics_viz import generate_all_training_visualizations
from use_cases.training.pipeline import build_training_pipeline
from use_cases.training.train_model import (
    train_epoch,
    train_epoch_coteaching,
    validate_epoch,
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

    training_logger = TrainingLogger(args.log_dir)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    # text_max_length must match the value in config so fallback tensors
    # in train_epoch/validate_epoch have the correct sequence length.
    text_max_length = config.get("text_max_length", 256)
    early_stopping_patience = args.early_stopping_patience or config.get("early_stopping_patience", 10)
    epochs_without_improvement = 0
    best_val_accuracy = -1.0

    use_dividemix = config.get("USE_DIVIDEMIX", False)
    use_coteaching = config.get("USE_COTEACHING", False) and not use_dividemix

    # Rolling history of per-attribute gaps (train - val), most recent first
    gap_history: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        if use_dividemix:
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
