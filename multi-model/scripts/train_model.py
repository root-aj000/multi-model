"""
Training script for the FG_MFN model.

Runs the training loop with configurable epochs, learning rate,
and warmup schedule. Outputs training metrics to the console and
saves the best model checkpoint.

All defaults (epochs, early_stopping_patience, log_dir, checkpoint_dir, etc.)
are read from model_config.json. CLI flags override config values when provided.
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch

from lib.utils.config import load_config
from lib.utils.logging import TrainingLogger
from lib.utils.metrics_viz import generate_all_training_visualizations
from use_cases.training.pipeline import build_training_pipeline
from use_cases.training.train_model import train_epoch, validate_epoch

logger = logging.getLogger(__name__)

# Path to the default configuration file (CLI default only — not a model param)
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
        help="Stop training if val accuracy does not improve for this many epochs (overrides config)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store training logs (overrides config)",
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
    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    optimizer = pipeline["optimizer"]
    criterion = pipeline["criterion"]
    scheduler = pipeline.get("scheduler")

    # Use cuda:0 as the primary device; DataParallel (applied in create_model)
    # will automatically spread the workload across all available GPUs.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"Using {n} GPU(s): {gpu_names}")

    num_epochs = args.epochs or config.get("epochs", 100)

    log_dir = args.log_dir or config.get("log_dir", "local/logs")
    checkpoint_dir = Path(config.get("checkpoint_dir", "saved_models"))
    training_logger = TrainingLogger(log_dir)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    # text_max_length must match the value in config so fallback tensors
    # in train_epoch/validate_epoch have the correct sequence length.
    text_max_length = config.get("text_max_length", 256)
    attribute_loss_weights = pipeline.get("attribute_loss_weights", {})
    early_stopping_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else config.get("early_stopping_patience", 5)
    )
    epochs_without_improvement = 0
    best_val_accuracy = -1.0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=mixup_alpha,
            text_max_length=text_max_length,
            attribute_loss_weights=attribute_loss_weights,
        )
        val_loss, val_accuracy = validate_epoch(
            model, val_loader, criterion, device,
            text_max_length=text_max_length,
        )

        if scheduler is not None:
            scheduler.step()

        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        training_logger.log_metrics(metrics, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_path = checkpoint_dir / f"best_model_epoch_{epoch + 1}_acc_{val_accuracy:.4f}.pt"
            best_tmp = checkpoint_dir / f"{best_path.name}.tmp"
            # Unwrap DataParallel before saving so the checkpoint is portable
            state_dict = (
                model.module.state_dict()
                if isinstance(model, torch.nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state_dict, best_tmp)
            os.replace(best_tmp, best_path)
            _prune_best_checkpoints(checkpoint_dir, keep=2)
            print(f"  -> Saved best model to {best_path}")
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs "
                f"without improvement."
            )
            break

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last_model.pt"
    last_tmp = checkpoint_dir / "last_model.pt.tmp"
    state_dict = (
        model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state_dict, last_tmp)
    os.replace(last_tmp, last_path)
    print(f"  -> Saved last model to {last_path}")

    training_logger.close()
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")

    # Generate training visualizations
    print("\n" + "="*60)
    print("Generating training visualizations...")
    print("="*60)
    training_log_path = Path(log_dir) / "training_log.csv"
    viz_results = generate_all_training_visualizations(
        str(training_log_path),
        log_dir
    )
    for viz_name, success in viz_results.items():
        status = "✓ Generated" if success else "✗ Failed/Skipped"
        print(f"  {status}: {viz_name}")
    print("="*60)


if __name__ == "__main__":
    main()
