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
from pathlib import Path

import torch

from lib.utils.config import load_config
from lib.utils.logging import TrainingLogger
from lib.utils.metrics_viz import generate_all_training_visualizations
from use_cases.training.pipeline import build_training_pipeline
from use_cases.training.train_model import train_epoch, validate_epoch

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
        default=5,
        help="Stop training if validation accuracy does not improve for this many epochs",
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
    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    optimizer = pipeline["optimizer"]
    criterion = pipeline["criterion"]
    scheduler = pipeline.get("scheduler")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = args.epochs or config.get("epochs", DEFAULT_EPOCHS)

    training_logger = TrainingLogger(args.log_dir)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    # text_max_length must match the value in config so fallback tensors
    # in train_epoch/validate_epoch have the correct sequence length.
    text_max_length = config.get("text_max_length", 256)
    early_stopping_patience = args.early_stopping_patience
    epochs_without_improvement = 0
    best_val_accuracy = -1.0

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_per_attr = train_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=mixup_alpha,
            text_max_length=text_max_length,
        )
        val_loss, val_accuracy, val_per_attr = validate_epoch(
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
