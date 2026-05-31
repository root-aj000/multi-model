"""
Training script for the FG_MFN model.

Runs the training loop with configurable epochs, learning rate,
and warmup schedule. Outputs training metrics to the console and
saves the best model checkpoint.
"""

import argparse
import logging
import os

import torch

from lib.utils.config import load_config
from lib.utils.logging import TrainingLogger
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
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=mixup_alpha,
            text_max_length=text_max_length,
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
            os.makedirs("saved_models", exist_ok=True)
            # BUG-16 FIX: include epoch number in filename so it is consistent
            # with the existing saved_models/best_model_epoch_N.pt convention.
            best_path = f"saved_models/best_model_epoch_{epoch + 1}.pt"
            best_tmp = f"{best_path}.tmp"
            torch.save(model.state_dict(), best_tmp)
            os.replace(best_tmp, best_path)
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


if __name__ == "__main__":
    main()
