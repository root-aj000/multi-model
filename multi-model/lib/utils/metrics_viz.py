"""
Comprehensive metrics visualization module for model training and evaluation.

Provides functions to generate all training and evaluation visualizations:
- Training/validation curves (loss and accuracy)
- Per-attribute accuracy plots
- Confusion matrix heatmaps
- Learning rate schedule plots
- Loss breakdown by attribute
- Metrics comparison charts
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning(
        "matplotlib/seaborn not installed — visualizations will be skipped. "
        "Install with: pip install matplotlib seaborn"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training curve visualizations
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    training_log_path: str,
    output_dir: str,
) -> bool:
    """
    Plot training and validation loss/accuracy curves from CSV log.

    Creates:
    - training_curves.png: 2x2 subplot with train/val loss and accuracy

    Args:
        training_log_path: Path to training_log.csv file
        output_dir: Directory to save visualization

    Returns:
        True if successful, False if matplotlib not available or data missing
    """
    if not MATPLOTLIB_AVAILABLE:
        return False

    log_path = Path(training_log_path)
    if not log_path.exists():
        logger.warning("Training log not found: %s", training_log_path)
        return False

    try:
        import pandas as pd
        df = pd.read_csv(log_path)
    except ImportError:
        logger.warning("pandas not installed — cannot read training log CSV")
        return False
    except Exception as e:
        logger.error("Failed to read training log: %s", e)
        return False

    if df.empty:
        logger.warning("Training log is empty")
        return False

    # Check for required columns
    required_cols = {"epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Missing columns in training log: %s", missing)
        return False

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Metrics Over Epochs", fontsize=16, fontweight="bold")

        # Loss curves
        axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2, marker=".")
        axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2, marker=".")
        axes[0, 0].set_xlabel("Epoch", fontsize=11)
        axes[0, 0].set_ylabel("Loss", fontsize=11)
        axes[0, 0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(df["epoch"], df["train_accuracy"], label="Train Accuracy", linewidth=2, marker=".")
        axes[0, 1].plot(df["epoch"], df["val_accuracy"], label="Val Accuracy", linewidth=2, marker=".")
        axes[0, 1].set_xlabel("Epoch", fontsize=11)
        axes[0, 1].set_ylabel("Accuracy", fontsize=11)
        axes[0, 1].set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Train vs Val Loss
        axes[1, 0].plot(df["epoch"], df["train_loss"], label="Train", linewidth=2, marker=".", color="blue")
        axes[1, 0].fill_between(df["epoch"], df["train_loss"], alpha=0.2, color="blue")
        axes[1, 0].plot(df["epoch"], df["val_loss"], label="Val", linewidth=2, marker=".", color="orange")
        axes[1, 0].fill_between(df["epoch"], df["val_loss"], alpha=0.2, color="orange")
        axes[1, 0].set_xlabel("Epoch", fontsize=11)
        axes[1, 0].set_ylabel("Loss", fontsize=11)
        axes[1, 0].set_title("Loss Comparison (Train vs Validation)", fontsize=12, fontweight="bold")
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # Best epoch marker
        best_epoch_idx = df["val_accuracy"].idxmax()
        best_epoch = df.loc[best_epoch_idx, "epoch"]
        best_val_acc = df.loc[best_epoch_idx, "val_accuracy"]

        axes[1, 1].plot(df["epoch"], df["val_accuracy"], label="Val Accuracy", linewidth=2.5, marker=".", markersize=6)
        axes[1, 1].plot(best_epoch, best_val_acc, "r*", markersize=20, label=f"Best: Epoch {int(best_epoch)}")
        axes[1, 1].set_xlabel("Epoch", fontsize=11)
        axes[1, 1].set_ylabel("Accuracy", fontsize=11)
        axes[1, 1].set_title("Best Validation Accuracy", fontsize=12, fontweight="bold")
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Training curves saved to %s", save_path)
        return True
    except Exception as e:
        logger.error("Failed to plot training curves: %s", e)
        return False


def plot_per_attribute_training(
    training_log_path: str,
    output_dir: str,
) -> bool:
    """
    Plot per-attribute training accuracy if available in CSV.

    Creates:
    - per_attribute_training.png: Per-attribute accuracy curves

    Args:
        training_log_path: Path to training_log.csv
        output_dir: Directory to save visualization

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        return False

    log_path = Path(training_log_path)
    if not log_path.exists():
        logger.warning("Training log not found: %s", training_log_path)
        return False

    try:
        import pandas as pd
        df = pd.read_csv(log_path)
    except ImportError:
        return False
    except Exception as e:
        logger.error("Failed to read training log: %s", e)
        return False

    # Find per-attribute accuracy columns
    attr_cols = [col for col in df.columns if "accuracy" in col.lower() and col not in
                 ["train_accuracy", "val_accuracy"]]

    if not attr_cols:
        logger.info("No per-attribute accuracy columns found in training log")
        return False

    try:
        fig, axes = plt.subplots(1, len(attr_cols), figsize=(5*len(attr_cols), 4))
        if len(attr_cols) == 1:
            axes = [axes]

        fig.suptitle("Per-Attribute Training Accuracy", fontsize=16, fontweight="bold")

        for idx, attr_col in enumerate(attr_cols):
            axes[idx].plot(df["epoch"], df[attr_col], linewidth=2, marker=".", markersize=5)
            axes[idx].set_xlabel("Epoch", fontsize=11)
            axes[idx].set_ylabel("Accuracy", fontsize=11)
            axes[idx].set_title(attr_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([0, 1])

        plt.tight_layout()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "per_attribute_training.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Per-attribute training plot saved to %s", save_path)
        return True
    except Exception as e:
        logger.error("Failed to plot per-attribute training: %s", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics visualizations
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    results_data: Dict[str, Any],
    output_dir: str,
    label_maps: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """
    Plot confusion matrices for each attribute.

    Creates:
    - confusion_matrices.png: Subplots with heatmaps for each attribute

    Args:
        results_data: Results dict from compute_metrics
        output_dir: Directory to save visualization
        label_maps: Optional mapping of attribute names to class labels

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        return False

    per_attr = results_data.get("per_attribute", {})
    if not per_attr:
        logger.warning("No per-attribute results found")
        return False

    attrs_with_cm = [attr for attr, res in per_attr.items()
                      if "confusion_matrix" in res]

    if not attrs_with_cm:
        logger.info("No confusion matrices in results")
        return False

    try:
        n_attrs = len(attrs_with_cm)
        cols = min(3, n_attrs)
        rows = (n_attrs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_attrs == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        else:
            axes = [axes[i] if isinstance(axes[i], np.ndarray) else [axes[i]]
                   for i in range(rows)]
            axes = [item for sublist in axes for item in (sublist if isinstance(sublist, list) else [sublist])]

        axes_flat = axes if isinstance(axes, list) else axes.flatten()

        fig.suptitle("Confusion Matrices per Attribute", fontsize=16, fontweight="bold")

        for idx, attr_name in enumerate(attrs_with_cm):
            ax = axes_flat[idx]
            cm = per_attr[attr_name]["confusion_matrix"]
            cm_array = np.array(cm)

            labels = (label_maps or {}).get(attr_name, None)
            if labels:
                labels = labels[:cm_array.shape[0]]
            else:
                labels = [str(i) for i in range(cm_array.shape[0])]

            sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues", ax=ax,
                       xticklabels=labels, yticklabels=labels, cbar=True)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            ax.set_title(f"{attr_name}\n(Accuracy: {per_attr[attr_name]['accuracy']:.4f})",
                        fontsize=11, fontweight="bold")

        # Hide extra subplots
        for idx in range(len(attrs_with_cm), len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "confusion_matrices.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Confusion matrices saved to %s", save_path)
        return True
    except Exception as e:
        logger.error("Failed to plot confusion matrices: %s", e)
        return False


def plot_per_attribute_metrics(
    results_data: Dict[str, Any],
    output_dir: str,
) -> bool:
    """
    Plot per-attribute precision, recall, F1 scores.

    Creates:
    - per_attribute_metrics.png: Bar charts comparing metrics

    Args:
        results_data: Results dict from compute_metrics
        output_dir: Directory to save visualization

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        return False

    per_attr = results_data.get("per_attribute", {})
    if not per_attr:
        logger.warning("No per-attribute results found")
        return False

    # Check for required metrics
    first_attr = next(iter(per_attr.values()), {})
    if "f1_weighted" not in first_attr:
        logger.info("No F1/precision/recall metrics in results")
        return False

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Per-Attribute Metrics Comparison", fontsize=16, fontweight="bold")

        attr_names = list(per_attr.keys())
        accuracies = [per_attr[a]["accuracy"] for a in attr_names]
        precisions = [per_attr[a].get("precision_weighted", 0) for a in attr_names]
        recalls = [per_attr[a].get("recall_weighted", 0) for a in attr_names]
        f1_scores = [per_attr[a].get("f1_weighted", 0) for a in attr_names]

        x = np.arange(len(attr_names))
        width = 0.35

        # Accuracy
        axes[0].bar(attr_names, accuracies, color="skyblue", edgecolor="black", linewidth=1.5)
        axes[0].set_ylabel("Accuracy", fontsize=11)
        axes[0].set_title("Accuracy by Attribute", fontsize=12, fontweight="bold")
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        # Precision & Recall
        axes[1].bar(x - width/2, precisions, width, label="Precision", color="lightgreen", edgecolor="black")
        axes[1].bar(x + width/2, recalls, width, label="Recall", color="lightcoral", edgecolor="black")
        axes[1].set_ylabel("Score", fontsize=11)
        axes[1].set_title("Precision & Recall", fontsize=12, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(attr_names, rotation=45, ha="right")
        axes[1].set_ylim([0, 1])
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis="y")

        # F1 Score
        axes[2].bar(attr_names, f1_scores, color="plum", edgecolor="black", linewidth=1.5)
        axes[2].set_ylabel("F1 Score", fontsize=11)
        axes[2].set_title("F1 Score by Attribute", fontsize=12, fontweight="bold")
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(f1_scores):
            axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "per_attribute_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Per-attribute metrics saved to %s", save_path)
        return True
    except Exception as e:
        logger.error("Failed to plot per-attribute metrics: %s", e)
        return False


def plot_macro_vs_weighted(
    results_data: Dict[str, Any],
    output_dir: str,
) -> bool:
    """
    Plot macro vs weighted accuracy comparison.

    Creates:
    - macro_weighted_comparison.png: Comparison chart

    Args:
        results_data: Results dict from compute_metrics
        output_dir: Directory to save visualization

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        return False

    macro_acc = results_data.get("accuracy_macro", 0)
    weighted_acc = results_data.get("accuracy_weighted", 0)

    try:
        fig, ax = plt.subplots(figsize=(8, 6))

        metrics = ["Macro Average", "Weighted Average"]
        values = [macro_acc, weighted_acc]
        colors = ["#FF9999", "#66B2FF"]

        bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=2, width=0.5)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Overall Accuracy: Macro vs Weighted", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        plt.tight_layout()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "macro_weighted_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Macro vs weighted comparison saved to %s", save_path)
        return True
    except Exception as e:
        logger.error("Failed to plot macro vs weighted: %s", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive visualization pipeline
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_training_visualizations(
    training_log_path: str,
    output_dir: str,
) -> Dict[str, bool]:
    """
    Generate all training-related visualizations.

    Args:
        training_log_path: Path to training_log.csv
        output_dir: Directory to save visualizations

    Returns:
        Dict mapping visualization name to success status
    """
    results = {}
    logger.info("Generating training visualizations...")

    results["training_curves"] = plot_training_curves(training_log_path, output_dir)
    results["per_attribute_training"] = plot_per_attribute_training(training_log_path, output_dir)

    return results


def generate_all_evaluation_visualizations(
    results_data: Dict[str, Any],
    output_dir: str,
    label_maps: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, bool]:
    """
    Generate all evaluation-related visualizations.

    Args:
        results_data: Results dict from compute_metrics
        output_dir: Directory to save visualizations
        label_maps: Optional mapping of attribute names to class labels

    Returns:
        Dict mapping visualization name to success status
    """
    results = {}
    logger.info("Generating evaluation visualizations...")

    results["confusion_matrices"] = plot_confusion_matrices(results_data, output_dir, label_maps)
    results["per_attribute_metrics"] = plot_per_attribute_metrics(results_data, output_dir)
    results["macro_weighted_comparison"] = plot_macro_vs_weighted(results_data, output_dir)

    return results


def generate_all_visualizations(
    training_log_path: str,
    results_data: Dict[str, Any],
    output_dir: str,
    label_maps: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, bool]]:
    """
    Generate all training and evaluation visualizations.

    Args:
        training_log_path: Path to training_log.csv
        results_data: Results dict from compute_metrics
        output_dir: Directory to save visualizations
        label_maps: Optional mapping of attribute names to class labels

    Returns:
        Dict with "training" and "evaluation" keys mapping to success dicts
    """
    all_results = {}
    all_results["training"] = generate_all_training_visualizations(training_log_path, output_dir)
    all_results["evaluation"] = generate_all_evaluation_visualizations(
        results_data, output_dir, label_maps
    )
    return all_results
