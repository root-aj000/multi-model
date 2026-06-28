"""
Standalone script to generate training visualizations from CSV log.

Usage:
    python scripts/visualize_training.py
    python scripts/visualize_training.py --input local/logs/training_log.csv --output local/logs
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_all(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    epochs = df["epoch"]

    # ── 1. Training curves (2x2) ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Metrics Over Epochs", fontsize=16, fontweight="bold")

    # Loss
    axes[0, 0].plot(epochs, df["train_loss"], label="Train Loss", linewidth=2, marker=".")
    axes[0, 0].plot(epochs, df["val_loss"], label="Val Loss", linewidth=2, marker=".")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, df["train_accuracy"], label="Train Accuracy", linewidth=2, marker=".")
    axes[0, 1].plot(epochs, df["val_accuracy"], label="Val Accuracy", linewidth=2, marker=".")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training & Validation Accuracy", fontweight="bold")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Loss comparison with fill
    axes[1, 0].plot(epochs, df["train_loss"], label="Train", linewidth=2, marker=".", color="blue")
    axes[1, 0].fill_between(epochs, df["train_loss"], alpha=0.2, color="blue")
    axes[1, 0].plot(epochs, df["val_loss"], label="Val", linewidth=2, marker=".", color="orange")
    axes[1, 0].fill_between(epochs, df["val_loss"], alpha=0.2, color="orange")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Loss Comparison", fontweight="bold")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Best epoch marker
    best_idx = df["val_accuracy"].idxmax()
    best_ep = df.loc[best_idx, "epoch"]
    best_acc = df.loc[best_idx, "val_accuracy"]
    axes[1, 1].plot(epochs, df["val_accuracy"], label="Val Accuracy", linewidth=2.5, marker=".", markersize=6)
    axes[1, 1].plot(best_ep, best_acc, "r*", markersize=20, label=f"Best: Ep {int(best_ep)} ({best_acc:.4f})")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Best Validation Accuracy", fontweight="bold")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out / 'training_curves.png'}")

    # ── 2. Train/Val gap over epochs ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    gap = df["train_accuracy"] - df["val_accuracy"]
    ax.bar(epochs, gap, color=["green" if g < 0.05 else "orange" if g < 0.10 else "red" for g in gap])
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="Healthy (<5%)")
    ax.axhline(y=0.10, color="red", linestyle="--", alpha=0.7, label="Overfitting (>10%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train - Val Accuracy")
    ax.set_title("Train/Val Accuracy Gap (Overfitting Indicator)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out / "train_val_gap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out / 'train_val_gap.png'}")

    # ── 3. Loss convergence ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, df["train_loss"], label="Train Loss", linewidth=2, marker=".")
    ax.plot(epochs, df["val_loss"], label="Val Loss", linewidth=2, marker=".")
    ax.fill_between(epochs, df["train_loss"], df["val_loss"], alpha=0.15, color="red")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Convergence (Gap = Overfitting)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "loss_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out / 'loss_convergence.png'}")

    # ── 4. Summary stats ──────────────────────────────────────────────
    print(f"\n  Summary:")
    print(f"    Epochs trained:    {len(df)}")
    print(f"    Best val acc:      {best_acc:.4f} at epoch {int(best_ep)}")
    print(f"    Final train acc:   {df['train_accuracy'].iloc[-1]:.4f}")
    print(f"    Final val acc:     {df['val_accuracy'].iloc[-1]:.4f}")
    print(f"    Final gap:         {gap.iloc[-1]:.4f}")
    print(f"    Final train loss:  {df['train_loss'].iloc[-1]:.4f}")
    print(f"    Final val loss:    {df['val_loss'].iloc[-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training visualizations")
    parser.add_argument("--input", "-i", default="local/logs/training_log.csv")
    parser.add_argument("--output", "-o", default="local/logs")
    args = parser.parse_args()

    print("Generating training visualizations...")
    plot_all(args.input, args.output)
    print("Done.")
