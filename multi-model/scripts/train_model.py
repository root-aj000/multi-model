"""
Training script for the FG_MFN model.

Outputs a live per-epoch table to the terminal showing train/val metrics
side-by-side for every attribute, plus a trend bar and gap column.
No extra dependencies — uses only the Python stdlib (shutil for term width).

All defaults are read from model_config.json; CLI flags override them.
"""

import argparse
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from lib.utils.config import load_config
from lib.utils.logging import TrainingLogger
from lib.utils.metrics_viz import generate_all_training_visualizations
from use_cases.training.pipeline import build_training_pipeline
from use_cases.training.train_model import train_epoch, validate_epoch

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/model/model_config.json"

BEST_CHECKPOINT_PATTERN = re.compile(
    r"^best_model_epoch_(\d+)(?:_acc_([0-9]+\.[0-9]+))?\.pt$"
)

# ── Attribute display order and short names ────────────────────────────────
ATTR_DISPLAY = [
    ("theme",            "Theme         "),
    ("dominant_colour",  "Dom. Colour   "),
    ("sentiment",        "Sentiment     "),
    ("emotion",          "Emotion       "),
    ("trust_safety",     "Trust/Safety  "),
    ("target_audience",  "Target Aud.   "),
    ("attention_score",  "Attn Score    "),
    ("predicted_ctr",    "Pred. CTR     "),
    ("likelihood_shares","Likelihood Sh."),
]

# Attributes whose ceiling is known to be ~50% (behavioural noise)
NOISE_ATTRS = {"attention_score", "predicted_ctr", "likelihood_shares"}


# ── Terminal helpers ───────────────────────────────────────────────────────

def _term_width() -> int:
    return shutil.get_terminal_size(fallback=(120, 40)).columns


def _bar(value: float, width: int = 10) -> str:
    """Compact ASCII progress bar: [████░░░░░░]"""
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _delta_str(current: float, previous: Optional[float]) -> str:
    """Return a coloured delta string: ↑+0.012 or ↓-0.008 or  ——"""
    if previous is None:
        return "  ——  "
    diff = current - previous
    if abs(diff) < 5e-4:
        return "  ——  "
    arrow = "↑" if diff > 0 else "↓"
    return f"{arrow}{diff:+.4f}"


def _acc_str(val: float, attr: str, best: Optional[float]) -> str:
    """Format accuracy: dim noise attrs, mark personal-bests with ★"""
    marker = "★" if (best is not None and abs(val - best) < 1e-6) else " "
    return f"{marker}{val:.4f}"


def _gap_indicator(gap: float) -> str:
    """Visual indicator of the train/val gap."""
    if gap < 0.05:
        return "✓"
    if gap < 0.15:
        return "~"
    if gap < 0.25:
        return "!"
    return "✗"  # severe overfit


# ── Header / epoch table printer ──────────────────────────────────────────

_TABLE_HEADER_PRINTED = False
_PREV_VAL: Dict[str, float] = {}
_BEST_VAL: Dict[str, float] = {}


def _print_run_header(config: dict, num_epochs: int, device: torch.device) -> None:
    """Print a one-time banner before training starts."""
    w = min(_term_width(), 100)
    sep = "═" * w
    print(f"\n{sep}")
    print(f"  FG-MFN  │  {config.get('IMAGE_BACKBONE','?')} + "
          f"{config.get('TEXT_ENCODER','?')}  │  fusion={config.get('FUSION_TYPE','?')}  │  "
          f"hidden={config.get('HIDDEN_DIM','?')}")
    print(f"  device={device}  │  epochs={num_epochs}  │  "
          f"batch={config.get('batch_size','?')}  │  "
          f"lr={config.get('learning_rate','?')}  │  "
          f"enc_lr={config.get('encoder_learning_rate','?')}")
    frozen = config.get('FREEZE_BACKBONE', False)
    smooth = config.get('label_smoothing', 0.0)
    warmup = config.get('warmup_epochs', 0)
    print(f"  freeze_backbone={frozen}  │  label_smoothing={smooth}  │  "
          f"warmup={warmup}  │  dropout={config.get('DROPOUT','?')}")
    print(sep)


def _print_epoch_table(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    train_per_attr: Dict[str, float],
    val_per_attr: Dict[str, float],
    best_val_acc: float,
    best_val_epoch: int,
    epochs_no_improve: int,
    patience: int,
    elapsed: float,
    lr_current: float,
) -> None:
    """
    Print a clean per-epoch summary table.

    Layout:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Epoch  8 / 100  │  Best val: 0.6565 @ ep 8  │  No improve: 0 / 10 │
    │  Time: 42.1s  │  LR: 3.00e-04  │  Train loss: 5.84  Val loss: 7.14 │
    ├──────────────────┬──────────────┬──────────────┬───────┬────────────┤
    │ Attribute        │  Train       │  Val         │  Gap  │  Trend     │
    ├──────────────────┼──────────────┼──────────────┼───────┼────────────┤
    │ Theme            │ ★0.9284      │ ★0.9370      │  0.01 │ ↑+0.0124   │
    │ ...              │              │              │       │            │
    ├──────────────────┼──────────────┼──────────────┼───────┼────────────┤
    │ OVERALL          │  0.6302      │  0.6398      │  0.01 │            │
    └──────────────────┴──────────────┴──────────────┴───────┴────────────┘
    """
    global _PREV_VAL, _BEST_VAL

    # Column widths
    W_ATTR  = 16
    W_VAL   = 13
    W_GAP   = 7
    W_TREND = 10
    total_w = W_ATTR + 2 + W_VAL + 2 + W_VAL + 2 + W_GAP + 2 + W_TREND + 1

    hbar  = "─" * total_w
    thick = "═" * total_w

    new_best = (val_acc >= best_val_acc - 1e-6)
    best_marker = " ★ NEW BEST" if new_best else ""

    # ── Header rows ──────────────────────────────────────────────────────
    print(f"\n┌{thick}┐")
    left = (f"  Epoch {epoch:>3} / {num_epochs}"
            f"  │  Best val: {best_val_acc:.4f} @ ep {best_val_epoch}"
            f"{best_marker}")
    right = f"No improve: {epochs_no_improve} / {patience}  "
    pad = total_w - len(left) - len(right) - 2
    print(f"│{left}{' ' * max(pad, 1)}{right}│")

    left2 = f"  Time: {elapsed:.1f}s  │  LR: {lr_current:.2e}"
    right2 = f"Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}  "
    pad2 = total_w - len(left2) - len(right2) - 2
    print(f"│{left2}{' ' * max(pad2, 1)}{right2}│")

    # ── Column header ────────────────────────────────────────────────────
    print(f"├{'─'*W_ATTR}┬{'─'*W_VAL}┬{'─'*W_VAL}┬{'─'*W_GAP}┬{'─'*W_TREND}┤")
    print(f"│{'Attribute':^{W_ATTR}}│{'Train':^{W_VAL}}│{'Val':^{W_VAL}}│"
          f"{'Gap':^{W_GAP}}│{'Trend (val)':^{W_TREND}}│")
    print(f"├{'─'*W_ATTR}┼{'─'*W_VAL}┼{'─'*W_VAL}┼{'─'*W_GAP}┼{'─'*W_TREND}┤")

    # ── Per-attribute rows ────────────────────────────────────────────────
    for attr_key, attr_label in ATTR_DISPLAY:
        t = train_per_attr.get(attr_key)
        v = val_per_attr.get(attr_key)
        if t is None or v is None:
            continue

        best_v = _BEST_VAL.get(attr_key)
        if best_v is None or v > best_v:
            _BEST_VAL[attr_key] = v
            best_v = v

        prev_v = _PREV_VAL.get(attr_key)
        gap    = t - v
        gap_ind = _gap_indicator(gap)

        t_str   = f" {t:.4f} "
        v_str   = f" {_acc_str(v, attr_key, best_v)} "
        gap_str = f" {gap:.3f}{gap_ind}"
        dlt_str = f" {_delta_str(v, prev_v)}"

        # Dim noise attributes with a "(noise)" marker
        label = attr_label.strip()
        if attr_key in NOISE_ATTRS:
            label = f"{label}*"

        print(f"│{label:<{W_ATTR}}│{t_str:^{W_VAL}}│{v_str:^{W_VAL}}│"
              f"{gap_str:^{W_GAP}}│{dlt_str:^{W_TREND}}│")

    # ── Overall row ──────────────────────────────────────────────────────
    print(f"├{'─'*W_ATTR}┼{'─'*W_VAL}┼{'─'*W_VAL}┼{'─'*W_GAP}┼{'─'*W_TREND}┤")
    ov_gap = train_acc - val_acc
    prev_ov = _PREV_VAL.get("__overall__")
    print(f"│{'OVERALL':<{W_ATTR}}│{f' {train_acc:.4f} ':^{W_VAL}}│"
          f"{f' {val_acc:.4f} ':^{W_VAL}}│"
          f"{f' {ov_gap:.3f}{_gap_indicator(ov_gap)}':^{W_GAP}}│"
          f"{f' {_delta_str(val_acc, prev_ov)}':^{W_TREND}}│")
    print(f"└{'─'*W_ATTR}┴{'─'*W_VAL}┴{'─'*W_VAL}┴{'─'*W_GAP}┴{'─'*W_TREND}┘")
    print(f"  * noise-ceiling attributes (behavioural, not predictable from content)")

    # Update prev for next epoch
    for attr_key, _ in ATTR_DISPLAY:
        if attr_key in val_per_attr:
            _PREV_VAL[attr_key] = val_per_attr[attr_key]
    _PREV_VAL["__overall__"] = val_acc


def _print_final_summary(
    best_val_acc: float,
    best_val_epoch: int,
    history: List[Dict],
) -> None:
    """Print a compact run summary after training finishes."""
    w = min(_term_width(), 100)
    sep = "═" * w
    print(f"\n{sep}")
    print(f"  Training complete  │  Best val accuracy: {best_val_acc:.4f}  @ epoch {best_val_epoch}")

    # Print peak val per attribute
    if history:
        best_ep = history[best_val_epoch - 1]
        print(f"\n  Per-attribute accuracy at best epoch ({best_val_epoch}):")
        for attr_key, attr_label in ATTR_DISPLAY:
            t = best_ep["train_per_attr"].get(attr_key, 0.0)
            v = best_ep["val_per_attr"].get(attr_key, 0.0)
            bar = _bar(v)
            noise = "*" if attr_key in NOISE_ATTRS else " "
            print(f"    {attr_label.strip():<16}{noise}  val={v:.4f} {bar}  train={t:.4f}  gap={t-v:+.4f}")
    print(sep)


# ── Checkpoint helpers ─────────────────────────────────────────────────────

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
    sorted_checkpoints = sorted(
        checkpoints, key=lambda item: (item[2], item[1]), reverse=True,
    )
    for path, _, _ in sorted_checkpoints[keep:]:
        try:
            path.unlink()
            logger.info("Removed old checkpoint: %s", path)
        except OSError as exc:
            logger.warning("Failed to remove checkpoint %s: %s", path, exc)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the FG_MFN multi-task ad-analysis model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_model.py
  python scripts/train_model.py --epochs 50 --batch-size 32
  python scripts/train_model.py --config my_config.json --log-dir runs/exp1
  python scripts/train_model.py --no-table          # plain text output
        """,
    )

    # ── Config & paths ──────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help=f"Path to JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory to store training logs CSV (overrides config log_dir)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory to save checkpoints (overrides config checkpoint_dir)",
    )

    # ── Training overrides ──────────────────────────────────────────────
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Total training epochs (overrides config epochs)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides config batch_size)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate for new layers (overrides config learning_rate)",
    )
    parser.add_argument(
        "--encoder-lr", type=float, default=None,
        help="LR for the pretrained text encoder (overrides config encoder_learning_rate)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=None,
        help="Linear warmup epochs (overrides config warmup_epochs)",
    )
    parser.add_argument(
        "--mixup-alpha", type=float, default=None,
        help="Mixup alpha; 0 disables Mixup (overrides config mixup_alpha)",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=None,
        help="Early-stopping patience in epochs (overrides config; 0 disables)",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", default=None,
        help="Freeze the visual backbone (overrides config FREEZE_BACKBONE)",
    )

    # ── Output control ──────────────────────────────────────────────────
    parser.add_argument(
        "--no-table", action="store_true",
        help="Print plain one-line output per epoch instead of the full table",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress all INFO logging (warnings and errors still shown)",
    )

    args = parser.parse_args()

    # ── Logging setup ───────────────────────────────────────────────────
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,       # keep logs on stderr, table on stdout
    )

    # ── Load and patch config ────────────────────────────────────────────
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.encoder_lr is not None:
        config["encoder_learning_rate"] = args.encoder_lr
    if args.warmup_epochs is not None:
        config["warmup_epochs"] = args.warmup_epochs
    if args.mixup_alpha is not None:
        config["mixup_alpha"] = args.mixup_alpha
    if args.early_stopping_patience is not None:
        config["early_stopping_patience"] = args.early_stopping_patience
    if args.freeze_backbone:
        config["FREEZE_BACKBONE"] = True

    # ── Build pipeline ───────────────────────────────────────────────────
    pipeline = build_training_pipeline(config)

    model        = pipeline["model"]
    train_loader = pipeline["train_loader"]
    val_loader   = pipeline["val_loader"]
    optimizer    = pipeline["optimizer"]
    criterion    = pipeline["criterion"]
    scheduler    = pipeline.get("scheduler")
    attribute_loss_weights = pipeline.get("attribute_loss_weights", {})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"Using {n} GPU(s): {gpu_names}", file=sys.stderr)

    num_epochs  = config.get("epochs", 100)
    log_dir     = args.log_dir or config.get("log_dir", "local/logs")
    ckpt_dir    = Path(args.checkpoint_dir or config.get("checkpoint_dir", "saved_models"))
    text_max_length = config.get("text_max_length", 256)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    early_stop_patience = config.get("early_stopping_patience", 5)

    training_logger       = TrainingLogger(log_dir)
    epochs_without_improve = 0
    best_val_acc          = -1.0
    best_val_epoch        = 1
    history: List[Dict]   = []
    use_table             = not args.no_table

    if use_table:
        _print_run_header(config, num_epochs, device)

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc, train_per_attr = train_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=mixup_alpha,
            text_max_length=text_max_length,
            attribute_loss_weights=attribute_loss_weights,
        )
        val_loss, val_acc, val_per_attr = validate_epoch(
            model, val_loader, criterion, device,
            text_max_length=text_max_length,
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        # Get current LR from first param group
        lr_current = optimizer.param_groups[0]["lr"]

        # ── Log to CSV ──────────────────────────────────────────────────
        training_logger.log_metrics({
            "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_loss,     "val_accuracy": val_acc,
        }, epoch + 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc,   "val_acc": val_acc,
            "train_per_attr": train_per_attr,
            "val_per_attr": val_per_attr,
        })

        # ── Checkpoint ──────────────────────────────────────────────────
        new_best = val_acc > best_val_acc
        if new_best:
            best_val_acc   = val_acc
            best_val_epoch = epoch + 1
            epochs_without_improve = 0
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            best_path = ckpt_dir / f"best_model_epoch_{epoch+1}_acc_{val_acc:.4f}.pt"
            best_tmp  = ckpt_dir / f"{best_path.name}.tmp"
            state_dict = (
                model.module.state_dict()
                if isinstance(model, torch.nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state_dict, best_tmp)
            os.replace(best_tmp, best_path)
            _prune_best_checkpoints(ckpt_dir, keep=2)
        else:
            epochs_without_improve += 1

        # ── Print output ─────────────────────────────────────────────────
        if use_table:
            _print_epoch_table(
                epoch=epoch + 1,
                num_epochs=num_epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                train_per_attr=train_per_attr,
                val_per_attr=val_per_attr,
                best_val_acc=best_val_acc,
                best_val_epoch=best_val_epoch,
                epochs_no_improve=epochs_without_improve,
                patience=early_stop_patience,
                elapsed=elapsed,
                lr_current=lr_current,
            )
            if new_best:
                print(f"  ★ Saved best model → {best_path}")
        else:
            # Plain one-liner
            mark = "★" if new_best else " "
            print(
                f"{mark}Epoch {epoch+1:>3}/{num_epochs} | "
                f"T-loss:{train_loss:.4f} T-acc:{train_acc:.4f} | "
                f"V-loss:{val_loss:.4f} V-acc:{val_acc:.4f} | "
                f"best:{best_val_acc:.4f}@ep{best_val_epoch} | "
                f"patience:{epochs_without_improve}/{early_stop_patience} | "
                f"{elapsed:.1f}s"
            )

        # ── Early stopping ───────────────────────────────────────────────
        if (early_stop_patience > 0
                and epochs_without_improve >= early_stop_patience):
            print(f"\n  Early stopping: no improvement for {epochs_without_improve} epochs.")
            break

    # ── Save last checkpoint ─────────────────────────────────────────────
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last_model.pt"
    last_tmp  = ckpt_dir / "last_model.pt.tmp"
    state_dict = (
        model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state_dict, last_tmp)
    os.replace(last_tmp, last_path)
    print(f"\n  Saved last model → {last_path}")

    training_logger.close()

    # ── Final summary ─────────────────────────────────────────────────────
    if use_table:
        _print_final_summary(best_val_acc, best_val_epoch, history)

    # ── Visualizations ────────────────────────────────────────────────────
    print("\n  Generating training visualizations...")
    training_log_path = Path(log_dir) / "training_log.csv"
    viz_results = generate_all_training_visualizations(str(training_log_path), log_dir)
    for viz_name, success in viz_results.items():
        status = "✓" if success else "✗"
        print(f"    {status} {viz_name}")


if __name__ == "__main__":
    main()
