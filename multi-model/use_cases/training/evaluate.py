"""
Model evaluation use case.

Fixes applied from code_review_accuracy_assessment:

  Issue 9  — Macro-average masked per-attribute imbalance.
             Now reports BOTH:
               accuracy_macro    : unweighted mean across attributes
               accuracy_weighted : sample-count-weighted mean across attributes
             The weighted metric is the honest number when attributes have
             different numbers of test samples.

  Issue 10 — Missing per-class metrics.
             compute_metrics now adds per-attribute precision, recall, and
             F1 (macro and weighted averages) via sklearn.metrics.
             Also adds per-attribute confusion matrices.
             sklearn is a standard ML dependency — no new installs needed.

Previously fixed:
  BUG-03 — Optional imported at top (not bottom).
  BUG-05 — Fallback seq_len = 256 (was 128).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "results.json"
_FALLBACK_SEQ_LEN = 256  # must match text_max_length in config


# ──────────────────────────────────────────────────────────────────────────────
# Inference loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    text_max_length: int = _FALLBACK_SEQ_LEN,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run inference on all batches and collect per-attribute predictions/labels.

    Args:
        model:           Model in eval mode (load_model() sets this automatically).
        dataloader:      DataLoader for the evaluation split.
        device:          Compute device.
        text_max_length: Fallback seq_len when batch has no text tensors.
                         Must match text_max_length in config.

    Returns:
        {
          "per_attr_preds"  : Dict[attr_name, List[int]],
          "per_attr_labels" : Dict[attr_name, List[int]],
          "all_preds"       : List[int],   # first attribute (backward compat)
          "all_labels"      : List[int],   # first attribute (backward compat)
        }
    """
    model.eval()

    per_attr_preds:  Dict[str, List[int]] = {}
    per_attr_labels: Dict[str, List[int]] = {}

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1]
            batch_size = inputs.size(0)

            if len(batch) >= 4:
                input_ids      = batch[2].to(device)
                attention_mask = batch[3].to(device)
            else:
                input_ids = torch.zeros(
                    batch_size, text_max_length, dtype=torch.long, device=device
                )
                attention_mask = torch.zeros(
                    batch_size, text_max_length, dtype=torch.long, device=device
                )

            outputs = model(inputs, input_ids, attention_mask)

            if isinstance(outputs, dict):
                for attr_name, logits in outputs.items():
                    _, preds = logits.max(1)
                    if attr_name not in per_attr_preds:
                        per_attr_preds[attr_name]  = []
                        per_attr_labels[attr_name] = []
                    per_attr_preds[attr_name].extend(preds.cpu().tolist())

                    if isinstance(labels, dict) and attr_name in labels:
                        al = labels[attr_name]
                        per_attr_labels[attr_name].extend(
                            al.cpu().tolist() if isinstance(al, torch.Tensor) else list(al)
                        )
                    elif not isinstance(labels, dict):
                        per_attr_labels[attr_name].extend(
                            labels.cpu().tolist()
                            if isinstance(labels, torch.Tensor) else list(labels)
                        )
            else:
                _, preds = outputs.max(1)
                attr_name = "output"
                if attr_name not in per_attr_preds:
                    per_attr_preds[attr_name]  = []
                    per_attr_labels[attr_name] = []
                per_attr_preds[attr_name].extend(preds.cpu().tolist())
                per_attr_labels[attr_name].extend(
                    labels.cpu().tolist()
                    if isinstance(labels, torch.Tensor) else list(labels)
                )

    first_attr = next(iter(per_attr_preds)) if per_attr_preds else None
    all_preds  = per_attr_preds.get(first_attr,  []) if first_attr else []
    all_labels = per_attr_labels.get(first_attr, []) if first_attr else []

    logger.info(
        "Evaluation complete: %d samples, %d attribute heads",
        len(all_preds), len(per_attr_preds),
    )
    return {
        "per_attr_preds":  per_attr_preds,
        "per_attr_labels": per_attr_labels,
        "all_preds":       all_preds,
        "all_labels":      all_labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds:        List[int],
    all_labels:       List[int],
    per_attr_preds:   Optional[Dict[str, List[int]]] = None,
    per_attr_labels:  Optional[Dict[str, List[int]]] = None,
    label_maps:       Optional[Dict[str, List[str]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics from predictions and labels.

    Issue 9 fix  — reports both macro and sample-weighted accuracy.
    Issue 10 fix — reports per-attribute precision, recall, F1, and
                   confusion matrices using sklearn.

    Args:
        all_preds:       Flat prediction list (first attribute or legacy).
        all_labels:      Flat label list (first attribute or legacy).
        per_attr_preds:  Per-attribute prediction lists.
        per_attr_labels: Per-attribute label lists.
        label_maps:      Optional dict mapping attr_name → list of label
                         strings. Used to name confusion matrix rows/cols.

    Returns:
        {
          "accuracy_macro"    : float  — unweighted mean across attributes
          "accuracy_weighted" : float  — sample-count-weighted mean
          "total"             : int    — total evaluation samples
          "per_attribute"     : {
              "<attr>": {
                  "accuracy"          : float,
                  "precision_macro"   : float,
                  "recall_macro"      : float,
                  "f1_macro"          : float,
                  "precision_weighted": float,
                  "recall_weighted"   : float,
                  "f1_weighted"       : float,
                  "support"           : int,
                  "confusion_matrix"  : List[List[int]],
                  "per_class"         : {
                      "<label>": {
                          "precision": float,
                          "recall"   : float,
                          "f1"       : float,
                          "support"  : int,
                      }, ...
                  }
              }, ...
          }
        }
    """
    metrics: Dict[str, Any] = {}

    if per_attr_preds and per_attr_labels:
        try:
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                accuracy_score,
            )
            _sklearn_available = True
        except ImportError:
            logger.warning(
                "sklearn not installed — precision/recall/F1 will not be computed. "
                "Install with: pip install scikit-learn"
            )
            _sklearn_available = False

        per_attr_results: Dict[str, Any] = {}
        acc_values:    List[float] = []
        weighted_sum:  float = 0.0
        total_samples: int   = 0

        for attr_name in per_attr_preds:
            preds = per_attr_preds[attr_name]
            lbls  = per_attr_labels.get(attr_name, [])
            n     = len(lbls)

            if n == 0:
                per_attr_results[attr_name] = {"accuracy": 0.0, "support": 0}
                continue

            correct  = sum(p == lb for p, lb in zip(preds, lbls))
            attr_acc = correct / n

            attr_result: Dict[str, Any] = {
                "accuracy": round(attr_acc, 4),
                "support":  n,
            }

            # Issue 10: per-class precision / recall / F1 + confusion matrix
            if _sklearn_available:
                # Determine label names for this attribute
                label_names = (label_maps or {}).get(attr_name, None)

                report = classification_report(
                    lbls, preds,
                    labels=list(range(len(label_names))) if label_names else None,
                    target_names=label_names,
                    output_dict=True,
                    zero_division=0,
                )

                attr_result["precision_macro"]    = round(report["macro avg"]["precision"],    4)
                attr_result["recall_macro"]        = round(report["macro avg"]["recall"],        4)
                attr_result["f1_macro"]            = round(report["macro avg"]["f1-score"],      4)
                attr_result["precision_weighted"]  = round(report["weighted avg"]["precision"],  4)
                attr_result["recall_weighted"]     = round(report["weighted avg"]["recall"],     4)
                attr_result["f1_weighted"]         = round(report["weighted avg"]["f1-score"],   4)

                # Per-class breakdown
                per_class: Dict[str, Any] = {}
                skip_keys = {"accuracy", "macro avg", "weighted avg"}
                for class_key, class_metrics in report.items():
                    if class_key in skip_keys:
                        continue
                    if isinstance(class_metrics, dict):
                        per_class[str(class_key)] = {
                            "precision": round(class_metrics["precision"], 4),
                            "recall":    round(class_metrics["recall"],    4),
                            "f1":        round(class_metrics["f1-score"],  4),
                            "support":   int(class_metrics["support"]),
                        }
                attr_result["per_class"] = per_class

                # Confusion matrix
                cm = confusion_matrix(lbls, preds)
                attr_result["confusion_matrix"] = cm.tolist()

            per_attr_results[attr_name] = attr_result
            acc_values.append(attr_acc)

            # Accumulate for weighted average (Issue 9)
            weighted_sum  += attr_acc * n
            total_samples += n

        # Issue 9: both macro and weighted accuracy
        macro_avg    = sum(acc_values) / len(acc_values) if acc_values else 0.0
        weighted_avg = weighted_sum / total_samples if total_samples > 0 else 0.0

        metrics["accuracy_macro"]    = round(macro_avg,    4)
        metrics["accuracy_weighted"] = round(weighted_avg, 4)
        # Keep "accuracy" as an alias for backward compatibility
        metrics["accuracy"]          = metrics["accuracy_macro"]
        metrics["total"]             = len(all_labels)
        metrics["per_attribute"]     = per_attr_results

        logger.info("Macro-average accuracy:    %.4f", macro_avg)
        logger.info("Weighted-average accuracy: %.4f", weighted_avg)
        for attr, res in per_attr_results.items():
            f1_str = (
                f", f1_weighted={res['f1_weighted']:.4f}"
                if "f1_weighted" in res else ""
            )
            logger.info(
                "  %-22s acc=%.4f%s (n=%d)",
                attr, res["accuracy"], f1_str, res["support"],
            )

    else:
        # Legacy single-attribute path
        total = len(all_labels)
        if total == 0:
            logger.warning("No labels provided; returning zero metrics")
            return {"accuracy": 0.0, "accuracy_macro": 0.0,
                    "accuracy_weighted": 0.0, "total": 0}
        correct = sum(p == lb for p, lb in zip(all_preds, all_labels))
        acc = round(correct / total, 4)
        metrics["accuracy"]          = acc
        metrics["accuracy_macro"]    = acc
        metrics["accuracy_weighted"] = acc
        metrics["total"]             = total

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_results(
    report_data: Dict[str, Any],
    output_dir: str,
    **kwargs: Any,
) -> None:
    """
    Save evaluation results to disk as a JSON file.

    Args:
        report_data: Evaluation results dict from compute_metrics.
        output_dir:  Directory to write results.json into.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / RESULTS_FILENAME
    with open(results_file, "w") as fh:
        json.dump(report_data, fh, indent=2)
    logger.info("Results saved to %s", results_file)
