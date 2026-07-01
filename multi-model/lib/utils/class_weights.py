"""
Class weight computation for imbalanced datasets.

BUG-25 FIX: Without class weights the model over-predicts majority classes.
This module computes inverse-frequency weights from the training CSV and
returns them in the format expected by setup_training_components().

The formula used is sklearn's "balanced" strategy:
    weight[c] = total_samples / (num_classes * count[c])

This gives rare classes a higher weight and common classes a lower weight,
so CrossEntropyLoss treats each class equally in expectation.

Usage (automatic — called by build_training_pipeline when CLASS_WEIGHTS
is absent from config):

    from lib.utils.class_weights import compute_class_weights_from_csv
    weights = compute_class_weights_from_csv(csv_path, label_maps)
    # weights == {"theme": [1.2, 0.9, ...], "sentiment": [...], ...}
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import pandas as pd

logger = logging.getLogger(__name__)


def compute_class_weights_from_csv(
    csv_path: Path,
    label_maps: Dict[str, List[str]],
    smooth: float = 0.0,
) -> Dict[str, List[float]]:
    """
    Compute per-attribute inverse-frequency class weights from a CSV file.

    For each attribute that has a label map, counts how many times each
    class appears in the CSV and applies the balanced weighting formula:

        weight[c] = total / (num_classes * count[c])

    Classes that are missing from the CSV entirely (count == 0) receive
    weight 0.0 and a warning is logged — they cannot contribute to the
    loss regardless of weighting.

    Args:
        csv_path:   Path to the training CSV file.
        label_maps: Dict mapping attribute name → ordered list of label
                    strings (same order as the model's output logits).
        smooth:     Optional additive smoothing applied to counts before
                    computing weights.  A small value (e.g. 1.0) prevents
                    division by zero for very rare classes.  Defaults to 0.

    Returns:
        Dict mapping attribute name → list of float weights, one per class,
        in the same order as label_maps[attr].  Attributes with only one
        class are omitted (weighting has no effect).

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    total = len(df)
    result: Dict[str, List[float]] = {}

    for attr, labels in label_maps.items():
        if len(labels) <= 1:
            # Single-class attribute — weighting is meaningless
            continue
        if attr not in df.columns:
            logger.debug("Column '%s' not in CSV — skipping class weight computation", attr)
            continue

        counts = df[attr].value_counts()
        num_classes = len(labels)
        weights: List[float] = []

        for label in labels:
            count = counts.get(label, 0) + smooth
            if count == 0:
                logger.warning(
                    "Attribute '%s', class '%s' has 0 samples in training set. "
                    "Assigning weight 0.0 — consider removing this class or "
                    "adding more training data.",
                    attr, label,
                )
                weights.append(0.0)
            else:
                weights.append(float(total / (num_classes * count)))

        result[attr] = weights
        logger.info(
            "Class weights for '%s': %s",
            attr,
            {lbl: round(w, 4) for lbl, w in zip(labels, weights)},
        )

    return result


def compute_cooccurrence_prior(
    csv_path: Path,
    attr_names: List[str],
) -> torch.Tensor:
    """
    Compute pairwise attribute co-occurrence as an attention bias prior.

    For each pair of attributes, computes normalized mutual information
    (NMI) between the label assignments. NMI ∈ [0, 1] where 1 means the
    two attributes are perfectly correlated (knowing one tells you the
    other) and 0 means they are independent.

    The result is a symmetric (N_ATTRS, N_ATTRS) matrix used as an
    additive attention bias in AttributeCooccurrenceGraph — strongly
    correlated attributes attend to each other more.

    Args:
        csv_path:   Path to the training CSV file.
        attr_names: Ordered list of attribute names (node order).

    Returns:
        (N, N) float tensor with values in [0, 1]. Diagonal is 1.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n = len(attr_names)
    prior = torch.eye(n)

    for i, attr_i in enumerate(attr_names):
        for j, attr_j in enumerate(attr_names):
            if i >= j:
                continue
            if attr_i not in df.columns or attr_j not in df.columns:
                continue

            # Build contingency table
            crosstab = pd.crosstab(df[attr_i], df[attr_j])
            contingency = crosstab.values
            if contingency.size == 0:
                continue

            # Compute mutual information
            contingency = contingency.astype(float)
            total = contingency.sum()
            p_xy = contingency / total
            p_x = p_xy.sum(axis=1, keepdims=True)
            p_y = p_xy.sum(axis=0, keepdims=True)
            p_xy = np.maximum(p_xy, 1e-12)
            p_x = np.maximum(p_x, 1e-12)
            p_y = np.maximum(p_y, 1e-12)
            mi = (p_xy * (np.log(p_xy) - np.log(p_x) - np.log(p_y))).sum()

            # Normalize by joint entropy → NMI in [0, 1]
            h_x = -(p_x * np.log(np.maximum(p_x, 1e-12))).sum()
            h_y = -(p_y * np.log(np.maximum(p_y, 1e-12))).sum()
            nmi = 2 * mi / (h_x + h_y) if (h_x + h_y) > 0 else 0.0

            prior[i, j] = nmi
            prior[j, i] = nmi

    return prior


def log_class_distribution(
    csv_path: Path,
    label_maps: Dict[str, List[str]],
) -> None:
    """
    Log the class distribution for each attribute to INFO.

    Useful for diagnosing imbalance at training startup.

    Args:
        csv_path:   Path to the training CSV file.
        label_maps: Dict mapping attribute name → list of label strings.
    """
    if not Path(csv_path).exists():
        logger.warning("Cannot log class distribution — CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    total = len(df)

    for attr, labels in label_maps.items():
        if attr not in df.columns:
            continue
        counts = df[attr].value_counts()
        dist = {lbl: int(counts.get(lbl, 0)) for lbl in labels}
        pct  = {lbl: round(100 * dist[lbl] / total, 1) for lbl in labels}
        logger.info(
            "Class distribution — '%s': counts=%s  pct=%s",
            attr, dist, pct,
        )
