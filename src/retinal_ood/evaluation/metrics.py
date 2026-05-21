"""Evaluation metrics for OOD detection.

Convention: label 1 is anomaly/OOD and higher scores mean more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


@dataclass(frozen=True)
class MetricResult:
    auroc: float
    auprc: float
    fpr_at_95_tpr: float


def _validate_binary_inputs(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_raw = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true_raw.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional arrays")
    if y_true_raw.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    if y_true_raw.size == 0:
        raise ValueError("y_true and scores must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    try:
        y_true_float = y_true_raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected binary labels 0/1") from exc
    if not np.isfinite(y_true_float).all():
        raise ValueError("labels must contain only finite values")
    if not np.isin(y_true_float, [0.0, 1.0]).all():
        labels = sorted(set(y_true_raw.astype(str).tolist()))
        raise ValueError(f"Expected binary labels 0/1, got {labels}")

    y_true = y_true_float.astype(int)
    labels = set(np.unique(y_true).tolist())
    if require_both_classes and len(labels) < 2:
        raise ValueError("Both ID label 0 and OOD label 1 are required for this metric")
    return y_true, scores


def fpr_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.95) -> float:
    """Compute false positive rate when OOD recall/TPR reaches target_recall.

    Returns the lowest FPR among thresholds with recall >= target_recall.
    """
    y_true, scores = _validate_binary_inputs(y_true, scores)
    if not np.isfinite(target_recall) or not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")

    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1, drop_intermediate=False)
    eligible_fpr = fpr[tpr >= target_recall]
    if eligible_fpr.size == 0:
        raise ValueError(f"Could not reach target_recall={target_recall}")
    return float(np.min(eligible_fpr))


def compute_ood_metrics(y_true: np.ndarray, scores: np.ndarray) -> MetricResult:
    """Compute standard global OOD metrics."""
    y_true, scores = _validate_binary_inputs(y_true, scores)
    return MetricResult(
        auroc=float(roc_auc_score(y_true, scores)),
        auprc=float(average_precision_score(y_true, scores)),
        fpr_at_95_tpr=fpr_at_recall(y_true, scores, 0.95),
    )


def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Return confusion matrix values at a fixed anomaly-score threshold."""
    y_true, scores = _validate_binary_inputs(y_true, scores, require_both_classes=False)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
