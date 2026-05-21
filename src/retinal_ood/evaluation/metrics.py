"""Evaluation metrics for OOD detection.

Convention: label 1 is anomaly/OOD and higher scores mean more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


@dataclass(frozen=True)
class MetricResult:
    auroc: float
    auprc: float
    fpr_at_95_tpr: float


def _validate_binary_inputs(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    labels = set(np.unique(y_true).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, got {labels}")
    if len(labels) < 2:
        raise ValueError("Both ID label 0 and OOD label 1 are required for this metric")
    return y_true, scores


def fpr_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.95) -> float:
    """Compute false positive rate when OOD recall/TPR reaches target_recall.

    Returns the lowest FPR among thresholds with recall >= target_recall.
    """
    y_true, scores = _validate_binary_inputs(y_true, scores)
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")

    thresholds = np.unique(scores)[::-1]
    best_fpr = 1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if recall >= target_recall:
            best_fpr = min(best_fpr, fpr)
    return float(best_fpr)


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
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
