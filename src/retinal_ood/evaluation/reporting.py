"""Reporting helpers for OOD evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from retinal_ood.evaluation.metrics import (
    MetricResult,
    compute_ood_metrics,
    confusion_at_threshold,
)


def metric_result_to_dict(result: MetricResult) -> dict[str, float]:
    return {
        "auroc": result.auroc,
        "auprc": result.auprc,
        "fpr_at_95_tpr": result.fpr_at_95_tpr,
    }


def build_score_rows(
    metadata_rows: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Build the per-image scores table without logging patient identifiers."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(metadata_rows) != labels.shape[0] or labels.shape[0] != scores.shape[0]:
        raise ValueError("metadata_rows, labels, and scores must have the same length")

    rows: list[dict[str, Any]] = []
    for metadata, label, score in zip(metadata_rows, labels, scores):
        rows.append(
            {
                "image_path": metadata.get("image_path", ""),
                "label": int(label),
                "ood_type": metadata.get("ood_type", metadata.get("category", "unknown")),
                "score": float(score),
                "prediction": int(score >= threshold),
                "threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def compute_per_ood_type_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_types: list[str],
) -> dict[str, dict[str, float | int]]:
    """Compute one-vs-ID metrics for each OOD category."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape[0] != scores.shape[0] or labels.shape[0] != len(ood_types):
        raise ValueError("labels, scores, and ood_types must have the same length")

    id_mask = labels == 0
    results: dict[str, dict[str, float | int]] = {}
    for ood_type in sorted({ood_types[i] for i in range(len(ood_types)) if labels[i] == 1}):
        ood_mask = np.array([(label == 1 and current == ood_type) for label, current in zip(labels, ood_types)])
        subset_mask = id_mask | ood_mask
        subset_labels = labels[subset_mask]
        subset_scores = scores[subset_mask]
        metrics = compute_ood_metrics(subset_labels, subset_scores)
        results[ood_type] = {
            **metric_result_to_dict(metrics),
            "id_count": int(id_mask.sum()),
            "ood_count": int(ood_mask.sum()),
        }
    return results


def build_metrics_report(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_types: list[str],
    *,
    threshold: float,
    threshold_source: str,
) -> dict[str, Any]:
    """Build metrics.json content for a completed OOD evaluation run."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    global_metrics = compute_ood_metrics(labels, scores)
    return {
        "global": metric_result_to_dict(global_metrics),
        "confusion_matrix": confusion_at_threshold(labels, scores, threshold),
        "threshold": {
            "value": float(threshold),
            "source": threshold_source,
        },
        "counts": {
            "total": int(labels.shape[0]),
            "id": int((labels == 0).sum()),
            "ood": int((labels == 1).sum()),
        },
        "per_ood_type": compute_per_ood_type_metrics(labels, scores, ood_types),
    }


def save_scores_csv(path: str | Path, scores_df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(path, index=False)


def save_roc_pr_plots(labels: np.ndarray, scores: np.ndarray, out_dir: str | Path) -> None:
    """Save ROC and precision-recall plots for report inspection."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close()
