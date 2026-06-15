"""Bootstrap confidence intervals for OOD gatekeeper metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retinal_ood.evaluation.metrics import compute_ood_metrics, confusion_at_threshold


def metric_snapshot(labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict[str, float]:
    """Compute headline OOD metrics and fixed-threshold safety rates."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    metrics = compute_ood_metrics(labels, scores)
    confusion = confusion_at_threshold(labels, scores, threshold)
    id_count = int((labels == 0).sum())
    ood_count = int((labels == 1).sum())
    return {
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "fpr_at_95_tpr": metrics.fpr_at_95_tpr,
        "id_false_rejection_rate": confusion["fp"] / id_count if id_count else float("nan"),
        "ood_recall_at_threshold": confusion["tp"] / ood_count if ood_count else float("nan"),
    }


def bootstrap_metric_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Return deterministic percentile bootstrap confidence intervals."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must be one-dimensional arrays with the same length")
    if labels.size == 0:
        raise ValueError("labels and scores must not be empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")

    estimate = metric_snapshot(labels, scores, threshold=threshold)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in estimate}
    attempts = 0
    max_attempts = max(n_bootstrap * 20, n_bootstrap + 100)
    while min(len(items) for items in values.values()) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, labels.shape[0], size=labels.shape[0])
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) < 2:
            continue
        sample = metric_snapshot(sample_labels, scores[indices], threshold=threshold)
        for metric, value in sample.items():
            if np.isfinite(value):
                values[metric].append(float(value))

    alpha = (1.0 - ci) / 2.0
    rows: list[dict[str, float | int | str]] = []
    for metric, estimate_value in estimate.items():
        samples = np.asarray(values[metric], dtype=float)
        if samples.size == 0:
            ci_low = ci_high = float("nan")
        else:
            ci_low, ci_high = np.quantile(samples, [alpha, 1.0 - alpha])
        rows.append(
            {
                "metric": metric,
                "estimate": float(estimate_value),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_bootstrap": int(samples.size),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows)
