"""Threshold calibration helpers."""

from __future__ import annotations

import numpy as np


def threshold_from_id_quantile(id_scores: np.ndarray, quantile: float = 0.95) -> float:
    """Set a deployment-like threshold from normal validation scores.

    If quantile=0.95, roughly 5% of validation ID scans would be rejected.
    """
    id_scores = np.asarray(id_scores).astype(float)
    if id_scores.ndim != 1:
        raise ValueError("id_scores must be a one-dimensional array")
    if id_scores.size == 0:
        raise ValueError("id_scores must not be empty")
    if not np.isfinite(id_scores).all():
        raise ValueError("id_scores must contain only finite values")
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    return float(np.quantile(id_scores, quantile))
