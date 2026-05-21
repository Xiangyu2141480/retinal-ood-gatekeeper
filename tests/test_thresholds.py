import numpy as np
import pytest

from retinal_ood.evaluation.thresholds import threshold_from_id_quantile


def test_threshold_from_id_quantile():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = threshold_from_id_quantile(scores, 0.5)
    assert 0.2 <= threshold <= 0.3


def test_threshold_from_id_quantile_rejects_empty_scores():
    with pytest.raises(ValueError, match="must not be empty"):
        threshold_from_id_quantile(np.array([]))


def test_threshold_from_id_quantile_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        threshold_from_id_quantile(np.array([0.1, np.inf]))


def test_threshold_from_id_quantile_rejects_invalid_quantile():
    with pytest.raises(ValueError, match="quantile"):
        threshold_from_id_quantile(np.array([0.1, 0.2]), 1.0)
