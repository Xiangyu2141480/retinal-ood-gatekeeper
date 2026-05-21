import numpy as np

from retinal_ood.evaluation.thresholds import threshold_from_id_quantile


def test_threshold_from_id_quantile():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = threshold_from_id_quantile(scores, 0.5)
    assert 0.2 <= threshold <= 0.3
