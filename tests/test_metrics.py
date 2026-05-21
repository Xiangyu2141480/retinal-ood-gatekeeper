import numpy as np

from retinal_ood.evaluation.metrics import compute_ood_metrics, confusion_at_threshold, fpr_at_recall


def test_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    result = compute_ood_metrics(y, s)
    assert result.auroc == 1.0
    assert result.auprc == 1.0
    assert result.fpr_at_95_tpr == 0.0


def test_confusion_at_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.7, 0.8, 0.9])
    cm = confusion_at_threshold(y, s, threshold=0.75)
    assert cm == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_fpr_at_recall_imperfect():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.7, 0.65, 0.8])
    assert 0.0 <= fpr_at_recall(y, s, 0.95) <= 1.0
