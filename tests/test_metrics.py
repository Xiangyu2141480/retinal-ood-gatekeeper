import numpy as np
import pytest

from retinal_ood.evaluation.metrics import compute_ood_metrics, confusion_at_threshold, fpr_at_recall


def test_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    result = compute_ood_metrics(y, s)
    assert result.auroc == 1.0
    assert result.auprc == 1.0
    assert result.fpr_at_95_tpr == 0.0


def test_metrics_hand_calculated_imperfect_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    result = compute_ood_metrics(y, s)
    assert result.auroc == pytest.approx(0.75)
    assert result.auprc == pytest.approx(5 / 6)
    assert result.fpr_at_95_tpr == pytest.approx(0.5)


def test_fpr_at_recall_handles_tied_scores_conservatively():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.2, 0.8])
    assert fpr_at_recall(y, s, 0.95) == pytest.approx(0.5)


def test_confusion_at_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.7, 0.8, 0.9])
    cm = confusion_at_threshold(y, s, threshold=0.75)
    assert cm == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_fpr_at_recall_imperfect():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.7, 0.65, 0.8])
    assert 0.0 <= fpr_at_recall(y, s, 0.95) <= 1.0


def test_metrics_reject_single_class_inputs():
    y = np.array([0, 0, 0])
    s = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="Both ID label 0 and OOD label 1"):
        compute_ood_metrics(y, s)


def test_metrics_reject_non_binary_labels():
    y = np.array([0, 1, 2])
    s = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="Expected binary labels"):
        compute_ood_metrics(y, s)


def test_metrics_reject_non_finite_scores():
    y = np.array([0, 1])
    s = np.array([0.1, np.nan])
    with pytest.raises(ValueError, match="scores must contain only finite values"):
        compute_ood_metrics(y, s)


def test_metrics_reject_mismatched_lengths():
    y = np.array([0, 1])
    s = np.array([0.1])
    with pytest.raises(ValueError, match="same length"):
        compute_ood_metrics(y, s)


def test_fpr_at_recall_rejects_invalid_target():
    y = np.array([0, 1])
    s = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="target_recall"):
        fpr_at_recall(y, s, 0.0)


def test_confusion_at_threshold_accepts_single_class_for_operating_point():
    y = np.array([0, 0, 0])
    s = np.array([0.1, 0.3, 0.5])
    cm = confusion_at_threshold(y, s, threshold=0.4)
    assert cm == {"tn": 2, "fp": 1, "fn": 0, "tp": 0}
