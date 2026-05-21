import numpy as np

from retinal_ood.evaluation.reporting import (
    build_metrics_report,
    build_score_rows,
    compute_per_ood_type_metrics,
)


def test_build_score_rows_omits_patient_identifiers():
    rows = [
        {
            "image_path": "toy/id.png",
            "ood_type": "id",
            "patient_id": "do-not-log",
        }
    ]

    scores_df = build_score_rows(rows, np.array([0]), np.array([0.1]), threshold=0.5)

    assert list(scores_df.columns) == ["image_path", "label", "ood_type", "score", "prediction", "threshold"]
    assert "patient_id" not in scores_df.columns


def test_compute_per_ood_type_metrics_one_vs_id():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    ood_types = ["id", "id", "modality_shift", "semantic_outlier"]

    metrics = compute_per_ood_type_metrics(labels, scores, ood_types)

    assert set(metrics) == {"modality_shift", "semantic_outlier"}
    assert metrics["modality_shift"]["auroc"] == 1.0
    assert metrics["semantic_outlier"]["auroc"] == 1.0
    assert metrics["modality_shift"]["id_count"] == 2
    assert metrics["modality_shift"]["ood_count"] == 1


def test_build_metrics_report_contains_global_confusion_and_categories():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    ood_types = ["id", "id", "modality_shift", "semantic_outlier"]

    report = build_metrics_report(
        labels,
        scores,
        ood_types,
        threshold=0.5,
        threshold_source="validation_id_quantile",
    )

    assert report["global"]["auroc"] == 1.0
    assert report["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert report["threshold"]["source"] == "validation_id_quantile"
    assert report["counts"] == {"total": 4, "id": 2, "ood": 2}
    assert set(report["per_ood_type"]) == {"modality_shift", "semantic_outlier"}
