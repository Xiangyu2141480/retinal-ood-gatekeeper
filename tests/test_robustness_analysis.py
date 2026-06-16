from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retinal_ood.evaluation.bootstrap import bootstrap_metric_ci, metric_snapshot
from retinal_ood.evaluation.robustness import (
    artifact_severity_summary,
    deterministic_id_subset,
    method_disagreement_table,
    pca_projection,
    threshold_policy_sweep,
)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_path": "id_a.png", "label": 0, "ood_type": "id", "ood_subtype": "id", "score": 0.10},
            {"image_path": "id_b.png", "label": 0, "ood_type": "id", "ood_subtype": "id", "score": 0.20},
            {"image_path": "id_c.png", "label": 0, "ood_type": "id", "ood_subtype": "id", "score": 0.30},
            {
                "image_path": "ood_a.png",
                "label": 1,
                "ood_type": "sensory_artifact",
                "ood_subtype": "text_watermark",
                "score": 0.60,
            },
            {
                "image_path": "ood_b.png",
                "label": 1,
                "ood_type": "modality_shift",
                "ood_subtype": "colour_fundus",
                "score": 0.80,
            },
            {
                "image_path": "ood_c.png",
                "label": 1,
                "ood_type": "semantic_outlier",
                "ood_subtype": "cifar10_natural",
                "score": 0.90,
            },
        ]
    )


def test_metric_snapshot_reports_threshold_safety_rates():
    snapshot = metric_snapshot(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.7, 0.6, 0.9]),
        threshold=0.65,
    )

    assert snapshot["auroc"] == pytest.approx(0.75)
    assert snapshot["id_false_rejection_rate"] == pytest.approx(0.5)
    assert snapshot["ood_recall_at_threshold"] == pytest.approx(0.5)


def test_bootstrap_metric_ci_is_deterministic_and_contains_estimate():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

    first = bootstrap_metric_ci(labels, scores, threshold=0.5, n_bootstrap=25, seed=7)
    second = bootstrap_metric_ci(labels, scores, threshold=0.5, n_bootstrap=25, seed=7)

    assert first.equals(second)
    assert set(first["metric"]) >= {"auroc", "auprc", "fpr_at_95_tpr", "id_false_rejection_rate"}
    auroc = first[first["metric"] == "auroc"].iloc[0]
    assert auroc["estimate"] == pytest.approx(1.0)
    assert 0.0 <= auroc["ci_low"] <= auroc["ci_high"] <= 1.0


def test_threshold_policy_sweep_distinguishes_research_and_deployment_thresholds():
    table = threshold_policy_sweep(
        _scores(),
        val_id_scores=np.array([0.05, 0.10, 0.25, 0.40]),
        policies=["research_95_tpr", "val_id_quantile_95", "fixed_id_rejection_10"],
    )

    assert set(table["policy"]) == {"research_95_tpr", "val_id_quantile_95", "fixed_id_rejection_10"}
    assert set(table["policy_kind"]) == {"research", "deployment"}
    quantile_row = table[table["policy"] == "val_id_quantile_95"].iloc[0]
    assert quantile_row["threshold"] == pytest.approx(float(np.quantile([0.05, 0.10, 0.25, 0.40], 0.95)))
    assert "per_ood_type_recall" in table.columns


def test_deterministic_id_subset_uses_only_label_zero_rows():
    manifest = pd.DataFrame(
        {
            "image_path": [f"img_{idx}.png" for idx in range(6)],
            "label": [0, 1, 0, 0, 1, 0],
            "ood_type": ["id", "sensory_artifact", "id", "id", "semantic_outlier", "id"],
        }
    )

    subset = deterministic_id_subset(manifest, size=3, seed=123)
    subset_again = deterministic_id_subset(manifest, size=3, seed=123)

    assert len(subset) == 3
    assert subset["label"].eq(0).all()
    assert subset["ood_type"].eq("id").all()
    assert subset["image_path"].tolist() == subset_again["image_path"].tolist()


def test_artifact_severity_summary_reports_reject_rate_and_correlation():
    rows = []
    for severity, score in [(0.1, 0.2), (0.2, 0.4), (0.3, 0.8)]:
        rows.append(
            {
                "scheme": "mahalanobis_feature",
                "artifact_type": "gaussian_noise",
                "severity_value": severity,
                "score": score,
                "deployment_threshold": 0.5,
                "research_threshold": 0.3,
            }
        )
    summary = artifact_severity_summary(pd.DataFrame(rows))

    assert list(summary["severity_value"]) == [0.1, 0.2, 0.3]
    assert summary.loc[summary["severity_value"] == 0.3, "deployment_reject_rate"].iloc[0] == pytest.approx(1.0)
    assert summary["severity_score_spearman"].iloc[-1] == pytest.approx(1.0)


def test_method_disagreement_table_assigns_case_types():
    maha = _scores().assign(score=[0.1, 0.2, 0.7, 0.7, 0.8, 0.9])
    patch = _scores().assign(score=[0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    table = method_disagreement_table(
        {"mahalanobis_feature": maha, "patchcore_l3": patch},
        thresholds={"mahalanobis_feature": 0.5, "patchcore_l3": 0.5},
        primary_method="mahalanobis_feature",
        secondary_method="patchcore_l3",
    )

    id_false_positive = table[table["image_path"] == "id_c.png"].iloc[0]
    patch_wrong = table[table["image_path"] == "ood_a.png"].iloc[0]
    assert id_false_positive["case_type"] == "primary_false_positive_id"
    assert patch_wrong["case_type"] == "primary_correct_secondary_wrong"


def test_pca_projection_returns_two_components_with_metadata():
    features = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [2.0, 2.0, 0.0], [2.2, 2.1, 0.0]])
    metadata = pd.DataFrame(
        {
            "image_path": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "ood_type": ["id", "id", "semantic_outlier", "modality_shift"],
            "score": [0.1, 0.2, 0.8, 0.7],
        }
    )

    projection = pca_projection(features, metadata)

    assert list(projection.columns) == ["image_path", "label", "ood_type", "score", "pc1", "pc2"]
    assert len(projection) == 4
    assert projection[["pc1", "pc2"]].notna().all().all()


def test_generate_robustness_cli_parses_score_root():
    script_path = Path("scripts/generate_dissertation_robustness_analysis.py")
    spec = importlib.util.spec_from_file_location("generate_dissertation_robustness_analysis", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parsed = module._parse_score_root("primary=reports/generated/primary")

    assert parsed["name"] == "primary"
    assert parsed["path"] == Path("reports/generated/primary")
    with pytest.raises(ValueError, match="name=path"):
        module._parse_score_root("primary")

    assert module._display_artifact_label("text_watermark_opacity_0_10") == "text watermark opacity 0.10"


def test_generate_robustness_cli_help_starts_without_torch_dll_error():
    script_path = Path("scripts/generate_dissertation_robustness_analysis.py")

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--results-dir" in completed.stdout


def test_bootstrap_dissertation_metrics_cli_parser():
    script_path = Path("scripts/bootstrap_dissertation_metrics.py")
    spec = importlib.util.spec_from_file_location("bootstrap_dissertation_metrics", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    name, label, path = module._parse_scheme_score("maha=Mahalanobis=reports/scores.csv")

    assert name == "maha"
    assert label == "Mahalanobis"
    assert path == Path("reports/scores.csv")
    with pytest.raises(ValueError, match="name=label=path"):
        module._parse_scheme_score("maha=reports/scores.csv")


def test_bootstrap_dissertation_metrics_builds_table(tmp_path: Path):
    script_path = Path("scripts/bootstrap_dissertation_metrics.py")
    spec = importlib.util.spec_from_file_location("bootstrap_dissertation_metrics", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scores_path = tmp_path / "scores.csv"
    pd.DataFrame(
        {
            "label": [0, 0, 0, 1, 1, 1],
            "score": [0.1, 0.2, 0.3, 0.6, 0.8, 0.9],
        }
    ).to_csv(scores_path, index=False)

    table = module.build_bootstrap_table(
        {"toy": ("Toy method", scores_path)},
        bootstrap_samples=10,
        seed=123,
    )

    assert set(table["scheme"]) == {"toy"}
    assert set(table["metric"]) >= {"auroc", "auprc", "fpr_at_95_tpr"}
