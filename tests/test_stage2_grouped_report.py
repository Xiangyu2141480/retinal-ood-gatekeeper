from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


REQUIRED_METHODS = {
    "image_statistics_logreg",
    "global_feature_knn",
    "nearest_centroid",
    "logistic_regression",
    "linear_svm",
    "random_forest_or_gradient_boosting",
    "feature_statistics_fusion",
    "hierarchical_classifier",
}


def _load_report_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_stage2_grouped_report.py"
    spec = importlib.util.spec_from_file_location("generate_stage2_grouped_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture_outputs(out_dir: Path, legacy_dir: Path) -> None:
    out_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    methods = sorted(REQUIRED_METHODS)
    complexity = {method: index + 1 for index, method in enumerate(methods)}
    complexity["linear_svm"] = 1
    complexity["hierarchical_classifier"] = 1

    family_rows: list[dict[str, object]] = []
    subtype_rows: list[dict[str, object]] = []
    overview_rows: list[dict[str, object]] = []
    for index, method in enumerate(methods):
        family_val = 0.70 + index * 0.01
        subtype_val = 0.50 + index * 0.01
        if method == "linear_svm":
            family_val = 0.96
        if method == "hierarchical_classifier":
            subtype_val = 0.84
        overview_rows.append({"method": method, "status": "ok"})
        for split, offset in (("val", 0.0), ("test", -0.02)):
            family_rows.append(
                {
                    "method": method,
                    "split": split,
                    "status": "ok",
                    "complexity": complexity[method],
                    "family_accuracy": family_val + offset + 0.01,
                    "family_macro_f1": family_val + offset,
                }
            )
            subtype_rows.append(
                {
                    "method": method,
                    "split": split,
                    "status": "ok",
                    "complexity": complexity[method],
                    "subtype_accuracy": subtype_val + offset + 0.01,
                    "subtype_macro_f1": subtype_val + offset,
                }
            )

    pd.DataFrame(overview_rows).to_csv(out_dir / "method_comparison.csv", index=False)
    pd.DataFrame(family_rows).to_csv(out_dir / "family_metrics.csv", index=False)
    pd.DataFrame(subtype_rows).to_csv(out_dir / "subtype_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "linear_svm",
                "split": "test",
                "family": family,
                "f1": score,
                "support": support,
            }
            for family, score, support in (
                ("modality_shift", 0.95, 80),
                ("sensory_artifact", 0.93, 240),
                ("semantic_outlier", 0.82, 100),
            )
        ]
    ).to_csv(out_dir / "per_family_f1_by_method.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "hierarchical_classifier",
                "split": "test",
                "subtype": subtype,
                "f1": score,
                "support": support,
            }
            for subtype, score, support in (
                ("colour_fundus", 0.96, 40),
                ("rectangle_annotation", 0.61, 30),
                ("cifar10_natural", 0.88, 100),
            )
        ]
    ).to_csv(out_dir / "per_subtype_f1_by_method.csv", index=False)
    pd.DataFrame(
        [[75, 3, 2], [2, 235, 3], [1, 4, 95]],
        index=["modality_shift", "sensory_artifact", "semantic_outlier"],
        columns=["modality_shift", "sensory_artifact", "semantic_outlier"],
    ).to_csv(out_dir / "family_confusion_matrix.csv")
    subtype_labels = [
        "colour_fundus",
        "oct_screenshot",
        "text_watermark",
        "rectangle_annotation",
        "arrow_annotation",
        "composite_layout",
        "blur_artifact",
        "border_crop",
        "gaussian_noise",
        "jpeg_compression",
        "cifar10_natural",
    ]
    subtype_matrix = pd.DataFrame(0, index=subtype_labels, columns=subtype_labels)
    subtype_support = [40, 40, 30, 30, 30, 30, 30, 30, 30, 30, 100]
    for label, support in zip(subtype_labels, subtype_support, strict=True):
        subtype_matrix.loc[label, label] = support
    subtype_matrix.to_csv(out_dir / "subtype_confusion_matrix.csv")
    pd.DataFrame(
        {
            "image_path": [f"image_{index}.png" for index in range(420)],
            "true_family": ["sensory_artifact"] * 420,
            "true_subtype": ["blur_artifact"] * 420,
        }
    ).to_csv(out_dir / "predictions_test.csv", index=False)
    (out_dir / "selected_models.json").write_text(
        json.dumps(
            {
                "selected_family_method": "linear_svm",
                "selected_subtype_method": "hierarchical_classifier",
                "selected_family_validation_value": 0.96,
                "selected_subtype_validation_value": 0.84,
                "selection_metric_family": "family_macro_f1",
                "selection_metric_subtype": "subtype_macro_f1",
                "test_evaluation_started_after_selection": True,
            }
        ),
        encoding="utf-8",
    )

    legacy_family = pd.DataFrame(
        [
            {
                "method": "linear_svm",
                "split": "val",
                "family_accuracy": 0.98,
                "family_macro_f1": 0.91,
            },
            {
                "method": "linear_svm",
                "split": "test",
                "family_accuracy": 0.92,
                "family_macro_f1": 0.90,
            },
        ]
    )
    legacy_subtype = pd.DataFrame(
        [
            {
                "method": "hierarchical_classifier",
                "split": "val",
                "subtype_accuracy": 0.82,
                "subtype_macro_f1": 0.78,
            },
            {
                "method": "hierarchical_classifier",
                "split": "test",
                "subtype_accuracy": 0.80,
                "subtype_macro_f1": 0.76,
            },
        ]
    )
    legacy_family.to_csv(legacy_dir / "family_metrics_by_method.csv", index=False)
    legacy_subtype.to_csv(legacy_dir / "subtype_metrics_by_method.csv", index=False)


def test_report_package_uses_generated_evidence_and_preserves_legacy(tmp_path: Path):
    module = _load_report_module()
    out_dir = tmp_path / "grouped"
    legacy_dir = tmp_path / "legacy"
    figures_dir = tmp_path / "figures"
    _write_fixture_outputs(out_dir, legacy_dir)
    hashes_before = module.snapshot_tree_hashes(legacy_dir)

    result = module.finalize_grouped_report(
        out_dir=out_dir,
        legacy_dir=legacy_dir,
        figures_dir=figures_dir,
        overlap_summary=pd.DataFrame(
            {
                "overlap_scope": ["train-val", "train-test", "val-test", "all-three"],
                "image_path_overlap_count": [0, 0, 0, 0],
                "group_id_overlap_count": [0, 0, 0, 0],
            }
        ),
    )

    assert module.snapshot_tree_hashes(legacy_dir) == hashes_before
    assert result.selected_family_method == "linear_svm"
    assert result.selected_subtype_method == "hierarchical_classifier"

    comparison = pd.read_csv(out_dir / "legacy_vs_grouped_comparison.csv")
    assert comparison.columns.tolist() == [
        "metric",
        "legacy row-level split",
        "parent-grouped split",
        "difference",
    ]
    assert comparison["metric"].tolist() == list(module.COMPARISON_METRICS)
    family_test = comparison.loc[comparison["metric"] == "family test macro-F1"].iloc[0]
    assert family_test["legacy row-level split"] == pytest.approx(0.90)
    assert family_test["parent-grouped split"] == pytest.approx(0.94)
    assert family_test["difference"] == pytest.approx(0.04)

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    for phrase in (
        "linear_svm",
        "hierarchical_classifier",
        "semantic_outlier",
        "rectangle_annotation",
        "validation",
        "predicted-family routing",
        "Stage 1 remains unchanged",
        "closed-set",
        "non-clinical",
        "eliminates cross-partition parent overlap",
        "variants remain dependent within a split",
    ):
        assert phrase in summary

    for figure in module.FIGURE_FILENAMES:
        png = figures_dir / f"{figure}.png"
        pdf = figures_dir / f"{figure}.pdf"
        assert png.stat().st_size > 1_000
        assert pdf.stat().st_size > 1_000

    figure_index = (figures_dir / "figure_index.md").read_text(encoding="utf-8")
    for figure in module.FIGURE_FILENAMES:
        assert f"{figure}.png" in figure_index
        assert f"{figure}.pdf" in figure_index


def test_hardest_class_reports_a_tie_without_arbitrary_single_winner():
    module = _load_report_module()
    metrics = pd.DataFrame(
        [
            {"method": "selected", "split": "test", "family": "alpha", "f1": 1.0},
            {"method": "selected", "split": "test", "family": "beta", "f1": 1.0},
            {"method": "other", "split": "test", "family": "gamma", "f1": 0.1},
        ]
    )

    label, score = module._hardest_class(
        metrics,
        method="selected",
        class_column="family",
    )

    assert label == "alpha, beta (tie)"
    assert score == pytest.approx(1.0)


def test_committed_grouped_outputs_are_complete_and_consistent():
    module = _load_report_module()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "reports" / "stage2_grouped"
    figures_dir = root / "reports" / "dissertation_figures" / "stage2_grouped"

    selected = json.loads((out_dir / "selected_models.json").read_text(encoding="utf-8"))
    family = pd.read_csv(out_dir / "family_metrics.csv")
    subtype = pd.read_csv(out_dir / "subtype_metrics.csv")
    methods = set(pd.read_csv(out_dir / "method_comparison.csv")["method"])

    assert methods == REQUIRED_METHODS
    assert "rbf_svm_optional" not in methods
    assert len(pd.read_csv(out_dir / "predictions_test.csv")) == 420
    assert int(pd.read_csv(out_dir / "family_confusion_matrix.csv", index_col=0).to_numpy().sum()) == 420
    assert int(pd.read_csv(out_dir / "subtype_confusion_matrix.csv", index_col=0).to_numpy().sum()) == 420

    family_winner = module.select_validation_winner(family, "family_macro_f1")
    subtype_winner = module.select_validation_winner(subtype, "subtype_macro_f1")
    assert selected["selected_family_method"] == family_winner
    assert selected["selected_subtype_method"] == subtype_winner
    assert selected["test_evaluation_started_after_selection"] is True

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert str(selected["selected_subtype_validation_value"]) in summary
    assert "No unique hardest family" in summary
    assert "modality_shift, semantic_outlier, sensory_artifact" in summary
    assert "eliminates cross-partition parent overlap" in summary
    assert "variants remain dependent within a split" in summary

    comparison = pd.read_csv(out_dir / "legacy_vs_grouped_comparison.csv")
    expected_differences = (
        comparison["parent-grouped split"] - comparison["legacy row-level split"]
    )
    assert comparison["difference"].tolist() == pytest.approx(
        expected_differences.tolist(),
        abs=1e-15,
    )

    for figure in module.FIGURE_FILENAMES:
        assert (figures_dir / f"{figure}.png").stat().st_size > 1_000
        assert (figures_dir / f"{figure}.pdf").stat().st_size > 1_000

    figure_index = (figures_dir / "figure_index.md").read_text(encoding="utf-8")
    for figure in module.FIGURE_FILENAMES:
        assert f"{figure}.png" in figure_index
        assert f"{figure}.pdf" in figure_index
