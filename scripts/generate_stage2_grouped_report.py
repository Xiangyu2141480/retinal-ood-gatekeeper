#!/usr/bin/env python
"""Run and package the parent-grouped Stage 2 attribution evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retinal_ood.evaluation.report_tables import dataframe_to_markdown
from retinal_ood.reason_attribution.comparison import (
    ComparisonConfig,
    run_reason_attribution_method_comparison,
    select_best_method,
)


REQUIRED_METHODS = (
    "image_statistics_logreg",
    "global_feature_knn",
    "nearest_centroid",
    "logistic_regression",
    "linear_svm",
    "random_forest_or_gradient_boosting",
    "feature_statistics_fusion",
    "hierarchical_classifier",
)

COMPARISON_METRICS = (
    "selected family validation macro-F1",
    "family test accuracy",
    "family test macro-F1",
    "selected subtype validation macro-F1",
    "subtype test accuracy",
    "subtype test macro-F1",
)

FIGURE_FILENAMES = (
    "figure_grouped_family_confusion_matrix",
    "figure_grouped_subtype_confusion_matrix",
    "figure_grouped_stage2_method_comparison",
    "figure_legacy_vs_grouped_stage2_metrics",
)

CANONICAL_OUTPUTS = (
    "method_comparison.csv",
    "family_metrics.csv",
    "subtype_metrics.csv",
    "selected_models.json",
    "predictions_test.csv",
    "family_confusion_matrix.csv",
    "subtype_confusion_matrix.csv",
)

DISPLAY_METHODS = {
    "image_statistics_logreg": "Image statistics + LR",
    "global_feature_knn": "Global feature kNN",
    "nearest_centroid": "Nearest centroid",
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM",
    "random_forest_or_gradient_boosting": "Random forest",
    "feature_statistics_fusion": "Feature-statistics fusion",
    "hierarchical_classifier": "Hierarchical classifier",
}


class GroupedReportResult(NamedTuple):
    selected_family_method: str
    selected_subtype_method: str
    comparison_path: Path
    summary_path: Path
    figure_paths: tuple[Path, ...]


def snapshot_tree_hashes(directory: str | Path) -> dict[str, str]:
    """Return SHA-256 hashes for every file under a directory."""
    root = Path(directory)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def select_validation_winner(metrics: pd.DataFrame, metric_column: str) -> str:
    """Apply the project's validation macro-F1 and tie-break rule."""
    return select_best_method(metrics, split="val", metric_column=metric_column)


def run_grouped_evaluation(
    *,
    root_dir: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path,
    out_dir: str | Path,
    figures_dir: str | Path,
    seed: int = 42,
) -> None:
    """Run all required methods with grouped manifests and fixed settings."""
    image_root = Path(root_dir)
    if image_root.name != "data":
        raise ValueError(
            "root_dir must be the project data directory so images/... resolves to "
            "data/images/..."
        )
    run_reason_attribution_method_comparison(
        ComparisonConfig(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            root_dir=image_root,
            out_dir=out_dir,
            figures_dir=figures_dir,
            seed=seed,
            unknown_threshold=0.5,
            include_subtype=True,
            include_hierarchical=True,
            include_rbf_svm=False,
            method_names=REQUIRED_METHODS,
            global_image_size=24,
            statistics_image_size=224,
        )
    )
    selected_path = Path(out_dir) / "selected_models.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected["image_root"] = str(image_root)
    selected["image_path_resolution"] = (
        "Manifest image_path is resolved relative to root_dir: "
        "images/... -> data/images/..."
    )
    selected_path.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finalize_grouped_report(
    *,
    out_dir: str | Path,
    legacy_dir: str | Path,
    figures_dir: str | Path,
    overlap_summary: pd.DataFrame | None = None,
) -> GroupedReportResult:
    """Create comparisons, summary, and paper-ready figures from result files."""
    out_path = Path(out_dir)
    legacy_path = Path(legacy_dir)
    figure_path = Path(figures_dir)
    figure_path.mkdir(parents=True, exist_ok=True)
    _require_files(out_path, CANONICAL_OUTPUTS)

    legacy_hashes_before = snapshot_tree_hashes(legacy_path)
    family = pd.read_csv(out_path / "family_metrics.csv")
    subtype = pd.read_csv(out_path / "subtype_metrics.csv")
    method_overview = pd.read_csv(out_path / "method_comparison.csv")
    selected = json.loads((out_path / "selected_models.json").read_text(encoding="utf-8"))
    selected_family = select_validation_winner(family, "family_macro_f1")
    selected_subtype = select_validation_winner(subtype, "subtype_macro_f1")
    _validate_selected_models(selected, selected_family, selected_subtype)
    _validate_methods(method_overview)

    if overlap_summary is None:
        overlap_summary = pd.read_csv(out_path / "group_overlap_summary.csv")
    _validate_zero_overlap(overlap_summary)

    comparison = _build_legacy_comparison(
        family=family,
        subtype=subtype,
        selected_family=selected_family,
        selected_subtype=selected_subtype,
        legacy_dir=legacy_path,
    )
    comparison_path = out_path / "legacy_vs_grouped_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    family_detail = pd.read_csv(out_path / "per_family_f1_by_method.csv")
    subtype_detail = pd.read_csv(out_path / "per_subtype_f1_by_method.csv")
    hardest_family, hardest_family_f1 = _hardest_class(
        family_detail,
        method=selected_family,
        class_column="family",
    )
    hardest_subtype, hardest_subtype_f1 = _hardest_class(
        subtype_detail,
        method=selected_subtype,
        class_column="subtype",
    )

    summary_path = out_path / "summary.md"
    summary_path.write_text(
        _summary_markdown(
            family=family,
            subtype=subtype,
            comparison=comparison,
            selected_family=selected_family,
            selected_subtype=selected_subtype,
            hardest_family=hardest_family,
            hardest_family_f1=hardest_family_f1,
            hardest_subtype=hardest_subtype,
            hardest_subtype_f1=hardest_subtype_f1,
        ),
        encoding="utf-8",
    )

    generated_figures = _write_figures(
        out_dir=out_path,
        figures_dir=figure_path,
        family=family,
        subtype=subtype,
        comparison=comparison,
        selected_family=selected_family,
        selected_subtype=selected_subtype,
    )
    _write_grouped_figure_index(figure_path)
    if snapshot_tree_hashes(legacy_path) != legacy_hashes_before:
        raise RuntimeError("Legacy row-level result files changed during grouped reporting")

    return GroupedReportResult(
        selected_family_method=selected_family,
        selected_subtype_method=selected_subtype,
        comparison_path=comparison_path,
        summary_path=summary_path,
        figure_paths=tuple(generated_figures),
    )


def generate_stage2_grouped_report(
    *,
    root_dir: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path,
    legacy_dir: str | Path,
    out_dir: str | Path,
    figures_dir: str | Path,
    seed: int = 42,
) -> GroupedReportResult:
    """Run the grouped experiment and create its complete evidence package."""
    legacy_hashes = snapshot_tree_hashes(legacy_dir)
    start = perf_counter()
    run_grouped_evaluation(
        root_dir=root_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        test_manifest=test_manifest,
        out_dir=out_dir,
        figures_dir=figures_dir,
        seed=seed,
    )
    result = finalize_grouped_report(
        out_dir=out_dir,
        legacy_dir=legacy_dir,
        figures_dir=figures_dir,
    )
    legacy_hashes_after = snapshot_tree_hashes(legacy_dir)
    if legacy_hashes_after != legacy_hashes:
        raise RuntimeError("Legacy row-level result files changed during grouped evaluation")
    provenance = {
        "seed": int(seed),
        "unknown_threshold": 0.5,
        "global_image_size": 24,
        "statistics_image_size": 224,
        "required_methods": list(REQUIRED_METHODS),
        "image_root": str(Path(root_dir)),
        "image_path_resolution": "images/... -> data/images/...",
        "train_manifest": str(Path(train_manifest)),
        "validation_manifest": str(Path(val_manifest)),
        "test_manifest": str(Path(test_manifest)),
        "runtime_seconds": perf_counter() - start,
        "legacy_hashes_before": legacy_hashes,
        "legacy_hashes_after": legacy_hashes_after,
    }
    (Path(out_dir) / "generation_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _require_files(directory: Path, filenames: tuple[str, ...]) -> None:
    missing = [name for name in filenames if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing grouped outputs in {directory}: {missing}")


def _validate_selected_models(
    selected: dict[str, object],
    family_winner: str,
    subtype_winner: str,
) -> None:
    if selected.get("selected_family_method") != family_winner:
        raise ValueError("selected_models.json disagrees with family validation selection")
    if selected.get("selected_subtype_method") != subtype_winner:
        raise ValueError("selected_models.json disagrees with subtype validation selection")
    if selected.get("test_evaluation_started_after_selection") is not True:
        raise ValueError("Grouped results do not record validation-before-test evaluation")


def _validate_methods(overview: pd.DataFrame) -> None:
    methods = set(overview["method"].astype(str))
    required = set(REQUIRED_METHODS)
    if methods != required:
        raise ValueError(f"Grouped comparison method set mismatch: {sorted(methods ^ required)}")
    if "rbf_svm_optional" in methods:
        raise ValueError("Optional RBF SVM must not be included in the grouped comparison")


def _validate_zero_overlap(overlap: pd.DataFrame) -> None:
    required_scopes = {"train-val", "train-test", "val-test", "all-three"}
    if set(overlap["overlap_scope"].astype(str)) != required_scopes:
        raise ValueError("Grouped overlap audit is missing required split scopes")
    for column in ("image_path_overlap_count", "group_id_overlap_count"):
        if int(overlap[column].sum()) != 0:
            raise ValueError(f"Grouped split has non-zero overlap in {column}")


def _build_legacy_comparison(
    *,
    family: pd.DataFrame,
    subtype: pd.DataFrame,
    selected_family: str,
    selected_subtype: str,
    legacy_dir: Path,
) -> pd.DataFrame:
    legacy_family = pd.read_csv(legacy_dir / "family_metrics_by_method.csv")
    legacy_subtype = pd.read_csv(legacy_dir / "subtype_metrics_by_method.csv")
    legacy_family_method = select_validation_winner(legacy_family, "family_macro_f1")
    legacy_subtype_method = select_validation_winner(legacy_subtype, "subtype_macro_f1")

    legacy_values = (
        _metric(legacy_family, legacy_family_method, "val", "family_macro_f1"),
        _metric(legacy_family, legacy_family_method, "test", "family_accuracy"),
        _metric(legacy_family, legacy_family_method, "test", "family_macro_f1"),
        _metric(legacy_subtype, legacy_subtype_method, "val", "subtype_macro_f1"),
        _metric(legacy_subtype, legacy_subtype_method, "test", "subtype_accuracy"),
        _metric(legacy_subtype, legacy_subtype_method, "test", "subtype_macro_f1"),
    )
    grouped_values = (
        _metric(family, selected_family, "val", "family_macro_f1"),
        _metric(family, selected_family, "test", "family_accuracy"),
        _metric(family, selected_family, "test", "family_macro_f1"),
        _metric(subtype, selected_subtype, "val", "subtype_macro_f1"),
        _metric(subtype, selected_subtype, "test", "subtype_accuracy"),
        _metric(subtype, selected_subtype, "test", "subtype_macro_f1"),
    )
    return pd.DataFrame(
        {
            "metric": COMPARISON_METRICS,
            "legacy row-level split": legacy_values,
            "parent-grouped split": grouped_values,
            "difference": np.asarray(grouped_values) - np.asarray(legacy_values),
        }
    )


def _metric(dataframe: pd.DataFrame, method: str, split: str, column: str) -> float:
    rows = dataframe[
        (dataframe["method"].astype(str) == method)
        & (dataframe["split"].astype(str) == split)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {method}/{split}, found {len(rows)}")
    return float(rows.iloc[0][column])


def _hardest_class(
    dataframe: pd.DataFrame,
    *,
    method: str,
    class_column: str,
) -> tuple[str, float]:
    rows = dataframe[
        (dataframe["method"].astype(str) == method)
        & (dataframe["split"].astype(str) == "test")
    ].dropna(subset=["f1"])
    if rows.empty:
        raise ValueError(f"No selected-method test rows for {method}")
    minimum = float(rows["f1"].min())
    tied = rows[np.isclose(rows["f1"].astype(float), minimum, rtol=0.0, atol=1e-12)]
    labels = sorted(tied[class_column].astype(str))
    label = labels[0] if len(labels) == 1 else f"{', '.join(labels)} (tie)"
    return label, minimum


def _summary_markdown(
    *,
    family: pd.DataFrame,
    subtype: pd.DataFrame,
    comparison: pd.DataFrame,
    selected_family: str,
    selected_subtype: str,
    hardest_family: str,
    hardest_family_f1: float,
    hardest_subtype: str,
    hardest_subtype_f1: float,
) -> str:
    family_val = _metric(family, selected_family, "val", "family_macro_f1")
    family_accuracy = _metric(family, selected_family, "test", "family_accuracy")
    family_test = _metric(family, selected_family, "test", "family_macro_f1")
    subtype_val = _metric(subtype, selected_subtype, "val", "subtype_macro_f1")
    subtype_accuracy = _metric(subtype, selected_subtype, "test", "subtype_accuracy")
    subtype_test = _metric(subtype, selected_subtype, "test", "subtype_macro_f1")
    differences = {
        row["metric"]: float(row["difference"])
        for _, row in comparison.iterrows()
    }
    family_hardest_line = (
        f"No unique hardest family; all families tied at test F1 {hardest_family_f1:.4f}: "
        f"`{hardest_family.removesuffix(' (tie)')}`."
        if hardest_family.endswith(" (tie)")
        else (
            f"Hardest family: `{hardest_family}` "
            f"(test F1 {_format_metric(hardest_family_f1)})."
        )
    )
    return f"""# Parent-grouped Stage 2 evaluation

## Selection and results

- Family method selected by validation macro-F1: `{selected_family}` ({_format_metric(family_val)}).
- Grouped test family accuracy: {_format_metric(family_accuracy)}; macro-F1: {_format_metric(family_test)}.
- Subtype method selected by validation macro-F1: `{selected_subtype}` ({_format_metric(subtype_val)}).
- Grouped test subtype accuracy: {_format_metric(subtype_accuracy)}; macro-F1: {_format_metric(subtype_test)}.
- {family_hardest_line}
- Hardest subtype: `{hardest_subtype}` (test F1 {_format_metric(hardest_subtype_f1)}).

All candidate models were fitted on grouped training data. Model selection was frozen from
grouped validation macro-F1 before test evaluation began. The hierarchical classifier uses
non-oracle predicted-family routing at inference time. Stage 1 remains unchanged: it is the
ID-only unsupervised OOD gatekeeper, and no Stage 2 labels or scores are Stage 1 inputs.

## Legacy sensitivity comparison

- Family validation macro-F1 difference: {_format_signed_metric(differences[COMPARISON_METRICS[0]])}.
- Family test accuracy difference: {_format_signed_metric(differences[COMPARISON_METRICS[1]])}.
- Family test macro-F1 difference: {_format_signed_metric(differences[COMPARISON_METRICS[2]])}.
- Subtype validation macro-F1 difference: {_format_signed_metric(differences[COMPARISON_METRICS[3]])}.
- Subtype test accuracy difference: {_format_signed_metric(differences[COMPARISON_METRICS[4]])}.
- Subtype test macro-F1 difference: {_format_signed_metric(differences[COMPARISON_METRICS[5]])}.

The legacy row-level results are retained only as a sensitivity comparison. In the final split,
train-validation, train-test, validation-test, and all-three image-path and group-ID overlaps are
all zero.

## Scope and limitations

This optional supervised Stage 2 module explains a rejected input after Stage 1; it does not
decide rejection and is not a disease classifier. The grouped protocol eliminates cross-partition parent overlap;
variants remain dependent within a split because one parent contributes multiple transformed variants there.
The evaluation remains controlled, closed-set, synthetic-backed, and non-clinical. Reason labels are likely rejection
explanations, not clinical diagnoses. Parent grouping does not establish patient-independent or
device-independent clinical generalisation.
"""


def _format_metric(value: float) -> str:
    return format(float(value), ".16g")


def _format_signed_metric(value: float) -> str:
    return f"{float(value):+.16g}"


def _write_figures(
    *,
    out_dir: Path,
    figures_dir: Path,
    family: pd.DataFrame,
    subtype: pd.DataFrame,
    comparison: pd.DataFrame,
    selected_family: str,
    selected_subtype: str,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    paths: list[Path] = []
    paths.extend(
        _plot_confusion(
            pd.read_csv(out_dir / "family_confusion_matrix.csv", index_col=0),
            figures_dir / FIGURE_FILENAMES[0],
            title=f"Grouped family confusion matrix ({DISPLAY_METHODS[selected_family]})",
            label="Family",
        )
    )
    paths.extend(
        _plot_confusion(
            pd.read_csv(out_dir / "subtype_confusion_matrix.csv", index_col=0),
            figures_dir / FIGURE_FILENAMES[1],
            title=f"Grouped subtype confusion matrix ({DISPLAY_METHODS[selected_subtype]})",
            label="Subtype",
        )
    )
    paths.extend(_plot_method_comparison(family, subtype, figures_dir / FIGURE_FILENAMES[2]))
    paths.extend(_plot_legacy_comparison(comparison, figures_dir / FIGURE_FILENAMES[3]))
    return paths


def _write_grouped_figure_index(figures_dir: Path) -> Path:
    titles = {
        FIGURE_FILENAMES[0]: "Parent-grouped family confusion matrix",
        FIGURE_FILENAMES[1]: "Parent-grouped subtype confusion matrix",
        FIGURE_FILENAMES[2]: "Parent-grouped Stage 2 method comparison",
        FIGURE_FILENAMES[3]: "Legacy versus parent-grouped metric comparison",
    }
    rows = [
        {
            "path": f"{stem}.{extension}",
            "format": extension.upper(),
            "title": titles[stem],
            "source": "reports/stage2_grouped",
        }
        for stem in FIGURE_FILENAMES
        for extension in ("png", "pdf")
    ]
    index_path = figures_dir / "figure_index.md"
    index_path.write_text(
        "# Parent-grouped Stage 2 Figure Index\n\n"
        + dataframe_to_markdown(pd.DataFrame(rows))
        + "\n\nCompatibility figures generated by the comparison engine are retained in this directory, "
        "but the four PNG/PDF pairs above are the primary grouped dissertation figures.\n",
        encoding="utf-8",
    )
    return index_path


def _plot_confusion(
    matrix: pd.DataFrame,
    output_stem: Path,
    *,
    title: str,
    label: str,
) -> list[Path]:
    count = len(matrix)
    size = (7.2, 6.4) if count > 5 else (5.6, 4.7)
    figure, axis = plt.subplots(figsize=size, constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", interpolation="nearest")
    threshold = float(matrix.to_numpy().max()) * 0.55
    for row in range(count):
        for column in range(count):
            value = int(matrix.iloc[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=7 if count > 5 else 9,
                color="white" if value > threshold else "#202020",
            )
    if count <= 5:
        x_labels = [str(label).replace("_", "\n") for label in matrix.columns]
        x_rotation = 0
        x_alignment = "center"
    else:
        x_labels = [str(label).replace("_", " ") for label in matrix.columns]
        x_rotation = 45
        x_alignment = "right"
    y_labels = [str(label).replace("_", " ") for label in matrix.index]
    axis.set_xticks(range(count), x_labels, rotation=x_rotation, ha=x_alignment)
    axis.set_yticks(range(count), y_labels)
    axis.set_xlabel(f"Predicted {label.lower()}")
    axis.set_ylabel(f"True {label.lower()}")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.82, label="Images")
    return _save_figure(figure, output_stem)


def _plot_method_comparison(
    family: pd.DataFrame,
    subtype: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    methods = [method for method in REQUIRED_METHODS if method in set(family["method"])]
    positions = np.arange(len(methods), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.1), sharey=True)
    for axis, dataframe, metric, title in (
        (axes[0], family, "family_macro_f1", "Family attribution"),
        (axes[1], subtype, "subtype_macro_f1", "Subtype attribution"),
    ):
        val = [_metric(dataframe, method, "val", metric) for method in methods]
        test = [_metric(dataframe, method, "test", metric) for method in methods]
        axis.barh(positions + 0.18, val, height=0.34, color="#2878B5", label="Validation")
        axis.barh(positions - 0.18, test, height=0.34, color="#D95F02", label="Test")
        axis.set_yticks(positions, [DISPLAY_METHODS[method] for method in methods])
        axis.set_xlim(0.0, 1.02)
        axis.set_xlabel("Macro-F1")
        axis.set_title(title)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncols=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.suptitle("Parent-grouped Stage 2 method comparison", fontsize=11, y=0.97)
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.15, top=0.85, wspace=0.05)
    return _save_figure(figure, output_stem)


def _plot_legacy_comparison(comparison: pd.DataFrame, output_stem: Path) -> list[Path]:
    labels = [
        "Family val\nmacro-F1",
        "Family test\naccuracy",
        "Family test\nmacro-F1",
        "Subtype val\nmacro-F1",
        "Subtype test\naccuracy",
        "Subtype test\nmacro-F1",
    ]
    positions = np.arange(len(labels), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        comparison["legacy row-level split"],
        width,
        color="#8C8C8C",
        label="Legacy row-level",
    )
    axis.bar(
        positions + width / 2,
        comparison["parent-grouped split"],
        width,
        color="#2878B5",
        label="Parent-grouped",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Metric value")
    axis.set_title("Legacy versus parent-grouped Stage 2 evaluation")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=2, loc="lower left")
    return _save_figure(figure, output_stem)


def _save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, pdf_path]


def _build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    manifests = repository / "datasets" / "dissertation_v1" / "manifests"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        default=str(repository / "data"),
        help="Image root; manifest images/... paths resolve under this data directory.",
    )
    parser.add_argument(
        "--train-manifest",
        default=str(manifests / "reason_grouped_train.csv"),
    )
    parser.add_argument(
        "--val-manifest",
        default=str(manifests / "reason_grouped_val.csv"),
    )
    parser.add_argument(
        "--test-manifest",
        default=str(manifests / "reason_grouped_test.csv"),
    )
    parser.add_argument(
        "--legacy-dir",
        default=str(
            repository
            / "reports"
            / "dissertation_results"
            / "reason_attribution_method_comparison"
        ),
    )
    parser.add_argument("--out-dir", default=str(repository / "reports" / "stage2_grouped"))
    parser.add_argument(
        "--figures-dir",
        default=str(repository / "reports" / "dissertation_figures" / "stage2_grouped"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = generate_stage2_grouped_report(
        root_dir=args.root_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        legacy_dir=args.legacy_dir,
        out_dir=args.out_dir,
        figures_dir=args.figures_dir,
        seed=args.seed,
    )
    print(f"selected_family_method: {result.selected_family_method}")
    print(f"selected_subtype_method: {result.selected_subtype_method}")
    print(f"summary: {result.summary_path}")
    print(f"comparison: {result.comparison_path}")


if __name__ == "__main__":
    main()
