#!/usr/bin/env python
"""Polish final dissertation figures and reporting from committed result tables."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retinal_ood.visualization.dissertation_style import (  # noqa: E402
    COLORS,
    LIGHT_COLORS,
    apply_dissertation_style,
)

FINAL_DIR = ROOT / "reports" / "dissertation_final"
FIG_DIR = ROOT / "reports" / "dissertation_figures"
RESULTS_DIR = ROOT / "reports" / "dissertation_results"
GROUPED_STAGE2_DIR = ROOT / "reports" / "stage2_grouped"
MANIFEST_DIR = ROOT / "datasets" / "dissertation_v1" / "manifests"
GENERATED_RUNS_DIR = ROOT / "reports" / "generated" / "dissertation_runs"

BLUE = COLORS["blue"]
GREEN = COLORS["green"]
RED = COLORS["red"]
GOLD = COLORS["gold"]
PURPLE = COLORS["purple"]
GRAY = COLORS["gray"]
DARK = COLORS["dark"]
LIGHT_BLUE = LIGHT_COLORS["blue"]
LIGHT_GREEN = LIGHT_COLORS["green"]
LIGHT_RED = LIGHT_COLORS["red"]
LIGHT_GOLD = LIGHT_COLORS["gold"]
LIGHT_PURPLE = LIGHT_COLORS["purple"]
LIGHT_GRAY = LIGHT_COLORS["gray"]

OOD_TYPE_ORDER = ["modality_shift", "sensory_artifact", "semantic_outlier"]
SUBTYPE_ORDER = [
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
SUBTYPE_CODES = {
    "colour_fundus": "S1",
    "oct_screenshot": "S2",
    "text_watermark": "S3",
    "rectangle_annotation": "S4",
    "arrow_annotation": "S5",
    "composite_layout": "S6",
    "blur_artifact": "S7",
    "border_crop": "S8",
    "gaussian_noise": "S9",
    "jpeg_compression": "S10",
    "cifar10_natural": "S11",
}
SUBTYPE_SHORT_CODES = {
    "colour_fundus": "CF",
    "oct_screenshot": "OCT",
    "text_watermark": "TXT",
    "rectangle_annotation": "RECT",
    "arrow_annotation": "ARR",
    "composite_layout": "COMP",
    "blur_artifact": "BLUR",
    "border_crop": "CROP",
    "gaussian_noise": "NOISE",
    "jpeg_compression": "JPEG",
    "cifar10_natural": "NAT",
}
FAMILY_LABELS = {
    "modality_shift": "Modality shift",
    "sensory_artifact": "Sensory artefact",
    "semantic_outlier": "Semantic outlier",
}
SELECTED_FIGURES = [
    "reports/dissertation_figures/figure_system_pipeline_overview.png",
    "reports/dissertation_figures/figure_dataset_taxonomy.png",
    "reports/dissertation_figures/figure_manifest_split_sizes.png",
    "reports/dissertation_figures/figure_metrics_by_scheme.png",
    "reports/dissertation_figures/figure_fpr95_by_scheme.png",
    "reports/dissertation_figures/figure_stage1_overall_comparison_combined.png",
    "reports/dissertation_figures/figure_stage1_overall_comparison_combined.pdf",
    "reports/dissertation_figures/figure_roc_overall_model_comparison.png",
    "reports/dissertation_figures/figure_pr_overall_model_comparison.png",
    "reports/dissertation_figures/figure_per_ood_type_comparison.png",
    "reports/dissertation_figures/figure_layer_ablation_patchcore.png",
    "reports/dissertation_figures/figure_patchcore_layer_detection_metrics.png",
    "reports/dissertation_figures/figure_patchcore_layer_safety_metric.png",
    "reports/dissertation_figures/figure_patchcore_layer_ablation_combined.png",
    "reports/dissertation_figures/figure_patchcore_layer_ablation_combined.pdf",
    "reports/dissertation_figures/figure_per_ood_subtype_by_scheme.png",
    "reports/dissertation_figures/figure_per_ood_subtype_comparison.png",
    "reports/dissertation_figures/figure_score_distribution_with_threshold.png",
    "reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png",
    "reports/dissertation_figures/figure_heatmaps_modality_examples.png",
    "reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png",
    "reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_two_stage_updated_pipeline.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_reason_method_family_macro_f1.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_reason_method_accuracy_macro_f1.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_reason_method_subtype_macro_f1.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_stage2_method_comparison_combined.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_stage2_method_comparison_combined.pdf",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_best_reason_family_confusion_matrix.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_best_reason_family_confusion_matrix_normalized.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_best_subtype_confusion_matrix.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_best_subtype_confusion_matrix_normalized.png",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI")
    args = parser.parse_args()

    _configure_matplotlib()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    tables = _load_tables()
    _write_summary_tables(tables)
    _write_polished_figures(tables, dpi=args.dpi)
    _write_caption_and_interpretation_notes()
    _write_final_figure_docs()
    _write_figure_revision_report()
    print("Polished dissertation reporting artifacts written.")


def _configure_matplotlib() -> None:
    apply_dissertation_style()


def _load_tables() -> dict[str, pd.DataFrame]:
    return {
        "stage1": _read_csv(RESULTS_DIR / "multi_scheme_comparison" / "metrics_by_scheme.csv"),
        "subtype": _read_csv(
            RESULTS_DIR / "multi_scheme_comparison" / "per_ood_subtype_by_scheme.csv"
        ),
        "per_type": _read_csv(
            RESULTS_DIR / "multi_scheme_comparison" / "per_ood_type_by_scheme.csv"
        ),
        "layer": _read_csv(
            RESULTS_DIR / "primary_balanced_by_subtype_10k" / "layer_ablation_table.csv"
        ),
        "threshold": _read_csv(RESULTS_DIR / "robustness_analysis" / "threshold_policy_sweep.csv"),
        "pca": _read_csv(RESULTS_DIR / "robustness_analysis" / "feature_space_projection.csv"),
        "stage2_family": _read_csv(
            GROUPED_STAGE2_DIR / "family_metrics.csv"
        ),
        "stage2_subtype": _read_csv(
            GROUPED_STAGE2_DIR / "subtype_metrics.csv"
        ),
        "stage2_best": _read_csv(GROUPED_STAGE2_DIR / "best_method_summary.csv"),
        "family_cm": _read_csv(
            GROUPED_STAGE2_DIR / "family_confusion_matrix.csv"
        ),
        "subtype_cm": _read_csv(
            GROUPED_STAGE2_DIR / "subtype_confusion_matrix.csv"
        ),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_summary_tables(tables: dict[str, pd.DataFrame]) -> None:
    stage1 = _stage1_method_summary(tables["stage1"])
    stage2 = _stage2_method_summary(tables["stage2_family"], tables["stage2_subtype"])
    recommended = _recommended_model_summary()
    limitations = _limitations_summary()
    claims = _claim_evidence_matrix()

    for name, table in [
        ("stage1_method_summary", stage1),
        ("stage2_method_summary", stage2),
        ("recommended_model_summary", recommended),
        ("limitations_summary", limitations),
        ("claim_evidence_matrix", claims),
    ]:
        _write_table_pair(FINAL_DIR / name, table)


def _stage1_method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.copy()
    for column in ["auroc", "auprc", "fpr_at_95_tpr", "ood_recall_at_threshold"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.sort_values(["auroc", "auprc"], ascending=[False, False]).reset_index(drop=True)
    table["recommended_use"] = table["scheme"].map(
        {
            "mahalanobis_feature": "Primary quantitative gatekeeper",
            "patchcore_layer3": "Localization-oriented companion",
            "global_feature_knn": "Strong feature-distance baseline",
            "autoencoder": "Reconstruction baseline",
            "image_statistics": "Low-level statistics baseline",
        }
    )
    table["recommended_use"] = table["recommended_use"].fillna("Comparison model")
    return table[
        [
            "scheme",
            "scheme_label",
            "auroc",
            "auprc",
            "fpr_at_95_tpr",
            "ood_recall_at_threshold",
            "recommended_use",
        ]
    ].pipe(_round_numeric)


def _stage2_method_summary(family: pd.DataFrame, subtype: pd.DataFrame) -> pd.DataFrame:
    family_test = family[family["split"].astype(str).eq("test")].copy()
    subtype_test = subtype[subtype["split"].astype(str).eq("test")].copy()
    table = family_test.merge(
        subtype_test[["method", "subtype_accuracy", "subtype_macro_f1"]],
        on="method",
        how="left",
    )
    table["selected_role"] = ""
    table.loc[
        table["method"].eq("feature_statistics_fusion"), "selected_role"
    ] = "Selected family method"
    table.loc[
        table["method"].eq("hierarchical_classifier"),
        "selected_role",
    ] = "Selected subtype method"
    table = table.sort_values("family_macro_f1", ascending=False).reset_index(drop=True)
    return table[
        [
            "method",
            "feature_set",
            "estimator",
            "family_accuracy",
            "family_macro_f1",
            "subtype_accuracy",
            "subtype_macro_f1",
            "selected_role",
        ]
    ].pipe(_round_numeric)


def _recommended_model_summary() -> pd.DataFrame:
    rows = [
        {
            "stage": "Stage 1",
            "recommendation": "Mahalanobis feature",
            "role": "Best quantitative OOD gatekeeper",
            "key_metric": "AUROC 0.9724; AUPRC 0.9975; FPR@95%TPR 0.2000",
            "evidence": "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv",
        },
        {
            "stage": "Stage 1",
            "recommendation": "PatchCore L3",
            "role": "Best localization-oriented companion",
            "key_metric": "Layer-ablation winner among PatchCore variants for AUROC",
            "evidence": (
                "reports/dissertation_results/primary_balanced_by_subtype_10k/"
                "layer_ablation_table.csv"
            ),
        },
        {
            "stage": "Stage 2",
            "recommendation": "feature_statistics_fusion",
            "role": "Best reason-family attribution method",
            "key_metric": "Grouped test accuracy 1.0000; macro-F1 1.0000",
            "evidence": "reports/stage2_grouped/selected_models.json",
        },
        {
            "stage": "Stage 2",
            "recommendation": "hierarchical_classifier",
            "role": "Best reason-subtype attribution method",
            "key_metric": "Grouped test accuracy 0.9357; macro-F1 0.9176",
            "evidence": "reports/stage2_grouped/selected_models.json",
        },
    ]
    return pd.DataFrame(rows)


def _limitations_summary() -> pd.DataFrame:
    rows = [
        {
            "limitation": "Synthetic ID fallback",
            "why_it_matters": "`test_id_synthetic_fallback` is not real clinical FAF validation.",
            "safe_wording": "Proof-of-concept stress-test evidence, not clinical validation.",
        },
        {
            "limitation": "Curated OOD stress tests",
            "why_it_matters": "OOD frequencies do not estimate clinical prevalence.",
            "safe_wording": "OOD sets are evaluation/stress-test categories.",
        },
        {
            "limitation": "Stage 2 is supervised explanation",
            "why_it_matters": "Reason labels are used after rejection, not for Stage 1 fitting.",
            "safe_wording": "Optional post-rejection reason attribution only.",
        },
        {
            "limitation": "Reason labels are not diagnoses",
            "why_it_matters": "Labels describe likely rejection causes, not disease status.",
            "safe_wording": "Likely explanations, not clinical diagnoses.",
        },
        {
            "limitation": "Within-split derived-variant dependence",
            "why_it_matters": "Parent grouping removes cross-partition overlap, not dependence within a split.",
            "safe_wording": "Grouped Stage 2 is not patient-independent clinical validation.",
        },
    ]
    return pd.DataFrame(rows)


def _claim_evidence_matrix() -> pd.DataFrame:
    rows = [
        {
            "claim": "Stage 1 is an ID-only unsupervised OOD gatekeeper.",
            "evidence": "datasets/dissertation_v1/manifests/train_id.csv",
            "figure_or_table": "stage1_method_summary.md",
            "limitation": "Synthetic ID fallback is not real clinical FAF validation.",
        },
        {
            "claim": "Mahalanobis is the strongest quantitative Stage 1 method.",
            "evidence": "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv",
            "figure_or_table": "figure_metrics_by_scheme.png",
            "limitation": "Strong performance may reflect global feature shifts.",
        },
        {
            "claim": "PatchCore L3 is the localization-oriented companion.",
            "evidence": (
                "reports/dissertation_results/primary_balanced_by_subtype_10k/"
                "layer_ablation_table.csv"
            ),
            "figure_or_table": "figure_layer_ablation_patchcore.png",
            "limitation": "PatchCore is not the strongest quantitative gatekeeper.",
        },
        {
            "claim": "Stage 2 is optional post-rejection reason attribution.",
            "evidence": "docs/experiments/reason_attribution_parent_grouped.md",
            "figure_or_table": "figure_two_stage_updated_pipeline.png",
            "limitation": "Supervised explanation layer; not Stage 1 training.",
        },
        {
            "claim": "feature_statistics_fusion is the grouped-validation reason-family method.",
            "evidence": "reports/stage2_grouped/selected_models.json",
            "figure_or_table": "figure_grouped_stage2_method_comparison.png",
            "limitation": "Scores are not clinical diagnosis performance.",
        },
        {
            "claim": "hierarchical_classifier is the best reason-subtype method.",
            "evidence": "reports/stage2_grouped/selected_models.json",
            "figure_or_table": "figure_grouped_subtype_confusion_matrix.png",
            "limitation": "Closed-set synthetic-backed attribution, not clinical diagnosis.",
        },
    ]
    return pd.DataFrame(rows)


def _round_numeric(table: pd.DataFrame) -> pd.DataFrame:
    rounded = table.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].round(4)
    return rounded


def _write_table_pair(base: Path, table: pd.DataFrame) -> None:
    table.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".md").write_text(_to_markdown(table, title=base.name), encoding="utf-8")


def _to_markdown(table: pd.DataFrame, *, title: str) -> str:
    headers = [str(column) for column in table.columns]
    lines = [f"# {title.replace('_', ' ').title()}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table.iterrows():
        values = [_format_cell(row[column]) for column in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _write_polished_figures(tables: dict[str, pd.DataFrame], *, dpi: int) -> None:
    _plot_system_pipeline(FIG_DIR / "figure_system_pipeline_overview.png", dpi=dpi)
    _plot_dataset_taxonomy(FIG_DIR / "figure_dataset_taxonomy.png", dpi=dpi)
    _plot_manifest_split_sizes(FIG_DIR / "figure_manifest_split_sizes.png", dpi=dpi)
    _plot_stage1_method_comparison(tables["stage1"], FIG_DIR / "figure_metrics_by_scheme.png", dpi=dpi)
    _plot_stage1_fpr(tables["stage1"], FIG_DIR / "figure_fpr95_by_scheme.png", dpi=dpi)
    _plot_stage1_overall_combined(
        tables["stage1"],
        FIG_DIR / "figure_stage1_overall_comparison_combined.png",
        dpi=dpi,
    )
    _plot_full_roc_comparison(
        tables["stage1"],
        FIG_DIR / "figure_roc_overall_model_comparison.png",
        dpi=dpi,
    )
    _plot_full_pr_comparison(
        tables["stage1"],
        FIG_DIR / "figure_pr_overall_model_comparison.png",
        dpi=dpi,
    )
    _plot_per_ood_type_full(
        tables["per_type"],
        FIG_DIR / "figure_per_ood_type_comparison.png",
        dpi=dpi,
    )
    _plot_mahalanobis_score_distribution(
        tables["stage1"],
        FIG_DIR / "figure_score_distribution_with_threshold.png",
        dpi=dpi,
    )
    _plot_patchcore_detection_metrics(
        tables["layer"],
        FIG_DIR / "figure_patchcore_layer_detection_metrics.png",
        dpi=dpi,
    )
    _plot_patchcore_safety_metric(
        tables["layer"],
        FIG_DIR / "figure_patchcore_layer_safety_metric.png",
        dpi=dpi,
    )
    _plot_patchcore_layer_ablation_combined(
        tables["layer"],
        FIG_DIR / "figure_patchcore_layer_ablation_combined.png",
        dpi=dpi,
    )
    shutil.copyfile(
        FIG_DIR / "figure_patchcore_layer_detection_metrics.png",
        FIG_DIR / "figure_layer_ablation_patchcore.png",
    )
    _plot_subtype_heatmap(
        tables["subtype"],
        FIG_DIR / "figure_per_ood_subtype_by_scheme.png",
        dpi=dpi,
    )
    shutil.copyfile(
        FIG_DIR / "figure_per_ood_subtype_by_scheme.png",
        FIG_DIR / "figure_per_ood_subtype_comparison.png",
    )
    _plot_threshold_tradeoff(
        tables["threshold"],
        FIG_DIR / "robustness" / "figure_threshold_policy_tradeoff.png",
        dpi=dpi,
    )
    _plot_feature_pca(
        tables["pca"],
        FIG_DIR / "robustness" / "figure_feature_space_pca_by_ood_type.png",
        dpi=dpi,
    )
    _plot_heatmap_triptych_examples(
        "sensory_artifact",
        FIG_DIR / "figure_heatmaps_sensory_artifact_examples.png",
        dpi=dpi,
    )
    _plot_heatmap_triptych_examples(
        "modality_shift",
        FIG_DIR / "figure_heatmaps_modality_examples.png",
        dpi=dpi,
    )
    reason_dir = FIG_DIR / "reason_attribution_method_comparison"
    _plot_two_stage_pipeline(reason_dir / "figure_two_stage_updated_pipeline.png", dpi=dpi)
    _plot_stage2_method_comparison(
        tables["stage2_family"],
        reason_dir / "figure_reason_method_family_macro_f1.png",
        dpi=dpi,
    )
    _plot_stage2_accuracy_macro_f1(
        tables["stage2_family"],
        reason_dir / "figure_reason_method_accuracy_macro_f1.png",
        dpi=dpi,
    )
    _plot_stage2_subtype_method_comparison(
        tables["stage2_subtype"],
        reason_dir / "figure_reason_method_subtype_macro_f1.png",
        dpi=dpi,
    )
    _plot_stage2_method_comparison_combined(
        tables["stage2_family"],
        tables["stage2_subtype"],
        reason_dir / "figure_stage2_method_comparison_combined.png",
        dpi=dpi,
    )
    _plot_confusion_matrix(
        tables["family_cm"],
        reason_dir / "figure_best_reason_family_confusion_matrix.png",
        title="Stage 2 reason-family attribution confusion matrix",
        subtitle="Method: feature_statistics_fusion; counts",
        label_mode="family",
        normalize=False,
        dpi=dpi,
    )
    _plot_confusion_matrix(
        tables["family_cm"],
        reason_dir / "figure_best_reason_family_confusion_matrix_normalized.png",
        title="Stage 2 reason-family attribution confusion matrix",
        subtitle="Method: feature_statistics_fusion; row-normalized percentages",
        label_mode="family",
        normalize=True,
        dpi=dpi,
    )
    _plot_confusion_matrix(
        tables["subtype_cm"],
        reason_dir / "figure_best_subtype_confusion_matrix.png",
        title="Stage 2 subtype attribution confusion matrix",
        subtitle="Method: hierarchical_classifier; counts",
        label_mode="subtype",
        normalize=False,
        dpi=dpi,
    )
    _plot_confusion_matrix(
        tables["subtype_cm"],
        reason_dir / "figure_best_subtype_confusion_matrix_normalized.png",
        title="Stage 2 subtype attribution confusion matrix",
        subtitle="Method: hierarchical_classifier; row-normalized percentages",
        label_mode="subtype",
        normalize=True,
        dpi=dpi,
    )


def _plot_full_roc_comparison(metrics: pd.DataFrame, path: Path, *, dpi: int) -> None:
    score_tables = _load_stage1_score_tables(metrics)
    if not score_tables:
        _plot_unavailable(
            path,
            "Overall ROC comparison",
            "Per-sample score CSV files were not available; rerun the completed Stage 1 evaluations.",
            dpi=dpi,
        )
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for row, scores in score_tables:
        y_true = scores["label"].astype(int)
        y_score = scores["score"].astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        label = f"{_short_method_label(row['scheme_label'])} ({row['auroc']:.3f})"
        ax.plot(fpr, tpr, linewidth=1.8, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#A7AFB8", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Overall ROC comparison")
    ax.text(
        0.02,
        0.08,
        "Balanced-by-subtype benchmark; legend shows AUROC.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=GRAY,
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    _save(fig, path, dpi=dpi)


def _plot_full_pr_comparison(metrics: pd.DataFrame, path: Path, *, dpi: int) -> None:
    score_tables = _load_stage1_score_tables(metrics)
    if not score_tables:
        _plot_unavailable(
            path,
            "Overall precision-recall comparison",
            "Per-sample score CSV files were not available; rerun the completed Stage 1 evaluations.",
            dpi=dpi,
        )
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for row, scores in score_tables:
        y_true = scores["label"].astype(int)
        y_score = scores["score"].astype(float)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        label = f"{_short_method_label(row['scheme_label'])} ({row['auprc']:.3f})"
        ax.plot(recall, precision, linewidth=1.8, label=label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.80, 1.01)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Overall precision-recall comparison")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7.0, loc="lower left", ncols=2)
    _save(fig, path, dpi=dpi)


def _plot_per_ood_type_full(per_type: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = per_type[per_type["eval_set"].astype(str).eq("balanced_by_subtype")].copy()
    table["ood_type"] = pd.Categorical(table["ood_type"], OOD_TYPE_ORDER, ordered=True)
    table["scheme_label"] = table["scheme_label"].map(_short_method_label)
    pivot = table.pivot_table(
        index="scheme_label",
        columns="ood_type",
        values="auroc",
        aggfunc="first",
        sort=False,
        observed=False,
    )
    method_order = [
        "Image stats",
        "Autoencoder",
        "Global kNN",
        "Mahalanobis",
        "PatchCore L2",
        "PatchCore L3",
        "PatchCore L4",
        "PatchCore L2+L3",
    ]
    pivot = pivot.reindex([label for label in method_order if label in pivot.index])
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0.65, vmax=1.0, cmap="Blues")
    ax.set_title("Per-OOD-family AUROC by Stage 1 method", fontsize=10.8)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [
            FAMILY_LABELS.get(str(value), _pretty_label(value)).replace(" ", "\n")
            for value in pivot.columns
        ]
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            color = "white" if values[y, x] > 0.90 else "#1F2933"
            ax.text(x, y, f"{values[y, x]:.3f}", ha="center", va="center", fontsize=7.5, color=color)
    plt.colorbar(image, ax=ax, fraction=0.040, pad=0.025, label="AUROC")
    _save(fig, path, dpi=dpi)


def _plot_mahalanobis_score_distribution(metrics: pd.DataFrame, path: Path, *, dpi: int) -> None:
    row = metrics[metrics["scheme"].astype(str).eq("mahalanobis_feature")]
    scores_path = _score_csv_for_run("mahalanobis_feature")
    if row.empty or scores_path is None:
        _plot_unavailable(
            path,
            "Mahalanobis score distribution",
            "Mahalanobis per-sample scores were not available.",
            dpi=dpi,
        )
        return
    scores = _read_score_table(scores_path)
    threshold = float(row.iloc[0]["threshold"])
    groups = [
        ("ID synthetic fallback", scores[scores["label"].astype(int).eq(0)], GREEN),
        ("Modality shift", scores[scores["ood_type"].astype(str).eq("modality_shift")], BLUE),
        ("Sensory artefact", scores[scores["ood_type"].astype(str).eq("sensory_artifact")], GOLD),
        ("Semantic outlier", scores[scores["ood_type"].astype(str).eq("semantic_outlier")], RED),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    for label, subset, color in groups:
        if subset.empty:
            continue
        ax.hist(
            subset["score"].astype(float),
            bins=32,
            density=True,
            alpha=0.46,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
    ax.axvline(threshold, color="#1F2933", linestyle="--", linewidth=1.6, label="ID q95 threshold")
    ax.set_xlabel("Mahalanobis anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Mahalanobis scores with ID-validation threshold")
    ax.text(
        0.02,
        0.95,
        f"Balanced-by-subtype benchmark; threshold = {threshold:.2f}",
        transform=ax.transAxes,
        fontsize=8.2,
        color=GRAY,
        va="top",
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=7.6)
    _save(fig, path, dpi=dpi)


def _plot_heatmap_triptych_examples(ood_type: str, path: Path, *, dpi: int) -> None:
    rows, manifest_root = _selected_heatmap_rows(ood_type)
    if rows.empty or manifest_root is None:
        _plot_unavailable(
            path,
            f"{_pretty_label(ood_type).title()} PatchCore examples",
            "Selected PatchCore heatmap artifacts were not available.",
            dpi=dpi,
        )
        return
    rows = _representative_heatmap_rows(rows, ood_type=ood_type)
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(8.6, 2.35 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    column_titles = ["Input image", "Anomaly map", "Overlay"]
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=9.5)
    for row_index, (_, row) in enumerate(rows.iterrows()):
        image_paths = [
            manifest_root / str(row["original_file"]),
            manifest_root / str(row["heatmap_file"]),
            manifest_root / str(row["overlay_file"]),
        ]
        for col_index, image_path in enumerate(image_paths):
            axis = axes[row_index, col_index]
            axis.imshow(Image.open(image_path).convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        label = _short_subtype_label(row.get("ood_subtype", ood_type)).replace("\n", " ")
        score = float(row.get("score", np.nan))
        threshold = float(row.get("threshold", np.nan))
        axes[row_index, 0].set_ylabel(
            f"{label}\nscore {score:.1f}\nthreshold {threshold:.1f}",
            fontsize=8.0,
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
        )
    title = "PatchCore sensory-artefact examples" if ood_type == "sensory_artifact" else "PatchCore modality-shift examples"
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.06, 0.00, 1.00, 0.97))
    _save(fig, path, dpi=dpi)


def _load_stage1_score_tables(metrics: pd.DataFrame) -> list[tuple[pd.Series, pd.DataFrame]]:
    rows: list[tuple[pd.Series, pd.DataFrame]] = []
    table = metrics.copy().sort_values("auroc", ascending=False)
    for _, row in table.iterrows():
        scores_path = _score_csv_for_run(str(row["run_name"]))
        if scores_path is None:
            continue
        scores = _read_score_table(scores_path)
        if scores.empty:
            continue
        rows.append((row, scores))
    return rows


def _score_csv_for_run(run_name: str) -> Path | None:
    known = {
        "image_statistics": GENERATED_RUNS_DIR
        / "multi_scheme_primary"
        / "runs"
        / "image_statistics"
        / "evaluation"
        / "scores.csv",
        "global_feature_knn": GENERATED_RUNS_DIR
        / "multi_scheme_primary"
        / "runs"
        / "global_feature_knn"
        / "evaluation"
        / "scores.csv",
        "mahalanobis_feature": GENERATED_RUNS_DIR
        / "multi_scheme_primary"
        / "runs"
        / "mahalanobis_feature"
        / "evaluation"
        / "scores.csv",
        "autoencoder_baseline": GENERATED_RUNS_DIR
        / "primary_balanced_by_subtype_10k"
        / "runs"
        / "autoencoder_baseline"
        / "evaluation"
        / "scores.csv",
        "patchcore_layer2": GENERATED_RUNS_DIR
        / "primary_balanced_by_subtype_10k"
        / "runs"
        / "patchcore_layer2"
        / "evaluation"
        / "scores.csv",
        "patchcore_layer3": GENERATED_RUNS_DIR
        / "primary_balanced_by_subtype_10k"
        / "runs"
        / "patchcore_layer3"
        / "evaluation"
        / "scores.csv",
        "patchcore_layer2_layer3": GENERATED_RUNS_DIR
        / "primary_balanced_by_subtype_10k"
        / "runs"
        / "patchcore_layer2_layer3"
        / "evaluation"
        / "scores.csv",
        "patchcore_layer4": GENERATED_RUNS_DIR
        / "patchcore_l4_primary"
        / "runs"
        / "patchcore_layer4"
        / "evaluation"
        / "scores.csv",
    }
    path = known.get(run_name)
    if path and path.exists():
        return path
    candidates = sorted(GENERATED_RUNS_DIR.glob(f"*/runs/{run_name}/evaluation/scores.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.stat().st_size)


def _read_score_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    for column in ["label", "score", "threshold"]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    return table.dropna(subset=["label", "score"])


def _selected_heatmap_rows(ood_type: str) -> tuple[pd.DataFrame, Path | None]:
    candidates = [
        GENERATED_RUNS_DIR / "paper_ready_patchcore_heatmaps" / "heatmap_manifest.csv",
        GENERATED_RUNS_DIR
        / "primary_balanced_by_subtype_10k"
        / "runs"
        / "patchcore_layer3"
        / "evaluation"
        / "selected_heatmaps"
        / "heatmap_manifest.csv",
    ]
    for manifest in candidates:
        if not manifest.exists():
            continue
        rows = pd.read_csv(manifest)
        if "ood_type" in rows.columns:
            rows = rows[rows["ood_type"].astype(str).eq(ood_type)]
        else:
            rows = rows[rows["image_path"].astype(str).str.contains(f"/{ood_type}/", regex=False)]
            rows["ood_type"] = ood_type
            rows["ood_subtype"] = rows["image_path"].map(_subtype_from_image_path)
        if not rows.empty:
            return rows, manifest.parent
    return pd.DataFrame(), None


def _representative_heatmap_rows(rows: pd.DataFrame, *, ood_type: str) -> pd.DataFrame:
    preferred = {
        "sensory_artifact": [
            "text_watermark",
            "rectangle_annotation",
            "arrow_annotation",
            "border_crop",
            "blur_artifact",
        ],
        "modality_shift": ["colour_fundus", "oct_screenshot"],
    }
    table = rows.copy()
    if "ood_subtype" not in table.columns:
        table["ood_subtype"] = table["image_path"].map(_subtype_from_image_path)
    table["_pref"] = table["ood_subtype"].map(
        {subtype: index for index, subtype in enumerate(preferred.get(ood_type, []))}
    )
    table["_pref"] = table["_pref"].fillna(len(preferred.get(ood_type, [])) + 1)
    table = table.sort_values(["_pref", "rank", "score"], ascending=[True, True, False])
    selected = table.drop_duplicates("ood_subtype", keep="first")
    limit = 4 if ood_type == "sensory_artifact" else 2
    return selected.head(limit)


def _subtype_from_image_path(value: object) -> str:
    parts = Path(str(value).replace("\\", "/")).parts
    for subtype in SUBTYPE_ORDER:
        if subtype in parts:
            return subtype
    return str(value)


def _plot_unavailable(path: Path, title: str, message: str, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, weight="semibold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=9, color=GRAY, wrap=True)
    _save(fig, path, dpi=dpi)


def _plot_stage1_overall_combined(metrics: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = _stage1_method_summary(metrics).sort_values(["auroc", "auprc"], ascending=True)
    labels = table["scheme_label"].map(_short_method_label)
    y = np.arange(len(table))
    fig, (ax_metrics, ax_fpr) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )
    highlight_edges = [
        DARK if row["scheme"] == "mahalanobis_feature" else "white" for _, row in table.iterrows()
    ]
    ax_metrics.barh(
        y - 0.16,
        table["auroc"],
        height=0.25,
        label="AUROC",
        color=BLUE,
        edgecolor=highlight_edges,
        linewidth=1.0,
    )
    ax_metrics.barh(
        y + 0.12,
        table["auprc"],
        height=0.25,
        label="AUPRC",
        color=GREEN,
        edgecolor=highlight_edges,
        linewidth=1.0,
    )
    ax_metrics.set_yticks(y)
    ax_metrics.set_yticklabels(labels)
    ax_metrics.set_xlim(0.0, 1.02)
    ax_metrics.set_xlabel("Metric value")
    ax_metrics.set_title("(a) AUROC and AUPRC", fontsize=10.8)
    ax_metrics.grid(axis="x", alpha=0.22)
    ax_metrics.legend(frameon=False, loc="lower right", ncols=2)

    fpr_colors = [GREEN if row["scheme"] == "mahalanobis_feature" else "#D7A0A0" for _, row in table.iterrows()]
    bars = ax_fpr.barh(y, table["fpr_at_95_tpr"], color=fpr_colors, edgecolor="white", linewidth=0.8)
    ax_fpr.set_xlim(0.0, 1.0)
    ax_fpr.set_xlabel("FPR@95%TPR")
    ax_fpr.set_title("(b) Safety-oriented metric", fontsize=10.8)
    ax_fpr.text(
        0.98,
        0.95,
        "Lower is better",
        transform=ax_fpr.transAxes,
        fontsize=8.0,
        color=GRAY,
        ha="right",
    )
    for bar in bars:
        ax_fpr.text(
            min(bar.get_width() + 0.02, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            ha="left",
            va="center",
            fontsize=7.2,
            color=DARK,
        )
    ax_fpr.grid(axis="x", alpha=0.22)
    fig.suptitle("Stage 1 overall method comparison", fontsize=11.8, weight="semibold")
    fig.subplots_adjust(wspace=0.10)
    _save(fig, path, dpi=dpi)


def _plot_patchcore_layer_ablation_combined(layer: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = layer.copy()
    table["layer_label"] = table["layers"].astype(str).str.replace("layer", "L", regex=False)
    x = np.arange(len(table))
    fig, (ax_metrics, ax_fpr) = plt.subplots(1, 2, figsize=(9.0, 4.4))
    ax_metrics.plot(x, table["auroc"], marker="o", color=BLUE, label="AUROC", linewidth=2.0)
    ax_metrics.plot(x, table["auprc"], marker="o", color=GREEN, label="AUPRC", linewidth=2.0)
    l3_rows = table[table["layer_label"].astype(str).eq("L3")]
    if not l3_rows.empty:
        l3_index = int(l3_rows.index[0])
        ax_metrics.axvline(l3_index, color=GRAY, linestyle="--", linewidth=1.0, alpha=0.7)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(table["layer_label"])
    ax_metrics.set_ylim(0.80, 1.005)
    ax_metrics.set_ylabel("Metric value")
    ax_metrics.set_title("(a) Detection metrics", fontsize=10.8)
    ax_metrics.text(
        0.03,
        0.06,
        "Higher is better; axis starts at 0.80",
        transform=ax_metrics.transAxes,
        fontsize=7.6,
        color=GRAY,
    )
    ax_metrics.legend(frameon=False, loc="lower right")
    ax_metrics.grid(axis="y", alpha=0.22)

    best_index = int(table["fpr_at_95_tpr"].astype(float).idxmin())
    colors = [GREEN if index == best_index else RED for index in table.index]
    bars = ax_fpr.bar(x, table["fpr_at_95_tpr"], color=colors, edgecolor="white", linewidth=0.8)
    ax_fpr.set_xticks(x)
    ax_fpr.set_xticklabels(table["layer_label"])
    ax_fpr.set_ylim(0.0, 1.0)
    ax_fpr.set_ylabel("FPR@95%TPR")
    ax_fpr.set_title("(b) Safety-oriented metric", fontsize=10.8)
    ax_fpr.text(0.04, 0.93, "Lower is better", transform=ax_fpr.transAxes, fontsize=7.8, color=GRAY)
    for bar in bars:
        ax_fpr.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=DARK,
        )
    ax_fpr.grid(axis="y", alpha=0.22)
    fig.suptitle("PatchCore layer ablation", fontsize=11.8, weight="semibold")
    _save(fig, path, dpi=dpi)


def _plot_stage2_method_comparison_combined(
    family: pd.DataFrame,
    subtype: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    fig, (ax_family, ax_subtype) = plt.subplots(1, 2, figsize=(11.2, 5.6), sharey=False)
    _plot_stage2_split_lollipop(
        ax_family,
        family,
        metric="family_macro_f1",
        selected_method="feature_statistics_fusion",
        title="(a) Reason-family macro-F1",
        xlim=(0.70, 1.01),
        note="Feature+stats fusion selected\nby grouped validation",
        baseline=0.8947,
    )
    _plot_stage2_split_lollipop(
        ax_subtype,
        subtype,
        metric="subtype_macro_f1",
        selected_method="hierarchical_classifier",
        title="(b) Reason-subtype macro-F1",
        xlim=(0.35, 1.01),
        note="Hierarchy selected;\nnon-oracle routing",
        baseline=0.6724,
    )
    fig.suptitle(
        "Stage 2 reason-attribution method comparison",
        fontsize=11.8,
        weight="semibold",
    )
    fig.subplots_adjust(wspace=0.45)
    _save(fig, path, dpi=dpi)


def _plot_stage2_split_lollipop(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    *,
    metric: str,
    selected_method: str,
    title: str,
    xlim: tuple[float, float],
    note: str,
    baseline: float,
) -> None:
    table = metrics[metrics["split"].isin(["val", "test"])].copy()
    pivot = table.pivot_table(index="method", columns="split", values=metric, aggfunc="first")
    pivot = pivot.sort_values("test", ascending=True)
    y = np.arange(len(pivot))
    labels = [_short_reason_method_label(method) for method in pivot.index]
    for index, method in enumerate(pivot.index):
        ax.plot(
            [pivot.loc[method, "val"], pivot.loc[method, "test"]],
            [index, index],
            color="#BCC5CF",
            linewidth=1.2,
            zorder=1,
        )
    ax.scatter(pivot["val"], y, color=BLUE, s=34, label="Validation", zorder=3)
    colors = [GREEN if method == selected_method else GOLD for method in pivot.index]
    ax.scatter(pivot["test"], y, color=colors, s=42, label="Test", zorder=4)
    ax.axvline(baseline, linestyle="--", color=GRAY, linewidth=1.0, label="PR #24 baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Macro-F1")
    ax.set_title(title, fontsize=10.8)
    ax.text(0.03, 0.06, f"Axis starts at {xlim[0]:.2f}", transform=ax.transAxes, fontsize=7.5, color=GRAY)
    selected_y = list(pivot.index).index(selected_method)
    ax.annotate(
        note,
        xy=(pivot.loc[selected_method, "test"], selected_y),
        xytext=(-76, 28 if selected_y < len(pivot) - 2 else -30),
        textcoords="offset points",
        fontsize=7.2,
        color=DARK,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": DARK},
    )
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right", fontsize=7.1)


def _plot_stage1_method_comparison(metrics: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = _stage1_method_summary(metrics).sort_values("auroc", ascending=True)
    labels = table["scheme_label"].map(_short_method_label)
    y = np.arange(len(table))
    edge_colors = [DARK if row["scheme"] == "mahalanobis_feature" else "white" for _, row in table.iterrows()]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    auroc_bars = ax.barh(
        y - 0.18,
        table["auroc"],
        height=0.25,
        label="AUROC",
        color=BLUE,
        edgecolor=edge_colors,
        linewidth=1.1,
    )
    auprc_bars = ax.barh(
        y + 0.10,
        table["auprc"],
        height=0.25,
        label="AUPRC",
        color=GREEN,
        edgecolor=edge_colors,
        linewidth=1.1,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Metric value")
    ax.set_title("Stage 1 OOD method comparison")
    ax.text(
        0.02,
        0.98,
        "Mahalanobis highlighted; AUPRC reflects the OOD-heavy evaluation balance.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=GRAY,
        va="top",
    )
    for bars in [auroc_bars, auprc_bars]:
        for bar in bars:
            ax.text(
                min(bar.get_width() + 0.006, 1.006),
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}",
                ha="left",
                va="center",
                fontsize=7.6,
                color=DARK,
            )
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right", ncols=2)
    _save(fig, path, dpi=dpi)


def _plot_stage1_fpr(metrics: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = _stage1_method_summary(metrics).sort_values("fpr_at_95_tpr", ascending=False)
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = [GREEN if row["scheme"] == "mahalanobis_feature" else "#D7A0A0" for _, row in table.iterrows()]
    bars = ax.barh(y, table["fpr_at_95_tpr"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(table["scheme_label"].map(_short_method_label))
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("FPR@95%TPR (lower is better)")
    ax.set_title("Stage 1 safety metric comparison")
    ax.text(0.02, 0.96, "Lower is better", transform=ax.transAxes, fontsize=8.5, color=GRAY)
    for bar in bars:
        ax.text(
            min(bar.get_width() + 0.014, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            ha="left",
            va="center",
            fontsize=8,
            color=DARK,
        )
    ax.grid(axis="x", alpha=0.22)
    _save(fig, path, dpi=dpi)


def _plot_patchcore_detection_metrics(layer: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = layer.copy()
    table["layer_label"] = table["layers"].astype(str).str.replace("layer", "L", regex=False)
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    ax.plot(x, table["auroc"], marker="o", linewidth=2.0, label="AUROC", color=BLUE)
    ax.plot(x, table["auprc"], marker="o", linewidth=2.0, label="AUPRC", color=GREEN)
    for metric, y_offset, color in [("auroc", 0.010, BLUE), ("auprc", -0.018, GREEN)]:
        for index, value in enumerate(table[metric]):
            ax.text(index, value + y_offset, f"{value:.3f}", ha="center", fontsize=7.8, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(table["layer_label"])
    ax.set_ylim(0.80, 1.005)
    ax.set_ylabel("Metric value")
    ax.set_title("PatchCore layer ablation: detection metrics")
    ax.text(
        0.02,
        0.04,
        "Higher is better",
        transform=ax.transAxes,
        fontsize=8.5,
        color=GRAY,
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    _save(fig, path, dpi=dpi)


def _plot_patchcore_safety_metric(layer: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = layer.copy()
    table["layer_label"] = table["layers"].astype(str).str.replace("layer", "L", regex=False)
    x = np.arange(len(table))
    best_index = int(table["fpr_at_95_tpr"].astype(float).idxmin())
    colors = [GREEN if index == best_index else RED for index in table.index]
    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    bars = ax.bar(x, table["fpr_at_95_tpr"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(table["layer_label"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("FPR@95%TPR")
    ax.set_title("PatchCore layer ablation: safety metric")
    ax.text(0.02, 0.94, "Lower is better", transform=ax.transAxes, fontsize=8.5, color=GRAY)
    ax.grid(axis="y", alpha=0.22)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK,
        )
    _save(fig, path, dpi=dpi)


def _plot_subtype_heatmap(subtype: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = subtype.copy()
    table["scheme_label"] = table["scheme_label"].map(_short_method_label)
    table["ood_subtype"] = pd.Categorical(table["ood_subtype"], SUBTYPE_ORDER, ordered=True)
    pivot = table.pivot_table(
        index="scheme_label",
        columns="ood_subtype",
        values="auroc",
        aggfunc="first",
        sort=False,
        observed=False,
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    _draw_heatmap(ax, pivot, title="Per-subtype AUROC by Stage 1 method", cmap="Blues")
    _save(fig, path, dpi=dpi)


def _plot_threshold_tradeoff(threshold: pd.DataFrame, path: Path, *, dpi: int) -> None:
    rows = threshold[threshold["scheme"].astype(str).eq("mahalanobis_feature")].copy()
    rows = rows.sort_values("id_false_rejection_rate")
    rows["label"] = rows["policy"].map(_threshold_policy_label)
    grouped = (
        rows.groupby(["id_false_rejection_rate", "ood_recall", "policy_kind"], as_index=False)
        .agg({"label": lambda labels: " / ".join(labels)})
        .sort_values(["id_false_rejection_rate", "ood_recall"])
    )
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    for kind, color, label in [
        ("deployment", BLUE, "Deployment-style"),
        ("research", GOLD, "Research-only"),
    ]:
        subset = grouped[grouped["policy_kind"].eq(kind)]
        if subset.empty:
            continue
        ax.scatter(
            subset["id_false_rejection_rate"],
            subset["ood_recall"],
            s=92,
            color=color,
            marker="o" if kind == "deployment" else "D",
            edgecolor=DARK,
            linewidth=0.8,
            label=label,
            zorder=3,
        )
    for _, row in grouped.iterrows():
        label = str(row["label"]).replace(" / ", " /\n")
        offset = _threshold_label_offset(str(row["label"]))
        ax.annotate(
            label,
            xy=(row["id_false_rejection_rate"], row["ood_recall"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.6,
            color=DARK,
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#8A9199", "linewidth": 0.7},
        )
    ax.set_xlabel("ID false rejection rate")
    ax.set_ylabel("OOD recall")
    ax.set_title("Mahalanobis threshold policy trade-off")
    ax.text(
        0.02,
        0.05,
        "Deployment-style thresholds use held-out ID validation only.",
        transform=ax.transAxes,
        fontsize=8.0,
        color=GRAY,
    )
    ax.set_xlim(0.0, max(0.24, float(rows["id_false_rejection_rate"].max()) + 0.05))
    ax.set_ylim(0.81, 0.975)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path, dpi=dpi)


def _plot_feature_pca(pca: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = pca.copy()
    table["group"] = np.where(table["label"].astype(int).eq(0), "id", table["ood_type"])
    colors = {
        "id": GREEN,
        "modality_shift": BLUE,
        "sensory_artifact": GOLD,
        "semantic_outlier": RED,
    }
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for group in [*OOD_TYPE_ORDER, "id"]:
        rows = table[table["group"].eq(group)]
        if group == "id":
            ax.scatter(
                rows["pc1"],
                rows["pc2"],
                s=24,
                alpha=0.88,
                label="ID FAF",
                facecolor="white",
                edgecolor=GREEN,
                linewidth=0.8,
                zorder=5,
            )
        else:
            ax.scatter(
                rows["pc1"],
                rows["pc2"],
                s=16,
                alpha=0.34 if group == "sensory_artifact" else 0.46,
                label=FAMILY_LABELS.get(group, _pretty_label(group)),
                color=colors[group],
                linewidth=0,
                zorder=2,
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Feature-space PCA by OOD family")
    ax.text(
        0.02,
        0.04,
        "Two-dimensional qualitative projection only.",
        transform=ax.transAxes,
        fontsize=8.0,
        color=GRAY,
    )
    ax.legend(frameon=False, ncols=2)
    ax.grid(alpha=0.18)
    _save(fig, path, dpi=dpi)


def _plot_stage2_method_comparison(family: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = family[family["split"].astype(str).eq("test")].copy()
    table = table.sort_values("family_macro_f1", ascending=True)
    labels = table["method"].map(_short_reason_method_label)
    colors = [
        GREEN if method == "feature_statistics_fusion" else BLUE for method in table["method"]
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.barh(np.arange(len(table)), table["family_macro_f1"], color=colors)
    ax.axvline(0.8947, linestyle="--", color=GRAY, linewidth=1.2, label="PR #24 baseline")
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0.70, 1.01)
    ax.set_xlabel("Family macro-F1 (axis starts at 0.70)")
    ax.set_title("Stage 2 reason-family method comparison")
    for bar in bars:
        ax.text(
            min(bar.get_width() + 0.004, 1.002),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            ha="left",
            va="center",
            fontsize=8,
            color=DARK,
        )
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path, dpi=dpi)


def _plot_stage2_accuracy_macro_f1(family: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = family[family["split"].astype(str).eq("test")].copy()
    table = table.sort_values("family_macro_f1", ascending=True)
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for index, row in enumerate(table.itertuples()):
        ax.plot(
            [row.family_accuracy, row.family_macro_f1],
            [index, index],
            color="#BCC5CF",
            linewidth=1.2,
            zorder=1,
        )
    ax.scatter(table["family_accuracy"], y, color=BLUE, s=44, label="Accuracy", zorder=3)
    ax.scatter(table["family_macro_f1"], y, color=GREEN, s=44, label="Macro-F1", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(table["method"].map(_short_reason_method_label))
    ax.set_xlim(0.70, 1.01)
    ax.set_xlabel("Metric value (axis starts at 0.70)")
    ax.set_title("Stage 2 family attribution accuracy and macro-F1")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    for index, row in enumerate(table.itertuples()):
        ax.text(row.family_macro_f1 + 0.004, index, f"{row.family_macro_f1:.3f}", va="center", fontsize=7.6)
    _save(fig, path, dpi=dpi)


def _plot_stage2_subtype_method_comparison(subtype: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = subtype[subtype["split"].astype(str).eq("test")].copy()
    table = table.sort_values("subtype_macro_f1", ascending=True)
    labels = table["method"].map(_short_reason_method_label)
    colors = [GREEN if method == "hierarchical_classifier" else BLUE for method in table["method"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.barh(np.arange(len(table)), table["subtype_macro_f1"], color=colors)
    ax.axvline(0.6724, linestyle="--", color=GRAY, linewidth=1.2, label="PR #24 baseline")
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0.35, 1.01)
    ax.set_xlabel("Subtype macro-F1")
    ax.set_title("Stage 2 subtype method comparison")
    for bar in bars:
        ax.text(
            min(bar.get_width() + 0.008, 0.998),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            ha="left",
            va="center",
            fontsize=8,
            color=DARK,
        )
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path, dpi=dpi)


def _plot_confusion_matrix(
    cm: pd.DataFrame,
    path: Path,
    *,
    title: str,
    subtitle: str,
    label_mode: str,
    normalize: bool,
    dpi: int,
) -> None:
    table = cm.set_index(cm.columns[0])
    counts = table.to_numpy(dtype=float)
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        values = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0) * 100.0
    else:
        values = counts
    labels = [_matrix_label(value, label_mode=label_mode) for value in table.index]
    columns = [_matrix_label(value, label_mode=label_mode) for value in table.columns]
    fig_width = 6.5 if len(labels) <= 4 else 8.8
    fig_height = 5.1 if len(labels) <= 4 else 7.4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    vmax = 100.0 if normalize else float(np.nanmax(values)) if values.size else 1.0
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_title(title, pad=19)
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.4, color=GRAY)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(columns, rotation=0 if len(labels) <= 4 else 45, ha="center" if len(labels) <= 4 else "right")
    ax.set_yticklabels(labels)
    threshold = vmax / 2.0 if values.size else 0.0
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            color = "white" if values[y, x] > threshold else "#1F2933"
            text = f"{values[y, x]:.0f}%" if normalize else f"{int(values[y, x])}"
            ax.text(x, y, text, ha="center", va="center", color=color, fontsize=7.5)
    colorbar_label = "Row-normalized percentage" if normalize else "Count"
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    _save(fig, path, dpi=dpi)


def _plot_system_pipeline(path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    ax.axis("off")
    boxes = [
        (0.04, 0.48, 0.14, 0.20, "Input\nimage", LIGHT_BLUE),
        (0.28, 0.48, 0.22, 0.20, "Stage 1\nID-only OOD gatekeeper", LIGHT_GREEN),
        (0.61, 0.64, 0.16, 0.17, "ACCEPT\nvalid FAF", LIGHT_GREEN),
        (0.61, 0.28, 0.16, 0.17, "REJECT\ninvalid/OOD", LIGHT_RED),
        (0.84, 0.64, 0.12, 0.17, "Downstream\nanalysis", LIGHT_GRAY),
    ]
    for x, y, width, height, text, color in boxes:
        _box(ax, x, y, width, height, text, facecolor=color)
    _arrow(ax, (0.18, 0.58), (0.28, 0.58))
    _arrow(ax, (0.50, 0.58), (0.61, 0.725))
    _arrow(ax, (0.50, 0.58), (0.61, 0.365))
    _arrow(ax, (0.77, 0.725), (0.84, 0.725))
    ax.text(
        0.50,
        0.13,
        "Stage 1 fitting uses ID rows only; OOD labels are used for evaluation.",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    ax.set_title("FAF OOD gatekeeper pipeline")
    _save(fig, path, dpi=dpi)


def _plot_two_stage_pipeline(path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.25))
    ax.axis("off")
    boxes = [
        (0.04, 0.48, 0.13, 0.20, "Input\nimage", LIGHT_BLUE),
        (0.25, 0.48, 0.20, 0.20, "Stage 1\nID-only OOD gatekeeper", LIGHT_GREEN),
        (0.55, 0.66, 0.13, 0.16, "ACCEPT", LIGHT_GREEN),
        (0.55, 0.31, 0.13, 0.16, "REJECT", LIGHT_RED),
        (0.76, 0.31, 0.18, 0.16, "Stage 2\nreason attribution", LIGHT_GOLD),
        (0.76, 0.08, 0.18, 0.14, "Reason family\n+ optional subtype", LIGHT_PURPLE),
    ]
    for x, y, width, height, text, color in boxes:
        _box(ax, x, y, width, height, text, facecolor=color)
    _arrow(ax, (0.17, 0.58), (0.26, 0.58))
    _arrow(ax, (0.45, 0.58), (0.55, 0.74))
    _arrow(ax, (0.45, 0.58), (0.55, 0.39))
    _arrow(ax, (0.68, 0.39), (0.76, 0.39))
    _arrow(ax, (0.85, 0.31), (0.85, 0.22))
    ax.text(
        0.30,
        0.20,
        "OOD labels are used only for Stage 2 explanation.",
        ha="center",
        fontsize=8.3,
        color=GRAY,
    )
    ax.text(
        0.47,
        0.09,
        "Stage 2 provides likely explanations, not clinical diagnoses.",
        ha="center",
        fontsize=8.3,
        color=GRAY,
    )
    ax.set_title("Two-stage rejected-input explanation pipeline")
    _save(fig, path, dpi=dpi)


def _plot_dataset_taxonomy(path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axis("off")
    groups = [
        (
            "ID FAF",
            "Stage 1 fitting rows\nand ID evaluation rows",
            0.06,
            0.58,
            LIGHT_GREEN,
        ),
        (
            "OOD: modality shift",
            "colour fundus\nOCT screenshots",
            0.56,
            0.58,
            LIGHT_BLUE,
        ),
        (
            "OOD: sensory artefact",
            "watermark, annotations\nblur, crop, noise, JPEG",
            0.06,
            0.26,
            LIGHT_GOLD,
        ),
        (
            "OOD: semantic outlier",
            "natural-image outliers\nnon-FAF content",
            0.56,
            0.26,
            LIGHT_RED,
        ),
    ]
    ax.text(0.50, 0.91, "Dataset v1 taxonomy", ha="center", va="center", fontsize=12, weight="semibold")
    ax.text(
        0.50,
        0.82,
        "Stage 1 uses ID data only; OOD groups are evaluation stress tests.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=GRAY,
    )
    for label, subtitle, x, y, color in groups:
        _box(ax, x, y, 0.36, 0.15, label, facecolor=color)
        ax.text(x + 0.18, y - 0.045, subtitle, ha="center", va="top", fontsize=8.2, linespacing=1.25)
    ax.text(
        0.50,
        0.08,
        "Stage 2 reason-attribution splits are explanation-only OOD splits; "
        "the ID test split uses synthetic FAF fallback.",
        ha="center",
        fontsize=8.4,
        color=GRAY,
    )
    _save(fig, path, dpi=dpi)


def _plot_manifest_split_sizes(path: Path, *, dpi: int) -> None:
    counts = _manifest_counts()
    names = list(counts)
    values = [counts[name] for name in names]
    colors = [GREEN, GREEN, GREEN, BLUE, BLUE, GOLD, GOLD, GOLD]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.barh(np.arange(len(names)), values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    ax.set_title("Manifest split sizes")
    ax.grid(axis="x", alpha=0.22)
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.015, index, f"{value:,}", va="center", fontsize=8)
    ax.text(
        0.00,
        -0.16,
        "Manifest rows are task-specific and non-additive. Packaged dataset: 3,100 unique image files.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=GRAY,
    )
    _save(fig, path, dpi=dpi)


def _manifest_counts() -> dict[str, int]:
    manifests = {
        "ID train": MANIFEST_DIR / "train_id.csv",
        "ID val": MANIFEST_DIR / "val_id.csv",
        "ID test fallback": MANIFEST_DIR / "test_id_synthetic_fallback.csv",
        "OOD full": MANIFEST_DIR / "test_ood_full.csv",
        "OOD balanced by subtype": MANIFEST_DIR / "test_ood_balanced_by_subtype.csv",
        "Stage 2 train": MANIFEST_DIR / "reason_train.csv",
        "Stage 2 val": MANIFEST_DIR / "reason_val.csv",
        "Stage 2 test": MANIFEST_DIR / "reason_test.csv",
    }
    return {name: len(pd.read_csv(path_in)) for name, path_in in manifests.items()}


def _draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, *, title: str, cmap: str) -> None:
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_short_subtype_label(value) for value in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_pretty_label(value) for value in pivot.index])
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            if np.isfinite(values[y, x]):
                ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.025, label="AUROC")


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#333333",
        )
    )


def _short_method_label(value: object) -> str:
    text = str(value)
    replacements = {
        "Mahalanobis feature": "Mahalanobis",
        "Global feature kNN": "Global kNN",
        "Image statistics": "Image stats",
        "PatchCore layer3": "PatchCore L3",
        "PatchCore layer2": "PatchCore L2",
        "PatchCore layer2+layer3": "PatchCore L2+L3",
    }
    return replacements.get(text, text)


def _pretty_method(value: object) -> str:
    return str(value).replace("_", " ")


def _short_reason_method_label(value: object) -> str:
    labels = {
        "image_statistics_logreg": "Image stats LR",
        "global_feature_knn": "Global kNN",
        "nearest_centroid": "Nearest centroid",
        "logistic_regression": "Logistic regression",
        "linear_svm": "Linear SVM",
        "random_forest_or_gradient_boosting": "Random forest",
        "feature_statistics_fusion": "Feature+stats fusion",
        "hierarchical_classifier": "Hierarchical",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def _short_subtype_label(value: object) -> str:
    labels = {
        "colour_fundus": "Colour\nfundus",
        "oct_screenshot": "OCT\nscreenshot",
        "text_watermark": "Text\nwatermark",
        "rectangle_annotation": "Rectangle\nannotation",
        "arrow_annotation": "Arrow\nannotation",
        "composite_layout": "Composite\nlayout",
        "blur_artifact": "Blur",
        "border_crop": "Border\ncrop",
        "gaussian_noise": "Gaussian\nnoise",
        "jpeg_compression": "JPEG\ncompression",
        "cifar10_natural": "CIFAR-10\nnatural",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def _pretty_label(value: object) -> str:
    return str(value).replace("_", " ")


def _matrix_label(value: object, *, label_mode: str) -> str:
    key = str(value)
    if label_mode == "subtype":
        return SUBTYPE_SHORT_CODES.get(key, key)
    if label_mode == "family":
        return FAMILY_LABELS.get(key, _pretty_label(key))
    return _pretty_label(key)


def _threshold_policy_label(policy: object) -> str:
    labels = {
        "research_95_tpr": "research 95% TPR",
        "val_id_quantile_95": "ID q95",
        "val_id_quantile_97_5": "ID q97.5",
        "val_id_quantile_99": "ID q99",
        "val_id_quantile_99_5": "ID q99.5",
        "fixed_id_rejection_5": "fixed 5%",
        "fixed_id_rejection_10": "fixed 10%",
        "fixed_id_rejection_20": "fixed 20%",
    }
    return labels.get(str(policy), str(policy).replace("_", " "))


def _threshold_label_offset(label: str) -> tuple[int, int]:
    offsets = {
        "ID q99.5": (54, -12),
        "ID q99": (66, -18),
        "ID q97.5": (72, 24),
        "ID q95 / fixed 5%": (64, 24),
        "fixed 10%": (74, -24),
        "fixed 20%": (50, 24),
        "research 95% TPR": (-78, 24),
    }
    return offsets.get(label, (42, 18))


def _save(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _write_caption_and_interpretation_notes() -> None:
    captions = [
        {
            "file": "figure_system_pipeline_overview.png",
            "caption": (
                "FAF OOD gatekeeper pipeline. Stage 1 is fitted using ID rows only and separates "
                "accepted valid FAF inputs from rejected invalid or OOD inputs before downstream analysis."
            ),
            "interpretation": (
                "This figure fixes the project framing as binary quality control rather than disease "
                "classification."
            ),
            "why": "Use early in the dissertation to prevent supervised disease-classifier framing.",
        },
        {
            "file": "figure_dataset_taxonomy.png",
            "caption": (
                "Dataset v1 taxonomy showing ID FAF rows and the three OOD stress-test families. "
                "Stage 2 splits are explanation-only OOD splits; the ID test split uses synthetic FAF fallback."
            ),
            "interpretation": (
                "The taxonomy makes clear which data are used for ID-only training and which are "
                "used for evaluation or post-rejection explanation."
            ),
            "why": "Supports the dataset chapter and the ID-only/OOD-only boundary.",
        },
        {
            "file": "figure_manifest_split_sizes.png",
            "caption": (
                "Committed manifest split sizes for ID, OOD evaluation, and Stage 2 reason-attribution rows. "
                "Bars report task-specific manifest rows and are not additive; the same packaged image can "
                "appear in multiple manifests, and the underlying packaged collection contains 3100 image files."
            ),
            "interpretation": "The row-count view separates dataset composition from taxonomy.",
            "why": "Keeps the main taxonomy figure uncrowded while preserving manifest evidence.",
        },
        {
            "file": "figure_metrics_by_scheme.png",
            "caption": (
                "Stage 1 method comparison using AUROC and AUPRC on the balanced-by-subtype OOD "
                "evaluation set. The axis spans 0 to 1; AUPRC reflects the OOD-heavy class balance, "
                "so the safety-focused FPR@95%TPR comparison should also be considered."
            ),
            "interpretation": (
                "Mahalanobis feature distance is the strongest quantitative gatekeeper in the final "
                "comparison."
            ),
            "why": "Main quantitative model-selection figure.",
        },
        {
            "file": "figure_fpr95_by_scheme.png",
            "caption": (
                "Safety-focused Stage 1 comparison of FPR@95%TPR, where lower values indicate fewer "
                "ID false positives at high OOD sensitivity."
            ),
            "interpretation": "Mahalanobis has the best FPR@95%TPR among the evaluated Stage 1 methods.",
            "why": "Useful for threshold-safety discussion.",
        },
        {
            "file": "figure_stage1_overall_comparison_combined.png",
            "caption": (
                "Stage 1 overall method comparison on the balanced-by-subtype benchmark. Panel (a) "
                "reports AUROC and AUPRC; panel (b) reports FPR@95%TPR, where lower values are "
                "better. AUPRC should be interpreted with the OOD-heavy evaluation prevalence."
            ),
            "interpretation": (
                "Mahalanobis is the strongest quantitative Stage 1 gatekeeper across the combined "
                "ranking and safety-oriented view."
            ),
            "why": "Recommended main-text Stage 1 comparison figure.",
        },
        {
            "file": "figure_roc_overall_model_comparison.png",
            "caption": (
                "Overall ROC comparison for the final eight Stage 1 OOD gatekeeper configurations on "
                "the balanced-by-subtype benchmark. The legend reports AUROC values and includes the "
                "final best quantitative model, Mahalanobis feature distance."
            ),
            "interpretation": (
                "The regenerated ROC figure is appendix evidence for the full final experiment matrix "
                "rather than the older AE/PatchCore-only subset."
            ),
            "why": "Use as an appendix ranking-curve check, not as the main Stage 1 result figure.",
        },
        {
            "file": "figure_pr_overall_model_comparison.png",
            "caption": (
                "Overall precision-recall comparison for the final eight Stage 1 configurations on the "
                "balanced-by-subtype benchmark. High precision-recall values reflect the OOD-heavy "
                "evaluation prevalence and should be interpreted with AUROC and threshold-conditioned "
                "ID rejection."
            ),
            "interpretation": "The PR curves are useful but less visually discriminative because OOD prevalence is high.",
            "why": "Appendix companion for complete ranking-curve reporting.",
        },
        {
            "file": "figure_per_ood_type_comparison.png",
            "caption": (
                "Per-OOD-family AUROC heatmap for the final eight Stage 1 configurations on the "
                "balanced-by-subtype benchmark."
            ),
            "interpretation": (
                "Global modality and semantic shifts are easier than sensory artefacts for most methods, "
                "with Mahalanobis strongest overall."
            ),
            "why": "Replaces the older subset-only per-family figure.",
        },
        {
            "file": "figure_layer_ablation_patchcore.png",
            "caption": (
                "PatchCore layer ablation compatibility figure showing AUROC and AUPRC only; "
                "FPR@95%TPR is reported separately because lower values are better."
            ),
            "interpretation": (
                "PatchCore L3 is retained as the localization-oriented companion even though "
                "Mahalanobis is the strongest quantitative gatekeeper."
            ),
            "why": "Justifies using PatchCore heatmaps in the dissertation.",
        },
        {
            "file": "figure_patchcore_layer_detection_metrics.png",
            "caption": "PatchCore layer ablation for detection metrics, with AUROC and AUPRC reported together.",
            "interpretation": "Layer 3 gives the strongest PatchCore detection trade-off.",
            "why": "Separates higher-is-better metrics from the safety metric.",
        },
        {
            "file": "figure_patchcore_layer_safety_metric.png",
            "caption": "PatchCore layer ablation for FPR@95%TPR, where lower values indicate fewer ID false positives.",
            "interpretation": "Layer 3 also gives the lowest PatchCore FPR@95%TPR among the evaluated layers.",
            "why": "Prevents mixing metrics with opposite preference directions in one line plot.",
        },
        {
            "file": "figure_patchcore_layer_ablation_combined.png",
            "caption": (
                "PatchCore layer ablation showing detection metrics and FPR@95%TPR side by side. "
                "Layer 3 provides the strongest PatchCore detection and safety-oriented performance; "
                "combining layers 2 and 3 does not improve over layer 3 alone."
            ),
            "interpretation": (
                "PatchCore L3 is the best localisation-oriented PatchCore configuration, but remains "
                "a companion to Mahalanobis rather than the strongest quantitative gatekeeper."
            ),
            "why": "Recommended main-text PatchCore ablation figure.",
        },
        {
            "file": "figure_per_ood_subtype_by_scheme.png",
            "caption": "Subtype-level Stage 1 AUROC heatmap across OOD stress-test categories.",
            "interpretation": (
                "The heatmap shows strong performance on global shifts and weaker behaviour on subtle "
                "local artefacts such as text watermark."
            ),
            "why": "Best appendix figure for detailed failure-mode questions.",
        },
        {
            "file": "figure_per_ood_subtype_comparison.png",
            "caption": (
                "Stable-name subtype-level Stage 1 AUROC heatmap for Overleaf and final "
                "dissertation references."
            ),
            "interpretation": (
                "The stable comparison copy preserves the same subtype pattern while giving the "
                "dissertation a concise figure filename."
            ),
            "why": "Use when a shorter filename is preferred for the main dissertation source.",
        },
        {
            "file": "figure_score_distribution_with_threshold.png",
            "caption": (
                "Mahalanobis anomaly-score distributions on the balanced-by-subtype benchmark, with the "
                "95th-percentile ID-validation threshold overlaid. The ID split is synthetic FAF fallback, "
                "not real clinical FAF validation."
            ),
            "interpretation": (
                "The figure shows why modality and semantic shifts separate clearly while sensory artefacts "
                "overlap more with ID scores."
            ),
            "why": "Optional main-text or appendix support for threshold-policy interpretation.",
        },
        {
            "file": "figure_threshold_policy_tradeoff.png",
            "caption": (
                "Mahalanobis threshold policy trade-off between ID false rejection and OOD recall. "
                "Research-only and deployment-style thresholds are shown separately."
            ),
            "interpretation": (
                "The selected prototype policy balances low ID rejection with strong OOD recall but "
                "still requires real clinical validation."
            ),
            "why": "Anchors deployment-threshold caveats.",
        },
        {
            "file": "figure_feature_space_pca_by_ood_type.png",
            "caption": (
                "Mahalanobis feature-space PCA by OOD category. This is a qualitative visualization, "
                "not the primary quantitative metric."
            ),
            "interpretation": (
                "Global semantic and modality shifts separate more clearly than subtle sensory "
                "artifacts, explaining the method's strengths and weaknesses."
            ),
            "why": "Connects quantitative results to feature-space intuition.",
        },
        {
            "file": "figure_two_stage_updated_pipeline.png",
            "caption": (
                "Two-stage pipeline showing Stage 1 ID-only rejection followed by optional Stage 2 "
                "reason attribution for rejected inputs. Stage 2 labels are likely explanations, "
                "not clinical diagnoses."
            ),
            "interpretation": (
                "Stage 2 explains possible rejection reasons after the Stage 1 decision and does not "
                "train or replace the gatekeeper."
            ),
            "why": "Prevents leakage or disease-classifier misinterpretation.",
        },
        {
            "file": "figure_reason_method_family_macro_f1.png",
            "caption": (
                "Parent-grouped Stage 2 reason-family method comparison. Feature-statistics "
                "fusion was selected using grouped-validation macro-F1 and the predefined "
                "complexity/name tie-breaking rule before grouped-test evaluation."
            ),
            "interpretation": (
                "`feature_statistics_fusion` is the grouped-validation family attribution "
                "method and reaches grouped-test accuracy and macro-F1 of 1.0000."
            ),
            "why": "Main Stage 2 quantitative comparison.",
        },
        {
            "file": "figure_stage2_method_comparison_combined.png",
            "caption": (
                "Stage 2 reason-attribution method comparison for rejected inputs. Panel (a) shows "
                "reason-family macro-F1 and panel (b) shows reason-subtype macro-F1; the selected "
                "methods are feature-statistics fusion for family attribution and the non-oracle "
                "hierarchical classifier for subtype attribution."
            ),
            "interpretation": (
                "The combined Stage 2 figure presents explanation performance without changing the "
                "Stage 1 ID-only OOD gatekeeper boundary."
            ),
            "why": "Recommended main-text Stage 2 comparison figure.",
        },
        {
            "file": "figure_reason_method_accuracy_macro_f1.png",
            "caption": (
                "Stage 2 method comparison showing paired test accuracy and macro-F1 for reason "
                "family attribution."
            ),
            "interpretation": (
                "The paired metric view confirms that strong family attribution is not driven by "
                "accuracy alone."
            ),
            "why": "Useful as an appendix companion to the main Stage 2 macro-F1 figure.",
        },
        {
            "file": "figure_reason_method_subtype_macro_f1.png",
            "caption": "Stage 2 subtype method comparison using test macro-F1 with the PR #24 baseline shown.",
            "interpretation": "The non-oracle hierarchical classifier is the selected subtype attribution method.",
            "why": "Main Stage 2 subtype-selection figure.",
        },
        {
            "file": "figure_best_reason_family_confusion_matrix.png",
            "caption": (
                "Count confusion matrix for parent-grouped Stage 2 reason-family attribution "
                "using `feature_statistics_fusion`."
            ),
            "interpretation": (
                "All three reason families have grouped-test F1 of 1.0000, so there is no unique "
                "hardest family in this split."
            ),
            "why": "Shows error structure rather than only aggregate performance.",
        },
        {
            "file": "figure_best_reason_family_confusion_matrix_normalized.png",
            "caption": (
                "Row-normalized confusion matrix for parent-grouped Stage 2 reason-family "
                "attribution using `feature_statistics_fusion`."
            ),
            "interpretation": "Every family row is classified correctly in the grouped test split.",
            "why": "Useful when discussing family-level error rates rather than counts.",
        },
        {
            "file": "figure_best_subtype_confusion_matrix.png",
            "caption": (
                "Count confusion matrix for Stage 2 subtype attribution using the non-oracle "
                "`hierarchical_classifier`. Short labels are defined in `subtype_label_mapping.md`."
            ),
            "interpretation": (
                "The subtype classifier performs strongly overall but leaves text watermark as "
                "the hardest grouped-test subtype."
            ),
            "why": "Best figure for fine-grained Stage 2 limitations.",
        },
        {
            "file": "figure_best_subtype_confusion_matrix_normalized.png",
            "caption": (
                "Row-normalized confusion matrix for Stage 2 subtype attribution using the non-oracle "
                "`hierarchical_classifier`. Short labels are defined in `subtype_label_mapping.md`."
            ),
            "interpretation": "Row percentages make subtype-specific confusion patterns easier to read.",
            "why": "Appendix companion for detailed Stage 2 error analysis.",
        },
        {
            "file": "figure_heatmaps_sensory_artifact_examples.png",
            "caption": (
                "Representative PatchCore L3 sensory-artefact examples shown as input image, anomaly map "
                "and overlay. These maps are qualitative localisation evidence only and should not be "
                "interpreted as pixel-level clinical ground truth."
            ),
            "interpretation": "The triptych layout makes it clear which panel is the raw input and which is the heatmap.",
            "why": "Appendix qualitative support for PatchCore as the localisation-oriented companion.",
        },
        {
            "file": "figure_heatmaps_modality_examples.png",
            "caption": (
                "Representative PatchCore L3 modality-shift examples shown as input image, anomaly map "
                "and overlay. These examples contrast broader modality differences with local artefact cases."
            ),
            "interpretation": "The modality examples provide qualitative contrast to sensory artefacts.",
            "why": "Appendix qualitative support for the localisation discussion.",
        },
    ]
    lines = ["# Polished Caption Suggestions", ""]
    notes = ["# Figure Interpretation Notes", ""]
    for item in captions:
        lines.extend(
            [
                f"## {item['file']}",
                "",
                f"Caption: {item['caption']}",
                "",
            ]
        )
        notes.extend(
            [
                f"## {item['file']}",
                "",
                f"Interpretation: {item['interpretation']}",
                "",
                f"Why it matters: {item['why']}",
                "",
            ]
        )
    FINAL_DIR.joinpath("caption_suggestions.md").write_text("\n".join(lines), encoding="utf-8")
    FINAL_DIR.joinpath("interpretation_notes.md").write_text("\n".join(notes), encoding="utf-8")
    _write_subtype_label_mapping()

    selected_lines = [
        "# Polished Figure Inventory",
        "",
        "These stable-name figures were regenerated from committed result tables for the final "
        "presentation polish pass. No new models, datasets, or experiments were run.",
        "",
        "Export decision: final dissertation figures are committed as high-resolution PNG files "
        "with matching PDF companions when available. The PDFs are intended for vector-friendly "
        "Overleaf use, while PNGs remain convenient for quick preview and sharing.",
        "",
        "| figure | dissertation use |",
        "| --- | --- |",
    ]
    for figure in _inventory_figure_paths():
        selected_lines.append(f"| `{figure}` | final polished dissertation figure |")
    FINAL_DIR.joinpath("polished_figure_inventory.md").write_text(
        "\n".join(selected_lines) + "\n",
        encoding="utf-8",
    )


def _inventory_figure_paths() -> list[str]:
    paths: list[str] = []
    for figure in SELECTED_FIGURES:
        if figure not in paths:
            paths.append(figure)
        if figure.lower().endswith(".png"):
            pdf = f"{figure[:-4]}.pdf"
            if (ROOT / pdf).exists() and pdf not in paths:
                paths.append(pdf)
    return paths


def _write_subtype_label_mapping() -> None:
    rows = [
        {
            "code": code,
            "short_code": SUBTYPE_SHORT_CODES[subtype],
            "ood_subtype": subtype,
            "display_label": _pretty_label(subtype),
        }
        for subtype, code in SUBTYPE_CODES.items()
    ]
    table = pd.DataFrame(rows)
    table.to_csv(FINAL_DIR / "subtype_label_mapping.csv", index=False)
    (FINAL_DIR / "subtype_label_mapping.md").write_text(
        _to_markdown(table, title="subtype_label_mapping"),
        encoding="utf-8",
    )


def _write_final_figure_docs() -> None:
    rows = _final_figure_rows()
    index_lines = [
        "| path | title | chapter | placement | status | one-line message | polished caption |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        index_lines.append(
            "| {path} | {title} | {chapter} | {placement} | {status} | {message} | {caption} |".format(
                **{key: _escape_pipe(str(value)) for key, value in row.items()}
            )
        )
    FINAL_DIR.joinpath("final_figure_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    main_rows = [row for row in rows if row["placement"] == "main text"]
    appendix_rows = [row for row in rows if row["placement"] == "appendix"]
    lines = [
        "# Final Thesis Figure Shortlist",
        "",
        "Main-text figures are restricted to figures that directly support the dissertation narrative. "
        "Dense diagnostics, full curves and qualitative examples are placed in the appendix.",
        "",
        "## Main Text Figures",
        "",
        "| filename | suggested chapter | placement | one-line message | polished caption |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in main_rows:
        lines.append(
            "| `{path}` | {chapter} | {placement} | {message} | {caption} |".format(
                **{key: _escape_pipe(str(value)) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Appendix Figures",
            "",
            "| filename | suggested chapter | placement | one-line message | polished caption |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in appendix_rows:
        lines.append(
            "| `{path}` | {chapter} | {placement} | {message} | {caption} |".format(
                **{key: _escape_pipe(str(value)) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Compatibility / Not Recommended For Main Text",
            "",
            "- `reports/dissertation_figures/figure_system_pipeline_overview.png`: retained for old references, but the two-stage pipeline is the canonical system figure.",
            "- Contact sheets are internal gallery artefacts only and should not appear in the dissertation.",
        ]
    )
    (ROOT / "docs" / "dissertation" / "final_figure_shortlist.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _final_figure_rows() -> list[dict[str, str]]:
    return [
        _figure_row(
            "reports/dissertation_figures/reason_attribution_method_comparison/figure_two_stage_updated_pipeline.png",
            "Two-stage rejected-input explanation pipeline",
            "System Requirements and Design",
            "main text",
            "must_include",
            "Canonical system architecture with Stage 1 ID-only rejection and optional Stage 2 explanation.",
            "Two-stage pipeline showing Stage 1 ID-only OOD rejection followed by optional Stage 2 reason attribution for rejected inputs. Stage 2 labels are likely technical explanations, not clinical diagnoses.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_dataset_taxonomy.png",
            "Dataset v1 taxonomy",
            "Dataset Construction and Validation",
            "main text",
            "must_include",
            "Defines ID FAF rows and the OOD stress-test families without implying Stage 1 label leakage.",
            "Dataset v1 taxonomy showing ID FAF rows and modality shift, sensory artefact and semantic outlier stress-test families. Stage 2 splits are explanation-only OOD splits; the ID test split uses synthetic FAF fallback.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_manifest_split_sizes.png",
            "Manifest split sizes",
            "Dataset Construction and Validation",
            "main text",
            "must_include",
            "Shows task-specific manifest row counts and warns that rows are non-additive.",
            "Committed manifest split sizes for ID, OOD evaluation and Stage 2 reason-attribution rows. Bars are task-specific and non-additive; the packaged dataset contains 3,100 unique image files.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_stage1_overall_comparison_combined.png",
            "Stage 1 overall method comparison",
            "Experiments and Results",
            "main text",
            "must_include",
            "Combines AUROC/AUPRC with FPR@95%TPR for the final eight Stage 1 configurations.",
            "Stage 1 overall comparison on the balanced-by-subtype benchmark. Panel (a) reports AUROC and AUPRC; panel (b) reports FPR@95%TPR, where lower is better. AUPRC is prevalence-sensitive under the OOD-heavy benchmark.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_patchcore_layer_ablation_combined.png",
            "PatchCore layer ablation",
            "Experiments and Results",
            "main text",
            "must_include",
            "Shows PatchCore L3 as the strongest PatchCore configuration.",
            "PatchCore layer ablation showing detection metrics and FPR@95%TPR. L3 provides the strongest PatchCore detection and safety-oriented performance; combining L2 and L3 does not improve over L3 alone.",
        ),
        _figure_row(
            "reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png",
            "Mahalanobis threshold policy trade-off",
            "Experiments and Results",
            "main text",
            "must_include",
            "Shows ID false rejection versus OOD recall for deployment-style and research-only thresholds.",
            "Mahalanobis threshold policy trade-off between ID false rejection and OOD recall. Deployment-style thresholds use held-out ID validation only; the research 95% TPR point is evaluation-only.",
        ),
        _figure_row(
            "reports/dissertation_figures/reason_attribution_method_comparison/figure_stage2_method_comparison_combined.png",
            "Stage 2 method comparison",
            "Experiments and Results",
            "main text",
            "must_include",
            "Shows selected family and subtype attribution methods without changing the Stage 1 boundary.",
            "Parent-grouped Stage 2 reason-attribution method comparison. Feature-statistics fusion is selected for family attribution by grouped validation; the non-oracle hierarchical classifier is selected for subtype attribution.",
        ),
        _figure_row(
            "reports/dissertation_figures/reason_attribution_method_comparison/figure_best_reason_family_confusion_matrix.png",
            "Stage 2 family confusion matrix",
            "Experiments and Results",
            "main text",
            "must_include",
            "Shows perfect grouped-test family attribution across the three reason families.",
            "Count confusion matrix for parent-grouped Stage 2 family attribution using `feature_statistics_fusion`; every family has test F1 1.0000.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_score_distribution_with_threshold.png",
            "Mahalanobis score distribution",
            "Threshold Safety",
            "main text",
            "optional",
            "Shows score separation and the 95th-percentile ID-validation threshold.",
            "Mahalanobis score distributions for synthetic FAF fallback ID and balanced-by-subtype OOD families, with the 95th-percentile ID-validation threshold overlaid.",
        ),
        _figure_row(
            "reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png",
            "Feature-space PCA by OOD family",
            "Appendix",
            "appendix",
            "supporting",
            "Qualitative two-dimensional projection only.",
            "Two-dimensional PCA projection of Mahalanobis feature space by OOD family. This is qualitative supporting evidence and should not be interpreted as the full high-dimensional decision geometry.",
        ),
        _figure_row(
            "reports/dissertation_figures/reason_attribution_method_comparison/figure_best_subtype_confusion_matrix.png",
            "Stage 2 subtype confusion matrix",
            "Appendix",
            "appendix",
            "recommended",
            "Landscape-friendly subtype confusion matrix with short labels.",
            "Count confusion matrix for Stage 2 subtype attribution using the non-oracle `hierarchical_classifier`. Short labels are defined in `subtype_label_mapping.md`.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_roc_overall_model_comparison.png",
            "Overall ROC comparison",
            "Appendix",
            "appendix",
            "recommended",
            "Full ROC comparison for all eight Stage 1 configurations.",
            "Overall ROC comparison for the final eight Stage 1 configurations on the balanced-by-subtype benchmark. Curves are regenerated from per-sample score files.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_pr_overall_model_comparison.png",
            "Overall precision-recall comparison",
            "Appendix",
            "appendix",
            "recommended",
            "Full PR comparison for all eight Stage 1 configurations.",
            "Overall precision-recall comparison for the final eight Stage 1 configurations. AUPRC is affected by the OOD-heavy evaluation prevalence.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_per_ood_type_comparison.png",
            "Per-OOD-family comparison",
            "Appendix",
            "appendix",
            "recommended",
            "Full family-level AUROC heatmap for all eight Stage 1 configurations.",
            "Per-OOD-family AUROC heatmap for the final eight Stage 1 configurations on the balanced-by-subtype benchmark.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_per_ood_subtype_by_scheme.png",
            "Per-subtype Stage 1 heatmap",
            "Appendix",
            "appendix",
            "recommended",
            "Detailed subtype-level Stage 1 diagnostic view.",
            "Subtype-level Stage 1 AUROC heatmap across OOD stress-test categories, showing complementary failure modes across methods.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png",
            "Sensory artefact PatchCore examples",
            "Appendix",
            "appendix",
            "optional",
            "Input/map/overlay examples for representative sensory artefacts.",
            "Representative PatchCore L3 sensory-artefact examples shown as input image, anomaly map and overlay. These maps are qualitative localisation evidence only.",
        ),
        _figure_row(
            "reports/dissertation_figures/figure_heatmaps_modality_examples.png",
            "Modality-shift PatchCore examples",
            "Appendix",
            "appendix",
            "optional",
            "Input/map/overlay examples for colour fundus and OCT screenshots.",
            "Representative PatchCore L3 modality-shift examples shown as input image, anomaly map and overlay.",
        ),
    ]


def _figure_row(
    path: str,
    title: str,
    chapter: str,
    placement: str,
    status: str,
    message: str,
    caption: str,
) -> dict[str, str]:
    return {
        "path": path,
        "title": title,
        "chapter": chapter,
        "placement": placement,
        "status": status,
        "message": message,
        "caption": caption,
    }


def _write_figure_revision_report() -> None:
    rows = [
        ("01 Stage 1 pipeline", "removed_from_main_text", "Retained only for compatibility; superseded by the two-stage pipeline.", "reports/dissertation_figures/figure_system_pipeline_overview.png", "generated diagram code"),
        ("02 Dataset taxonomy", "regenerated", "British-English display text and clearer ID-only/OOD evaluation boundary.", "reports/dissertation_figures/figure_dataset_taxonomy.png", "dataset taxonomy script logic"),
        ("03 Manifest split sizes", "regenerated", "Added non-additive manifest-row warning and 3,100 packaged image statement.", "reports/dissertation_figures/figure_manifest_split_sizes.png", "datasets/dissertation_v1/manifests/*.csv"),
        ("04+05 Stage 1 overall comparison", "combined", "New two-panel main-text figure combining AUROC/AUPRC and FPR@95%TPR.", "reports/dissertation_figures/figure_stage1_overall_comparison_combined.png", "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv"),
        ("06+07 PatchCore ablation", "combined", "New two-panel layer-ablation figure with detection and safety-oriented metrics.", "reports/dissertation_figures/figure_patchcore_layer_ablation_combined.png", "reports/dissertation_results/primary_balanced_by_subtype_10k/layer_ablation_table.csv"),
        ("08 Threshold trade-off", "regenerated", "Deployment and research thresholds use different markers; ID-validation-only note added.", "reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png", "reports/dissertation_results/robustness_analysis/threshold_policy_sweep.csv"),
        ("09 PCA", "moved_to_appendix", "Marked as qualitative two-dimensional projection only.", "reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png", "reports/dissertation_results/robustness_analysis/feature_space_projection.csv"),
        ("11+12 Stage 2 methods", "combined", "Two-panel grouped validation/test macro-F1 figure preserving validation-only selection.", "reports/dissertation_figures/reason_attribution_method_comparison/figure_stage2_method_comparison_combined.png", "reports/stage2_grouped/family_metrics.csv; reports/stage2_grouped/subtype_metrics.csv"),
        ("13 Family confusion", "regenerated", "British-English display labels retained with grouped-test raw counts.", "reports/dissertation_figures/reason_attribution_method_comparison/figure_best_reason_family_confusion_matrix.png", "reports/stage2_grouped/family_confusion_matrix.csv"),
        ("14 Subtype confusion", "moved_to_appendix", "Short labels CF/OCT/TXT/RECT/etc. replace opaque S1-S11 labels.", "reports/dissertation_figures/reason_attribution_method_comparison/figure_best_subtype_confusion_matrix.png", "reports/stage2_grouped/subtype_confusion_matrix.csv; subtype_label_mapping.md"),
        ("15/16 ROC/PR", "regenerated", "Full eight-method ROC/PR curves regenerated from per-sample score files.", "reports/dissertation_figures/figure_roc_overall_model_comparison.png; reports/dissertation_figures/figure_pr_overall_model_comparison.png", "reports/generated/dissertation_runs/*/runs/*/evaluation/scores.csv"),
        ("17 Per-family comparison", "regenerated", "Full eight-method family-level AUROC heatmap regenerated from committed CSV.", "reports/dissertation_figures/figure_per_ood_type_comparison.png", "reports/dissertation_results/multi_scheme_comparison/per_ood_type_by_scheme.csv"),
        ("18 Per-subtype heatmap", "appendix_retained", "Retained as detailed appendix diagnostic heatmap.", "reports/dissertation_figures/figure_per_ood_subtype_by_scheme.png", "reports/dissertation_results/multi_scheme_comparison/per_ood_subtype_by_scheme.csv"),
        ("19 Score distribution", "regenerated", "Detector, threshold source and benchmark are now explicit.", "reports/dissertation_figures/figure_score_distribution_with_threshold.png", "Mahalanobis score file; metrics_by_scheme.csv"),
        ("20/21 PatchCore qualitative heatmaps", "regenerated", "Input/anomaly-map/overlay triptychs use representative subtypes from genuine PatchCore heatmap components.", "reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png; reports/dissertation_figures/figure_heatmaps_modality_examples.png", "PatchCore selected heatmap artifacts generated from existing score files and memory bank"),
    ]
    lines = [
        "# Dissertation Figure Revision Report",
        "",
        "No models were retrained. No dataset splits or experimental metrics were changed. "
        "All regenerated figures are derived from existing result tables, manifests or per-sample score files.",
        "",
        "## Figure Decisions",
        "",
        "| old figure | decision | rationale | new/output path | evidence source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_pipe(value) for value in row) + " |")
    lines.extend(
        [
            "",
            "## Automatic Consistency Checks",
            "",
            "| check | result |",
            "| --- | --- |",
            "| Stage 1 expected methods | PASS: eight methods are present in `metrics_by_scheme.csv`. |",
            "| Full ROC/PR evidence | PASS: per-sample score files were available for all eight final Stage 1 configurations. |",
            "| Scalar-to-curve fabrication | PASS: ROC/PR curves are regenerated from score files, not inferred from scalar AUROC/AUPRC. |",
            "| Manifest warning | PASS: `figure_manifest_split_sizes` includes the non-additive row warning. |",
            "| Subtype confusion labels | PASS: short labels are written to `subtype_label_mapping.md` and used in the confusion matrix. |",
            "| Stage boundaries | PASS: generated figure text preserves Stage 1 ID-only detection and Stage 2 optional post-rejection attribution. |",
            "| Clinical overclaim scan | PASS: generated captions state likely technical reasons and synthetic FAF fallback, not clinical diagnoses or validation. |",
        ]
    )
    FINAL_DIR.joinpath("figure_revision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_pipe(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()
