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

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retinal_ood.visualization.dissertation_style import (  # noqa: E402
    COLORS,
    LIGHT_COLORS,
    apply_dissertation_style,
    save_figure,
)

FINAL_DIR = ROOT / "reports" / "dissertation_final"
FIG_DIR = ROOT / "reports" / "dissertation_figures"
RESULTS_DIR = ROOT / "reports" / "dissertation_results"
MANIFEST_DIR = ROOT / "datasets" / "dissertation_v1" / "manifests"

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
FAMILY_LABELS = {
    "modality_shift": "Modality shift",
    "sensory_artifact": "Sensory artifact",
    "semantic_outlier": "Semantic outlier",
}
SELECTED_FIGURES = [
    "reports/dissertation_figures/figure_system_pipeline_overview.png",
    "reports/dissertation_figures/figure_dataset_taxonomy.png",
    "reports/dissertation_figures/figure_manifest_split_sizes.png",
    "reports/dissertation_figures/figure_metrics_by_scheme.png",
    "reports/dissertation_figures/figure_fpr95_by_scheme.png",
    "reports/dissertation_figures/figure_layer_ablation_patchcore.png",
    "reports/dissertation_figures/figure_patchcore_layer_detection_metrics.png",
    "reports/dissertation_figures/figure_patchcore_layer_safety_metric.png",
    "reports/dissertation_figures/figure_per_ood_subtype_by_scheme.png",
    "reports/dissertation_figures/figure_per_ood_subtype_comparison.png",
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
    print("Polished dissertation reporting artifacts written.")


def _configure_matplotlib() -> None:
    apply_dissertation_style()


def _load_tables() -> dict[str, pd.DataFrame]:
    return {
        "stage1": _read_csv(RESULTS_DIR / "multi_scheme_comparison" / "metrics_by_scheme.csv"),
        "subtype": _read_csv(
            RESULTS_DIR / "multi_scheme_comparison" / "per_ood_subtype_by_scheme.csv"
        ),
        "layer": _read_csv(
            RESULTS_DIR / "primary_balanced_by_subtype_10k" / "layer_ablation_table.csv"
        ),
        "threshold": _read_csv(RESULTS_DIR / "robustness_analysis" / "threshold_policy_sweep.csv"),
        "pca": _read_csv(RESULTS_DIR / "robustness_analysis" / "feature_space_projection.csv"),
        "stage2_family": _read_csv(
            RESULTS_DIR / "reason_attribution_method_comparison" / "family_metrics_by_method.csv"
        ),
        "stage2_subtype": _read_csv(
            RESULTS_DIR / "reason_attribution_method_comparison" / "subtype_metrics_by_method.csv"
        ),
        "stage2_best": _read_csv(
            RESULTS_DIR / "reason_attribution_method_comparison" / "best_method_summary.csv"
        ),
        "family_cm": _read_csv(
            RESULTS_DIR
            / "reason_attribution_method_comparison"
            / "best_reason_family_confusion_matrix.csv"
        ),
        "subtype_cm": _read_csv(
            RESULTS_DIR
            / "reason_attribution_method_comparison"
            / "best_subtype_confusion_matrix.csv"
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
    table.loc[table["method"].eq("linear_svm"), "selected_role"] = "Selected family method"
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
            "recommendation": "linear_svm",
            "role": "Best reason-family attribution method",
            "key_metric": "Family accuracy 0.9881; macro-F1 0.9901",
            "evidence": (
                "reports/dissertation_results/reason_attribution_method_comparison/"
                "best_method_summary.csv"
            ),
        },
        {
            "stage": "Stage 2",
            "recommendation": "hierarchical_classifier",
            "role": "Best reason-subtype attribution method",
            "key_metric": "Subtype accuracy 0.9238; macro-F1 0.9059",
            "evidence": (
                "reports/dissertation_results/reason_attribution_method_comparison/"
                "best_method_summary.csv"
            ),
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
            "limitation": "Parent-image-hash overlap",
            "why_it_matters": "Generated artifact variants share parents across Stage 2 splits.",
            "safe_wording": "Stage 2 scores may be optimistic for parent-independent generalization.",
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
            "evidence": "docs/experiments/reason_attribution_method_comparison.md",
            "figure_or_table": "figure_two_stage_updated_pipeline.png",
            "limitation": "Supervised explanation layer; not Stage 1 training.",
        },
        {
            "claim": "linear_svm is the best reason-family method.",
            "evidence": (
                "reports/dissertation_results/reason_attribution_method_comparison/"
                "best_method_summary.csv"
            ),
            "figure_or_table": "figure_reason_method_family_macro_f1.png",
            "limitation": "Scores are not clinical diagnosis performance.",
        },
        {
            "claim": "hierarchical_classifier is the best reason-subtype method.",
            "evidence": (
                "reports/dissertation_results/reason_attribution_method_comparison/"
                "best_method_summary.csv"
            ),
            "figure_or_table": "figure_best_subtype_confusion_matrix.png",
            "limitation": "Parent-image-hash overlap remains documented.",
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
    _plot_confusion_matrix(
        tables["family_cm"],
        reason_dir / "figure_best_reason_family_confusion_matrix.png",
        title="Stage 2 reason-family attribution confusion matrix",
        subtitle="Method: linear_svm; counts",
        label_mode="family",
        normalize=False,
        dpi=dpi,
    )
    _plot_confusion_matrix(
        tables["family_cm"],
        reason_dir / "figure_best_reason_family_confusion_matrix_normalized.png",
        title="Stage 2 reason-family attribution confusion matrix",
        subtitle="Method: linear_svm; row-normalized percentages",
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
    ax.set_xlim(0.70, 1.02)
    ax.set_xlabel("Metric value (axis starts at 0.70)")
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
                label=_pretty_label(group),
                color=colors[group],
                linewidth=0,
                zorder=2,
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Mahalanobis feature-space PCA")
    ax.legend(frameon=False, ncols=2)
    ax.grid(alpha=0.18)
    _save(fig, path, dpi=dpi)


def _plot_stage2_method_comparison(family: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = family[family["split"].astype(str).eq("test")].copy()
    table = table.sort_values("family_macro_f1", ascending=True)
    labels = table["method"].map(_short_reason_method_label)
    colors = [GREEN if method == "linear_svm" else BLUE for method in table["method"]]
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
            "OOD: sensory artifact",
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
        "Stage 1 is trained on ID rows only; OOD groups are evaluation stress tests.",
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
        "Stage 2 rows are OOD explanation splits; synthetic ID fallback is not real clinical FAF validation.",
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
        "OOD balanced": MANIFEST_DIR / "test_ood_balanced_by_subtype.csv",
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
        return SUBTYPE_CODES.get(key, key)
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
    save_figure(fig, path, dpi=dpi)


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
                "Committed manifest split sizes for ID, OOD evaluation, and Stage 2 reason-attribution rows."
            ),
            "interpretation": "The row-count view separates dataset composition from taxonomy.",
            "why": "Keeps the main taxonomy figure uncrowded while preserving manifest evidence.",
        },
        {
            "file": "figure_metrics_by_scheme.png",
            "caption": (
                "Stage 1 method comparison using AUROC and AUPRC on the balanced-by-subtype OOD "
                "evaluation set. AUPRC reflects the OOD-heavy class balance, so the safety-focused "
                "FPR@95%TPR comparison should also be considered."
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
            "file": "figure_per_ood_subtype_by_scheme.png",
            "caption": "Subtype-level Stage 1 AUROC heatmap across OOD stress-test categories.",
            "interpretation": (
                "The heatmap shows strong performance on global shifts and weaker behavior on subtle "
                "local artifacts such as text watermark."
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
            "caption": "Stage 2 reason-family method comparison with the PR #24 baseline shown.",
            "interpretation": (
                "`linear_svm` is selected as the final family attribution method by validation "
                "macro-F1 and holds strong test macro-F1."
            ),
            "why": "Main Stage 2 quantitative comparison.",
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
            "caption": "Count confusion matrix for Stage 2 reason-family attribution using `linear_svm`.",
            "interpretation": (
                "Most family-level errors occur for semantic outliers, the hardest reason family."
            ),
            "why": "Shows error structure rather than only aggregate performance.",
        },
        {
            "file": "figure_best_reason_family_confusion_matrix_normalized.png",
            "caption": "Row-normalized confusion matrix for Stage 2 reason-family attribution using `linear_svm`.",
            "interpretation": "Percentages make the rare family-level errors easier to compare across rows.",
            "why": "Useful when discussing family-level error rates rather than counts.",
        },
        {
            "file": "figure_best_subtype_confusion_matrix.png",
            "caption": (
                "Count confusion matrix for Stage 2 subtype attribution using the non-oracle "
                "`hierarchical_classifier`. Subtype codes are defined in `subtype_label_mapping.md`."
            ),
            "interpretation": (
                "The subtype classifier performs strongly overall but leaves rectangle annotation as "
                "the hardest subtype."
            ),
            "why": "Best figure for fine-grained Stage 2 limitations.",
        },
        {
            "file": "figure_best_subtype_confusion_matrix_normalized.png",
            "caption": (
                "Row-normalized confusion matrix for Stage 2 subtype attribution using the non-oracle "
                "`hierarchical_classifier`. Subtype codes are defined in `subtype_label_mapping.md`."
            ),
            "interpretation": "Row percentages make subtype-specific confusion patterns easier to read.",
            "why": "Appendix companion for detailed Stage 2 error analysis.",
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
        "Export decision: this pass commits high-resolution PNG files only. PDF/SVG companions were "
        "not committed to avoid doubling the figure artifact set; the PNGs are suitable for direct "
        "Overleaf use and can be regenerated from `scripts/polish_dissertation_figures_and_reporting.py`.",
        "",
        "| figure | dissertation use |",
        "| --- | --- |",
    ]
    for figure in SELECTED_FIGURES:
        selected_lines.append(f"| `{figure}` | final polished dissertation figure |")
    FINAL_DIR.joinpath("polished_figure_inventory.md").write_text(
        "\n".join(selected_lines) + "\n",
        encoding="utf-8",
    )


def _write_subtype_label_mapping() -> None:
    rows = [
        {"code": code, "ood_subtype": subtype, "display_label": _pretty_label(subtype)}
        for subtype, code in SUBTYPE_CODES.items()
    ]
    table = pd.DataFrame(rows)
    table.to_csv(FINAL_DIR / "subtype_label_mapping.csv", index=False)
    (FINAL_DIR / "subtype_label_mapping.md").write_text(
        _to_markdown(table, title="subtype_label_mapping"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
