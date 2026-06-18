#!/usr/bin/env python
"""Polish final dissertation figures and reporting from committed result tables."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "reports" / "dissertation_final"
FIG_DIR = ROOT / "reports" / "dissertation_figures"
RESULTS_DIR = ROOT / "reports" / "dissertation_results"
MANIFEST_DIR = ROOT / "datasets" / "dissertation_v1" / "manifests"

BLUE = "#2F6F9F"
GREEN = "#3B7F5C"
RED = "#B85C5C"
GOLD = "#B38B2E"
GRAY = "#5B6470"
LIGHT_BLUE = "#E7F0F7"
LIGHT_GREEN = "#E8F3ED"
LIGHT_RED = "#F6E7E7"
LIGHT_GOLD = "#F8F1DF"

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
SELECTED_FIGURES = [
    "reports/dissertation_figures/figure_system_pipeline_overview.png",
    "reports/dissertation_figures/figure_dataset_taxonomy.png",
    "reports/dissertation_figures/figure_metrics_by_scheme.png",
    "reports/dissertation_figures/figure_fpr95_by_scheme.png",
    "reports/dissertation_figures/figure_layer_ablation_patchcore.png",
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
    "figure_best_reason_family_confusion_matrix.png",
    "reports/dissertation_figures/reason_attribution_method_comparison/"
    "figure_best_subtype_confusion_matrix.png",
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
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "font.size": 9,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


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
    _plot_stage1_method_comparison(tables["stage1"], FIG_DIR / "figure_metrics_by_scheme.png", dpi=dpi)
    _plot_stage1_fpr(tables["stage1"], FIG_DIR / "figure_fpr95_by_scheme.png", dpi=dpi)
    _plot_layer_ablation(tables["layer"], FIG_DIR / "figure_layer_ablation_patchcore.png", dpi=dpi)
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
    _plot_confusion_matrix(
        tables["family_cm"],
        reason_dir / "figure_best_reason_family_confusion_matrix.png",
        title="Selected Stage 2 family method: linear_svm",
        dpi=dpi,
    )
    _plot_confusion_matrix(
        tables["subtype_cm"],
        reason_dir / "figure_best_subtype_confusion_matrix.png",
        title="Selected Stage 2 subtype method: hierarchical_classifier",
        dpi=dpi,
    )


def _plot_stage1_method_comparison(metrics: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = _stage1_method_summary(metrics).sort_values("auroc", ascending=True)
    labels = table["scheme_label"].map(_short_method_label)
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(y - 0.18, table["auroc"], height=0.25, label="AUROC", color=BLUE)
    ax.barh(y + 0.10, table["auprc"], height=0.25, label="AUPRC", color=GREEN)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Metric value")
    ax.set_title("Stage 1 method comparison")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path, dpi=dpi)


def _plot_stage1_fpr(metrics: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = _stage1_method_summary(metrics).sort_values("fpr_at_95_tpr", ascending=False)
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    colors = [GREEN if row["scheme"] == "mahalanobis_feature" else RED for _, row in table.iterrows()]
    ax.barh(y, table["fpr_at_95_tpr"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(table["scheme_label"].map(_short_method_label))
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("FPR@95%TPR (lower is better)")
    ax.set_title("Stage 1 safety metric comparison")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, path, dpi=dpi)


def _plot_layer_ablation(layer: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = layer.copy()
    table["layer_label"] = table["layers"].astype(str).str.replace("layer", "L", regex=False)
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(x, table["auroc"], marker="o", linewidth=2.0, label="AUROC", color=BLUE)
    ax.plot(x, table["auprc"], marker="o", linewidth=2.0, label="AUPRC", color=GREEN)
    ax.plot(
        x,
        table["fpr_at_95_tpr"],
        marker="o",
        linewidth=2.0,
        label="FPR@95%TPR",
        color=RED,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(table["layer_label"])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Metric value")
    ax.set_title("PatchCore layer ablation")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=3, loc="lower center")
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
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    _draw_heatmap(ax, pivot, title="Per-subtype AUROC by Stage 1 method", cmap="Blues")
    _save(fig, path, dpi=dpi)


def _plot_threshold_tradeoff(threshold: pd.DataFrame, path: Path, *, dpi: int) -> None:
    rows = threshold[threshold["scheme"].astype(str).eq("mahalanobis_feature")].copy()
    rows = rows.sort_values("id_false_rejection_rate")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    colors = [GOLD if kind == "research" else BLUE for kind in rows["policy_kind"]]
    ax.scatter(
        rows["id_false_rejection_rate"],
        rows["ood_recall"],
        s=90,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
    )
    for _, row in rows.iterrows():
        ax.text(
            row["id_false_rejection_rate"] + 0.006,
            row["ood_recall"],
            str(row["policy"]).replace("val_id_quantile_", "q"),
            fontsize=8,
            va="center",
        )
    ax.set_xlabel("ID false rejection rate")
    ax.set_ylabel("OOD recall")
    ax.set_title("Mahalanobis threshold policy trade-off")
    ax.set_xlim(0.0, max(0.22, float(rows["id_false_rejection_rate"].max()) + 0.04))
    ax.set_ylim(0.75, 1.01)
    ax.grid(alpha=0.22)
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
    for group in ["id", *OOD_TYPE_ORDER]:
        rows = table[table["group"].eq(group)]
        ax.scatter(
            rows["pc1"],
            rows["pc2"],
            s=16 if group == "id" else 18,
            alpha=0.58,
            label=group,
            color=colors[group],
            linewidth=0,
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
    labels = table["method"].map(_pretty_method)
    colors = [GREEN if method == "linear_svm" else BLUE for method in table["method"]]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh(np.arange(len(table)), table["family_macro_f1"], color=colors)
    ax.axvline(0.8947, linestyle="--", color=GRAY, linewidth=1.2, label="PR #24 baseline")
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("Family macro-F1")
    ax.set_title("Stage 2 reason-family method comparison")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path, dpi=dpi)


def _plot_stage2_accuracy_macro_f1(family: pd.DataFrame, path: Path, *, dpi: int) -> None:
    table = family[family["split"].astype(str).eq("test")].copy()
    table = table.sort_values("family_macro_f1", ascending=False)
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - 0.18, table["family_accuracy"], width=0.34, color=BLUE, label="Accuracy")
    ax.bar(x + 0.18, table["family_macro_f1"], width=0.34, color=GREEN, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(table["method"].map(_pretty_method), rotation=25, ha="right")
    ax.set_ylim(0.35, 1.02)
    ax.set_ylabel("Metric value")
    ax.set_title("Stage 2 family attribution accuracy and macro-F1")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=2)
    _save(fig, path, dpi=dpi)


def _plot_confusion_matrix(cm: pd.DataFrame, path: Path, *, title: str, dpi: int) -> None:
    table = cm.set_index(cm.columns[0])
    values = table.to_numpy(dtype=float)
    labels = [_pretty_label(value) for value in table.index]
    columns = [_pretty_label(value) for value in table.columns]
    fig_width = 6.4 if len(labels) <= 4 else 9.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.74))
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    threshold = np.nanmax(values) / 2.0 if values.size else 0.0
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            color = "white" if values[y, x] > threshold else "#1F2933"
            ax.text(x, y, f"{int(values[y, x])}", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    _save(fig, path, dpi=dpi)


def _plot_system_pipeline(path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.axis("off")
    boxes = [
        (0.04, 0.46, 0.14, 0.22, "Input\nimage", LIGHT_BLUE),
        (0.28, 0.46, 0.20, 0.22, "Stage 1\nID-only OOD gatekeeper", LIGHT_GREEN),
        (0.60, 0.64, 0.17, 0.18, "ACCEPT\nvalid FAF", LIGHT_GREEN),
        (0.60, 0.28, 0.17, 0.18, "REJECT\ninvalid/OOD", LIGHT_RED),
        (0.83, 0.64, 0.13, 0.18, "Downstream\nanalysis", "#F7F8FA"),
    ]
    for x, y, width, height, text, color in boxes:
        _box(ax, x, y, width, height, text, facecolor=color)
    _arrow(ax, (0.18, 0.57), (0.28, 0.57))
    _arrow(ax, (0.48, 0.57), (0.60, 0.73))
    _arrow(ax, (0.48, 0.57), (0.60, 0.37))
    _arrow(ax, (0.77, 0.73), (0.83, 0.73))
    ax.text(
        0.38,
        0.26,
        "Stage 1 fitting uses ID rows only; OOD labels are evaluation-only.",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    ax.set_title("Overall FAF OOD gatekeeper pipeline")
    _save(fig, path, dpi=dpi)


def _plot_two_stage_pipeline(path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.axis("off")
    boxes = [
        (0.04, 0.48, 0.13, 0.20, "Input\nimage", LIGHT_BLUE),
        (0.26, 0.48, 0.18, 0.20, "Stage 1\nID-only gatekeeper", LIGHT_GREEN),
        (0.55, 0.66, 0.15, 0.17, "ACCEPT", LIGHT_GREEN),
        (0.55, 0.30, 0.15, 0.17, "REJECT", LIGHT_RED),
        (0.78, 0.30, 0.17, 0.17, "Stage 2\nreason attribution", LIGHT_GOLD),
    ]
    for x, y, width, height, text, color in boxes:
        _box(ax, x, y, width, height, text, facecolor=color)
    _arrow(ax, (0.17, 0.58), (0.26, 0.58))
    _arrow(ax, (0.44, 0.58), (0.55, 0.75))
    _arrow(ax, (0.44, 0.58), (0.55, 0.39))
    _arrow(ax, (0.70, 0.39), (0.78, 0.39))
    ax.text(0.86, 0.22, "Likely explanation,\nnot diagnosis", ha="center", fontsize=8.5, color=GRAY)
    ax.text(0.35, 0.28, "OOD labels are not used for Stage 1 fitting.", ha="center", fontsize=8.5)
    ax.set_title("Two-stage rejected-input explanation pipeline")
    _save(fig, path, dpi=dpi)


def _plot_dataset_taxonomy(path: Path, *, dpi: int) -> None:
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
    counts = {name: len(pd.read_csv(path_in)) for name, path_in in manifests.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    ax.axis("off")
    groups = [
        (
            "ID FAF",
            "Stage 1 training/validation\nand synthetic fallback ID",
            0.06,
            0.61,
            LIGHT_GREEN,
        ),
        (
            "Modality shift",
            "colour fundus\nOCT screenshot",
            0.56,
            0.61,
            LIGHT_BLUE,
        ),
        (
            "Sensory artifact",
            "watermark, annotations\nblur, crop, noise, JPEG",
            0.06,
            0.27,
            LIGHT_GOLD,
        ),
        (
            "Semantic outlier",
            "natural-image outliers\nnon-FAF content",
            0.56,
            0.27,
            LIGHT_RED,
        ),
    ]
    ax.text(0.50, 0.89, "Dataset v1 taxonomy", ha="center", va="center", fontsize=12)
    ax.text(
        0.50,
        0.80,
        "Binary gatekeeper framing: ID accepted, OOD rejected",
        ha="center",
        va="center",
        fontsize=8.5,
        color=GRAY,
    )
    for label, subtitle, x, y, color in groups:
        _box(ax, x, y, 0.36, 0.13, label, facecolor=color)
        ax.text(x + 0.18, y - 0.045, subtitle, ha="center", va="top", fontsize=8.2, linespacing=1.25)

    ax2 = axes[1]
    names = list(counts)
    values = [counts[name] for name in names]
    ax2.barh(np.arange(len(names)), values, color=[GREEN, GREEN, GREEN, BLUE, BLUE, GOLD, GOLD, GOLD])
    ax2.set_yticks(np.arange(len(names)))
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlabel("Rows")
    ax2.set_title("Committed manifest sizes")
    ax2.grid(axis="x", alpha=0.22)
    for index, value in enumerate(values):
        ax2.text(value + max(values) * 0.015, index, str(value), va="center", fontsize=8)
    _save(fig, path, dpi=dpi)


def _draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, *, title: str, cmap: str) -> None:
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_pretty_label(value) for value in pivot.columns], rotation=38, ha="right")
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


def _pretty_label(value: object) -> str:
    return str(value).replace("_", " ")


def _save(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def _write_caption_and_interpretation_notes() -> None:
    captions = [
        {
            "file": "figure_system_pipeline_overview.png",
            "caption": (
                "Overall system pipeline for the Stage 1 ID-only FAF OOD gatekeeper. "
                "The gatekeeper accepts likely valid FAF inputs and rejects invalid/OOD inputs before "
                "downstream analysis."
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
                "Dataset taxonomy and committed manifest sizes for ID FAF, modality-shift OOD, "
                "sensory-artifact OOD, semantic outliers, and Stage 2 reason-attribution splits."
            ),
            "interpretation": (
                "The taxonomy makes clear which data are used for ID-only training and which are "
                "used for evaluation or post-rejection explanation."
            ),
            "why": "Supports the dataset chapter and the ID-only/OOD-only boundary.",
        },
        {
            "file": "figure_metrics_by_scheme.png",
            "caption": (
                "Stage 1 method comparison using AUROC and AUPRC on the balanced-by-subtype OOD "
                "evaluation set."
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
            "caption": "PatchCore layer ablation across evaluated ResNet feature layers.",
            "interpretation": (
                "PatchCore L3 is retained as the localization-oriented companion even though "
                "Mahalanobis is the strongest quantitative gatekeeper."
            ),
            "why": "Justifies using PatchCore heatmaps in the dissertation.",
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
                "Mahalanobis threshold policy trade-off between ID false rejection and OOD recall."
            ),
            "interpretation": (
                "The selected prototype policy balances low ID rejection with strong OOD recall but "
                "still requires real clinical validation."
            ),
            "why": "Anchors deployment-threshold caveats.",
        },
        {
            "file": "figure_feature_space_pca_by_ood_type.png",
            "caption": "PCA projection of Mahalanobis feature space by OOD category.",
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
                "reason attribution for rejected inputs."
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
            "file": "figure_best_reason_family_confusion_matrix.png",
            "caption": "Confusion matrix for the selected Stage 2 `linear_svm` family method.",
            "interpretation": (
                "Most family-level errors occur for semantic outliers, the hardest reason family."
            ),
            "why": "Shows error structure rather than only aggregate performance.",
        },
        {
            "file": "figure_best_subtype_confusion_matrix.png",
            "caption": "Confusion matrix for the non-oracle Stage 2 hierarchical subtype method.",
            "interpretation": (
                "The subtype classifier performs strongly overall but leaves rectangle annotation as "
                "the hardest subtype."
            ),
            "why": "Best figure for fine-grained Stage 2 limitations.",
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

    selected_lines = [
        "# Polished Figure Inventory",
        "",
        "These stable-name figures were regenerated from committed result tables for the final "
        "presentation polish pass. No new models, datasets, or experiments were run.",
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


if __name__ == "__main__":
    main()
