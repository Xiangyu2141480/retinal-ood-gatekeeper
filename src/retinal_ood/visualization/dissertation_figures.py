"""Generate dissertation-ready figures from aggregate evaluation metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from retinal_ood.evaluation.report_tables import ReportTables, build_report_tables

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

FIGURE_FILENAMES = {
    "global_metrics": "global_metrics.png",
    "fpr_at_95_tpr": "fpr_at_95_tpr.png",
    "layer_ablation": "layer_ablation.png",
    "per_ood_auroc": "per_ood_auroc.png",
    "confusion_matrices": "confusion_matrices.png",
}

_LAYER_ORDER = {
    "layer1": 0,
    "layer2": 1,
    "layer3": 2,
    "layer4": 3,
    "layer2+layer3": 4,
}
_BAR_COLOR = "#3366AA"
_AUPRC_COLOR = "#228833"
_FPR_COLOR = "#CC6677"


@dataclass(frozen=True)
class DissertationFigureOutputs:
    figures: dict[str, Path]
    manifest: Path


def generate_dissertation_figures(
    runs_dir: str | Path,
    out_dir: str | Path = "reports/generated/figures",
    *,
    dpi: int = 180,
) -> DissertationFigureOutputs:
    """Build report tables from ``runs_dir`` and save dissertation figure PNGs."""
    tables = build_report_tables(runs_dir)
    return save_dissertation_figures(tables, out_dir, dpi=dpi)


def save_dissertation_figures(
    tables: ReportTables,
    out_dir: str | Path,
    *,
    dpi: int = 180,
) -> DissertationFigureOutputs:
    """Save standard result figures for dissertation/report writing."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if tables.overall.empty:
        raise ValueError("overall report table must contain at least one run")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = _prepare_overall_table(tables.overall)
    per_ood = _prepare_per_ood_table(tables.per_ood)

    figures: dict[str, Path] = {}
    figures["global_metrics"] = _plot_global_metrics(
        overall,
        out_dir / FIGURE_FILENAMES["global_metrics"],
        dpi=dpi,
    )
    figures["fpr_at_95_tpr"] = _plot_fpr_at_95_tpr(
        overall,
        out_dir / FIGURE_FILENAMES["fpr_at_95_tpr"],
        dpi=dpi,
    )
    figures["layer_ablation"] = _plot_layer_ablation(
        overall,
        out_dir / FIGURE_FILENAMES["layer_ablation"],
        dpi=dpi,
    )
    if not per_ood.empty:
        figures["per_ood_auroc"] = _plot_per_ood_auroc(
            per_ood,
            out_dir / FIGURE_FILENAMES["per_ood_auroc"],
            dpi=dpi,
        )
    figures["confusion_matrices"] = _plot_confusion_matrices(
        overall,
        out_dir / FIGURE_FILENAMES["confusion_matrices"],
        dpi=dpi,
    )
    manifest = _write_figure_manifest(out_dir / "figure_manifest.json", figures, overall, per_ood)
    return DissertationFigureOutputs(figures=figures, manifest=manifest)


def _prepare_overall_table(overall: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        overall,
        [
            "run_name",
            "layers",
            "auroc",
            "auprc",
            "fpr_at_95_tpr",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )
    df = overall.copy()
    for column in ["auroc", "auprc", "fpr_at_95_tpr", "tn", "fp", "fn", "tp"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["figure_label"] = [_figure_label(row) for _, row in df.iterrows()]
    df["layer_sort"] = df["layers"].map(lambda value: _LAYER_ORDER.get(str(value), 99))
    return df.sort_values(by=["layer_sort", "run_name"], kind="stable").reset_index(drop=True)


def _prepare_per_ood_table(per_ood: pd.DataFrame) -> pd.DataFrame:
    if per_ood.empty:
        return per_ood.copy()
    _require_columns(per_ood, ["run_name", "layers", "ood_type", "auroc"])
    df = per_ood.copy()
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
    df["figure_label"] = [_figure_label(row) for _, row in df.iterrows()]
    df["layer_sort"] = df["layers"].map(lambda value: _LAYER_ORDER.get(str(value), 99))
    return df.sort_values(by=["layer_sort", "run_name", "ood_type"], kind="stable").reset_index(drop=True)


def _plot_global_metrics(overall: pd.DataFrame, path: Path, *, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=_figure_size(len(overall), height=4.2))
    x = np.arange(len(overall))
    width = 0.36
    ax.bar(x - width / 2, overall["auroc"], width, label="AUROC", color=_BAR_COLOR)
    ax.bar(x + width / 2, overall["auprc"], width, label="AUPRC", color=_AUPRC_COLOR)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Global OOD Detection Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(overall["figure_label"], rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, path, dpi=dpi)


def _plot_fpr_at_95_tpr(overall: pd.DataFrame, path: Path, *, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=_figure_size(len(overall), height=4.2))
    x = np.arange(len(overall))
    ax.bar(x, overall["fpr_at_95_tpr"], color=_FPR_COLOR)
    y_max = max(1.0, float(overall["fpr_at_95_tpr"].max(skipna=True)) * 1.15)
    ax.set_ylim(0.0, y_max)
    ax.set_ylabel("False positive rate")
    ax.set_title("FPR at 95% OOD TPR (Lower Is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(overall["figure_label"], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, path, dpi=dpi)


def _plot_layer_ablation(overall: pd.DataFrame, path: Path, *, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=_figure_size(len(overall), height=4.4))
    x = np.arange(len(overall))
    ax.plot(x, overall["auroc"], marker="o", label="AUROC", color=_BAR_COLOR)
    ax.plot(x, overall["auprc"], marker="o", label="AUPRC", color=_AUPRC_COLOR)
    ax.plot(x, overall["fpr_at_95_tpr"], marker="o", label="FPR@95TPR", color=_FPR_COLOR)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("PatchCore Layer Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(overall["figure_label"], rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, path, dpi=dpi)


def _plot_per_ood_auroc(per_ood: pd.DataFrame, path: Path, *, dpi: int) -> Path:
    pivot = per_ood.pivot_table(
        index="figure_label",
        columns="ood_type",
        values="auroc",
        aggfunc="first",
        sort=False,
    )
    fig_width = max(5.5, 1.0 + 1.1 * len(pivot.columns))
    fig_height = max(3.6, 1.2 + 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Per-OOD Category AUROC")
    ax.set_xlabel("OOD category")
    ax.set_ylabel("Run")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iat[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="AUROC")
    return _save_figure(fig, path, dpi=dpi)


def _plot_confusion_matrices(overall: pd.DataFrame, path: Path, *, dpi: int) -> Path:
    run_count = len(overall)
    cols = min(3, run_count)
    rows = int(np.ceil(run_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")

    for axis, (_, row) in zip(axes.ravel(), overall.iterrows()):
        matrix = np.array(
            [
                [row["tn"], row["fp"]],
                [row["fn"], row["tp"]],
            ],
            dtype=float,
        )
        axis.axis("on")
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(str(row["figure_label"]))
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Pred ID", "Pred OOD"])
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["True ID", "True OOD"])
        for y in range(2):
            for x in range(2):
                axis.text(x, y, f"{int(matrix[y, x])}", ha="center", va="center", color="black")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrices at Selected Thresholds", y=1.02)
    return _save_figure(fig, path, dpi=dpi)


def _write_figure_manifest(
    path: Path,
    figures: dict[str, Path],
    overall: pd.DataFrame,
    per_ood: pd.DataFrame,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "note": (
            "Generated from aggregate metrics tables only. "
            "No per-image scores, patient identifiers, raw image paths, or medical images are included."
        ),
        "run_count": int(len(overall)),
        "ood_category_count": int(per_ood["ood_type"].nunique()) if not per_ood.empty else 0,
        "figures": {key: value.name for key, value in sorted(figures.items())},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _figure_label(row: pd.Series) -> str:
    layers = str(row.get("layers", "")).strip()
    if layers:
        return layers
    return str(row.get("run_name", "run"))


def _figure_size(run_count: int, *, height: float) -> tuple[float, float]:
    return (max(5.8, 1.25 * run_count), height)


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
