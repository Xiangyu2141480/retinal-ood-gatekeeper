#!/usr/bin/env python
"""Evaluate the optional Stage 2 rejected-input reason attribution module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.evaluation.report_tables import dataframe_to_markdown
from retinal_ood.reason_attribution.classifier import (
    ReasonAttributionModel,
    compute_classification_outputs,
    unknown_threshold_sweep,
)


def evaluate_reason_attribution(
    *,
    test_manifest: str | Path,
    root_dir: str | Path,
    model_dir: str | Path,
    out_dir: str | Path,
    figures_dir: str | Path = "reports/dissertation_figures/reason_attribution",
    unknown_threshold: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 0,
    stage1_scores: str | Path | None = None,
) -> dict[str, Path]:
    """Evaluate Stage 2 reason attribution and write dissertation-ready outputs."""
    from retinal_ood.reason_attribution.features import extract_features_from_manifest

    model_path, feature_config = _load_model_config(model_dir)
    model = ReasonAttributionModel.load(model_path)
    test_table = extract_features_from_manifest(
        test_manifest,
        root_dir=root_dir,
        config=feature_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    predictions = model.predict(test_table.features, unknown_threshold=unknown_threshold)
    true_subtype = test_table.metadata["ood_subtype"].astype(str).to_numpy()
    outputs = compute_classification_outputs(
        true_family=test_table.metadata["ood_type"].astype(str).to_numpy(),
        pred_family=predictions.family,
        family_argmax=predictions.family_argmax,
        family_confidence=predictions.family_confidence,
        metadata=test_table.metadata,
        out_dir=out_dir,
        true_subtype=true_subtype if predictions.subtype is not None else None,
        pred_subtype=predictions.subtype,
        subtype_confidence=predictions.subtype_confidence,
        unknown_threshold=unknown_threshold,
    )
    output_dir = Path(out_dir)
    thresholds = np.linspace(0.0, 1.0, 21)
    sweep = unknown_threshold_sweep(
        true_family=test_table.metadata["ood_type"].astype(str).to_numpy(),
        family_argmax=predictions.family_argmax,
        family_confidence=predictions.family_confidence,
        thresholds=thresholds,
    )
    sweep_csv = output_dir / "reason_unknown_threshold_sweep.csv"
    sweep_md = output_dir / "reason_unknown_threshold_sweep.md"
    sweep.to_csv(sweep_csv, index=False)
    sweep_md.write_text("# Unknown Threshold Sweep\n\n" + dataframe_to_markdown(sweep) + "\n", encoding="utf-8")
    outputs["unknown_sweep_csv"] = sweep_csv
    outputs["unknown_sweep_md"] = sweep_md

    summary = _write_evaluation_summary(
        output_dir / "reason_evaluation_summary.md",
        family_metrics=pd.read_csv(outputs["family_metrics_csv"]),
        subtype_metrics=pd.read_csv(outputs["subtype_metrics_csv"]) if "subtype_metrics_csv" in outputs else None,
        predictions_csv=outputs["predictions_csv"],
        unknown_threshold=unknown_threshold,
    )
    outputs["evaluation_summary"] = summary
    e2e = _write_end_to_end_report(
        out_dir=output_dir,
        predictions=pd.read_csv(outputs["predictions_csv"]),
        stage1_scores=stage1_scores,
    )
    outputs.update(e2e)
    figure_outputs = generate_reason_attribution_figures(
        results_dir=output_dir,
        root_dir=root_dir,
        figures_dir=figures_dir,
        has_subtype=predictions.subtype is not None,
    )
    outputs.update(figure_outputs)
    return outputs


def _load_model_config(model_dir: str | Path) -> tuple[Path, Any]:
    from retinal_ood.reason_attribution.features import FeatureExtractionConfig

    model_root = Path(model_dir)
    metadata_path = model_root / "reason_model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = metadata.get("feature_config", {})
    feature_config = FeatureExtractionConfig(
        feature_mode=cfg.get("feature_mode", "image_statistics"),
        image_size=int(cfg.get("image_size", 224)),
        backbone=cfg.get("backbone", "resnet50"),
        layers=tuple(cfg.get("layers", ["layer4"])),
        pretrained=bool(cfg.get("pretrained", True)),
        feature_backend=cfg.get("feature_backend", "torchvision"),
        grayscale_to_rgb=bool(cfg.get("grayscale_to_rgb", True)),
        normalize=cfg.get("normalize"),
        device=cfg.get("device", "cpu"),
    )
    return model_root / metadata.get("model_path", "reason_model.npz"), feature_config


def _write_evaluation_summary(
    path: Path,
    *,
    family_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame | None,
    predictions_csv: Path,
    unknown_threshold: float,
) -> Path:
    family_accuracy = _metric_value(family_metrics, "accuracy", "overall")
    family_macro_f1 = _metric_value(family_metrics, "macro_f1", "overall")
    family_f1 = family_metrics[family_metrics["metric"] == "f1"].copy()
    hardest_family = _hardest_class(family_f1)
    predictions = pd.read_csv(predictions_csv)
    unknown_rate = float((predictions["pred_reason_family"].astype(str) == "unknown_ood").mean())
    lines = [
        "# Reason Attribution Evaluation Summary",
        "",
        "Stage 2 was evaluated on OOD-only rows as a post-hoc explanation module. "
        "Stage 1 remains the ID-only unsupervised binary OOD gatekeeper.",
        "",
        f"- unknown threshold gamma: {unknown_threshold:.2f}",
        f"- reason family accuracy: {family_accuracy:.4f}",
        f"- reason family macro-F1: {family_macro_f1:.4f}",
        f"- hardest reason family by F1: {hardest_family}",
        f"- unknown_ood rate at gamma={unknown_threshold:.2f}: {unknown_rate:.4f}",
    ]
    if subtype_metrics is not None:
        subtype_accuracy = _metric_value(subtype_metrics, "accuracy", "overall")
        subtype_macro_f1 = _metric_value(subtype_metrics, "macro_f1", "overall")
        subtype_f1 = subtype_metrics[subtype_metrics["metric"] == "f1"].copy()
        lines.extend(
            [
                f"- subtype accuracy: {subtype_accuracy:.4f}",
                f"- subtype macro-F1: {subtype_macro_f1:.4f}",
                f"- hardest subtype by F1: {_hardest_class(subtype_f1)}",
            ]
        )
    else:
        lines.append("- subtype classifier: not trained for this run")
    lines.extend(
        [
            "",
            "Reason labels are likely explanations for rejection, not clinical diagnoses.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_end_to_end_report(
    *,
    out_dir: Path,
    predictions: pd.DataFrame,
    stage1_scores: str | Path | None,
) -> dict[str, Path]:
    csv_path = out_dir / "end_to_end_reason_attribution.csv"
    md_path = out_dir / "end_to_end_reason_attribution.md"
    if stage1_scores is None:
        table = pd.DataFrame(
            [
                {
                    "metric": "end_to_end_reason_accuracy",
                    "value": np.nan,
                    "notes": (
                        "Stage 2 was evaluated as a post-hoc module on ground-truth OOD rows. "
                        "Full end-to-end explanation accuracy requires Stage 1 prediction files."
                    ),
                }
            ]
        )
    else:
        scores = pd.read_csv(stage1_scores)
        merged = predictions.merge(
            scores[["image_path", "prediction"]],
            on="image_path",
            how="left",
            suffixes=("", "_stage1"),
        )
        stage1_rejected = merged["prediction"].fillna(0).astype(int) == 1
        reason_correct = merged["true_reason_family"].astype(str) == merged["pred_reason_family"].astype(str)
        table = pd.DataFrame(
            [
                {
                    "metric": "end_to_end_reason_accuracy",
                    "value": float((stage1_rejected & reason_correct).mean()),
                    "notes": "OOD correctly rejected by Stage 1 and reason correctly classified by Stage 2.",
                }
            ]
        )
    table.to_csv(csv_path, index=False)
    md_path.write_text("# End-to-End Reason Attribution\n\n" + dataframe_to_markdown(table) + "\n", encoding="utf-8")
    return {"end_to_end_csv": csv_path, "end_to_end_md": md_path}


def generate_reason_attribution_figures(
    *,
    results_dir: str | Path,
    root_dir: str | Path,
    figures_dir: str | Path,
    has_subtype: bool,
) -> dict[str, Path]:
    results = Path(results_dir)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = {
        "figure_two_stage_pipeline": _plot_two_stage_pipeline(figures / "figure_two_stage_pipeline.png"),
        "figure_reason_family_confusion_matrix": _plot_confusion_matrix(
            results / "reason_family_confusion_matrix.csv",
            figures / "figure_reason_family_confusion_matrix.png",
            title="Reason Family Confusion Matrix",
        ),
        "figure_reason_attribution_examples": _plot_examples(
            results / "reason_predictions.csv",
            root_dir=Path(root_dir),
            path=figures / "figure_reason_attribution_examples.png",
        ),
        "figure_reason_unknown_threshold": _plot_unknown_threshold(
            results / "reason_unknown_threshold_sweep.csv",
            figures / "figure_reason_unknown_threshold.png",
        ),
    }
    subtype_confusion = results / "reason_subtype_confusion_matrix.csv"
    if has_subtype and subtype_confusion.exists():
        outputs["figure_reason_subtype_confusion_matrix"] = _plot_confusion_matrix(
            subtype_confusion,
            figures / "figure_reason_subtype_confusion_matrix.png",
            title="Reason Subtype Confusion Matrix",
        )
    else:
        outputs["figure_reason_subtype_confusion_matrix"] = _plot_no_subtype(
            figures / "figure_reason_subtype_confusion_matrix.png"
        )
    outputs["figure_index"] = _write_figure_index(figures)
    outputs["caption_suggestions"] = _write_caption_suggestions(figures)
    outputs["figure_selection_guide"] = _write_selection_guide(figures)
    return outputs


def _plot_two_stage_pipeline(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
    ax.axis("off")
    boxes = [
        (0.08, 0.55, "Input image"),
        (0.29, 0.55, "Stage 1\nID-only OOD gatekeeper"),
        (0.50, 0.73, "ACCEPT\nvalid FAF"),
        (0.50, 0.36, "REJECT\nOOD / invalid"),
        (0.71, 0.36, "Stage 2\nreason attribution"),
        (0.90, 0.36, "Reason family\n+ optional subtype\n+ unknown_ood"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#2f4858"},
            transform=ax.transAxes,
        )
    arrows = [
        ((0.14, 0.55), (0.21, 0.55)),
        ((0.39, 0.61), (0.45, 0.71)),
        ((0.39, 0.49), (0.45, 0.38)),
        ((0.56, 0.36), (0.64, 0.36)),
        ((0.78, 0.36), (0.84, 0.36)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8}, xycoords=ax.transAxes)
    ax.text(
        0.36,
        0.10,
        "OOD taxonomy labels are used only for Stage 2 explanation after rejection.",
        ha="center",
        fontsize=9,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrix(csv_path: Path, path: Path, *, title: str) -> Path:
    matrix = pd.read_csv(csv_path, index_col=0)
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(matrix.columns)), max(4, 0.45 * len(matrix))), dpi=200)
    im = ax.imshow(values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, str(int(values[row, col])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_examples(predictions_csv: Path, *, root_dir: Path, path: Path) -> Path:
    predictions = pd.read_csv(predictions_csv)
    selected = (
        predictions.sort_values("reason_confidence", ascending=False)
        .groupby("true_reason_family", group_keys=False)
        .head(2)
        .head(6)
    )
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=200)
    for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
        image_path = root_dir / str(row["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
            ax.imshow(image)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "image missing", ha="center", va="center")
        ax.axis("off")
        subtype = row.get("pred_reason_subtype", "")
        subtitle = f"{row['pred_reason_family']} ({row['reason_confidence']:.2f})"
        if isinstance(subtype, str) and subtype:
            subtitle += f"\n{subtype}"
        ax.set_title(subtitle, fontsize=8)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Representative rejected-input reason attributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_unknown_threshold(csv_path: Path, path: Path) -> Path:
    sweep = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
    ax.plot(sweep["unknown_threshold"], sweep["coverage"], marker="o", label="coverage")
    ax.plot(sweep["unknown_threshold"], sweep["known_accuracy"], marker="s", label="accuracy on known outputs")
    ax.set_xlabel("Unknown threshold gamma")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unknown-OOD threshold trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_no_subtype(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    ax.axis("off")
    ax.text(0.5, 0.5, "Subtype classifier was not trained for this run.", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_figure_index(figures: Path) -> Path:
    rows = [
        ("figure_two_stage_pipeline.png", "Two-stage gatekeeper and reason attribution pipeline"),
        ("figure_reason_family_confusion_matrix.png", "Reason family confusion matrix"),
        ("figure_reason_subtype_confusion_matrix.png", "Reason subtype confusion matrix"),
        ("figure_reason_attribution_examples.png", "Representative reason attribution examples"),
        ("figure_reason_unknown_threshold.png", "Unknown-threshold coverage and accuracy trade-off"),
    ]
    path = figures / "figure_index.md"
    table = pd.DataFrame(rows, columns=["path", "title"])
    path.write_text("# Reason Attribution Figure Index\n\n" + dataframe_to_markdown(table) + "\n", encoding="utf-8")
    return path


def _write_caption_suggestions(figures: Path) -> Path:
    path = figures / "caption_suggestions.md"
    path.write_text(
        "\n".join(
            [
                "# Reason Attribution Caption Suggestions",
                "",
                "- `figure_two_stage_pipeline.png`: Two-stage FAF OOD gatekeeper extension. Stage 1 remains an ID-only unsupervised accept/reject gatekeeper; Stage 2 explains rejected inputs only.",
                "- `figure_reason_family_confusion_matrix.png`: Confusion matrix for Stage 2 reason family attribution across modality shift, sensory artifact, and semantic outlier classes.",
                "- `figure_reason_subtype_confusion_matrix.png`: Optional subtype confusion matrix showing fine-grained reason attribution performance.",
                "- `figure_reason_attribution_examples.png`: Representative rejected inputs with predicted explanation labels and confidence values.",
                "- `figure_reason_unknown_threshold.png`: Coverage/accuracy trade-off induced by the low-confidence `unknown_ood` threshold.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_selection_guide(figures: Path) -> Path:
    path = figures / "figure_selection_guide.md"
    path.write_text(
        "\n".join(
            [
                "# Reason Attribution Figure Selection Guide",
                "",
                "| figure | include | suggested chapter | notes |",
                "| --- | --- | --- | --- |",
                "| figure_two_stage_pipeline.png | must_include | Methodology | Clarifies Stage 1/Stage 2 boundary. |",
                "| figure_reason_family_confusion_matrix.png | must_include | Results | Main Stage 2 explanation result. |",
                "| figure_reason_subtype_confusion_matrix.png | optional | Appendix | Fine-grained subtype performance can be visually dense. |",
                "| figure_reason_attribution_examples.png | optional | Discussion | Useful qualitative explanation examples. |",
                "| figure_reason_unknown_threshold.png | optional | Limitations | Shows unknown_ood coverage trade-off. |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _metric_value(metrics: pd.DataFrame, metric: str, class_name: str) -> float:
    row = metrics[(metrics["metric"] == metric) & (metrics["class"] == class_name)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["value"])


def _hardest_class(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "not_available"
    row = metrics.sort_values("value", ascending=True, kind="stable").iloc[0]
    return f"{row['class']} (F1={float(row['value']):.4f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate optional Stage 2 reason attribution.")
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", default="reports/dissertation_figures/reason_attribution")
    parser.add_argument("--unknown-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stage1-scores", help="Optional Stage 1 scores.csv with image_path,prediction columns")
    args = parser.parse_args()
    outputs = evaluate_reason_attribution(
        test_manifest=args.test_manifest,
        root_dir=args.root_dir,
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        figures_dir=args.figures_dir,
        unknown_threshold=args.unknown_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stage1_scores=args.stage1_scores,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
