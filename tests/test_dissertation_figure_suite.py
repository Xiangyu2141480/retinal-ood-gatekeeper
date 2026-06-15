from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from retinal_ood.visualization.dissertation_suite import generate_dissertation_figure_suite


def _write_scores(path: Path, *, score_offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "image_path": "images/dissertation_v1/id/test/id_001.png",
            "label": 0,
            "ood_type": "id",
            "ood_subtype": "id",
            "score": 0.10 + score_offset,
            "prediction": 0,
            "threshold": 0.50,
        },
        {
            "image_path": "images/dissertation_v1/id/test/id_002.png",
            "label": 0,
            "ood_type": "id",
            "ood_subtype": "id",
            "score": 0.20 + score_offset,
            "prediction": 0,
            "threshold": 0.50,
        },
        {
            "image_path": "images/dissertation_v1/ood/sensory_artifact/text_watermark/a.png",
            "label": 1,
            "ood_type": "sensory_artifact",
            "ood_subtype": "text_watermark",
            "score": 0.86 + score_offset,
            "prediction": 1,
            "threshold": 0.50,
        },
        {
            "image_path": "images/dissertation_v1/ood/modality_shift/colour_fundus/b.png",
            "label": 1,
            "ood_type": "modality_shift",
            "ood_subtype": "colour_fundus",
            "score": 0.82 + score_offset,
            "prediction": 1,
            "threshold": 0.50,
        },
        {
            "image_path": "images/dissertation_v1/ood/semantic_outlier/cifar10_natural/c.png",
            "label": 1,
            "ood_type": "semantic_outlier",
            "ood_subtype": "cifar10_natural",
            "score": 0.78 + score_offset,
            "prediction": 1,
            "threshold": 0.50,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_fake_heatmap(root: Path, outcome: str, image_path: str, filename: str) -> None:
    heatmap_dir = root / "runs" / "patchcore_layer2_layer3" / "evaluation" / "selected_heatmaps"
    panel_dir = heatmap_dir / outcome
    panel_dir.mkdir(parents=True, exist_ok=True)
    for suffix, color in {
        "original": (60, 70, 80),
        "heatmap": (180, 60, 80),
        "overlay": (30, 180, 40),
    }.items():
        Image.new("RGB", (32, 24), color=color).save(panel_dir / f"{filename}_{suffix}.png")

    manifest = heatmap_dir / "heatmap_manifest.csv"
    new_file = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "outcome",
                "rank",
                "sample_index",
                "image_path",
                "label",
                "score",
                "prediction",
                "threshold",
                "original_file",
                "heatmap_file",
                "overlay_file",
            ],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "outcome": outcome,
                "rank": 1,
                "sample_index": 1,
                "image_path": image_path,
                "label": 1,
                "score": 0.86,
                "prediction": 1,
                "threshold": 0.5,
                "original_file": f"{outcome}/{filename}_original.png",
                "heatmap_file": f"{outcome}/{filename}_heatmap.png",
                "overlay_file": f"{outcome}/{filename}_overlay.png",
            }
        )


def _write_report_fixture(root: Path) -> Path:
    reports_dir = root / "reports" / "dissertation_results" / "primary_balanced_by_subtype"
    reports_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment": "autoencoder_baseline",
                "model": "conv_autoencoder",
                "backbone": "",
                "layers": "",
                "auroc": 0.74,
                "auprc": 0.70,
                "fpr_at_95_tpr": 0.42,
                "threshold_at_95_tpr": 0.66,
                "threshold": 0.50,
                "threshold_source": "validation_id_quantile",
                "id_false_rejection_rate": 0.10,
                "id_count": 2,
                "ood_count": 3,
                "warnings": "",
                "metrics_file": "runs/autoencoder_baseline/evaluation/metrics.json",
            },
            {
                "experiment": "patchcore_layer1",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer1",
                "auroc": 0.80,
                "auprc": 0.77,
                "fpr_at_95_tpr": 0.33,
                "threshold_at_95_tpr": 0.63,
                "threshold": 0.50,
                "threshold_source": "validation_id_quantile",
                "id_false_rejection_rate": 0.05,
                "id_count": 2,
                "ood_count": 3,
                "warnings": "",
                "metrics_file": "runs/patchcore_layer1/evaluation/metrics.json",
            },
            {
                "experiment": "patchcore_layer2_layer3",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer2+layer3",
                "auroc": 0.94,
                "auprc": 0.91,
                "fpr_at_95_tpr": 0.12,
                "threshold_at_95_tpr": 0.58,
                "threshold": 0.50,
                "threshold_source": "validation_id_quantile",
                "id_false_rejection_rate": 0.02,
                "id_count": 2,
                "ood_count": 3,
                "warnings": "synthetic fallback only",
                "metrics_file": "runs/patchcore_layer2_layer3/evaluation/metrics.json",
            },
        ]
    ).to_csv(reports_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": "patchcore_layer2_layer3",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer2+layer3",
                "ood_type": ood_type,
                "auroc": value,
                "auprc": value - 0.02,
                "fpr_at_95_tpr": 1.0 - value,
                "id_count": 2,
                "ood_count": 1,
                "metrics_file": "runs/patchcore_layer2_layer3/evaluation/metrics.json",
            }
            for ood_type, value in [
                ("modality_shift", 0.96),
                ("sensory_artifact", 0.90),
                ("semantic_outlier", 0.98),
            ]
        ]
    ).to_csv(reports_dir / "per_ood_type_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": "patchcore_layer2_layer3",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer2+layer3",
                "ood_type": ood_type,
                "ood_subtype": subtype,
                "auroc": value,
                "auprc": value - 0.02,
                "fpr_at_95_tpr": 1.0 - value,
                "id_count": 2,
                "ood_count": 1,
                "metrics_file": "runs/patchcore_layer2_layer3/evaluation/metrics.json",
            }
            for ood_type, subtype, value in [
                ("sensory_artifact", "text_watermark", 0.90),
                ("modality_shift", "colour_fundus", 0.96),
                ("semantic_outlier", "cifar10_natural", 0.98),
            ]
        ]
    ).to_csv(reports_dir / "per_ood_subtype_metrics.csv", index=False)
    _write_scores(reports_dir / "runs" / "autoencoder_baseline" / "evaluation" / "scores.csv", score_offset=-0.03)
    _write_scores(reports_dir / "runs" / "patchcore_layer1" / "evaluation" / "scores.csv", score_offset=-0.01)
    _write_scores(reports_dir / "runs" / "patchcore_layer2_layer3" / "evaluation" / "scores.csv", score_offset=0.00)
    _write_fake_heatmap(
        reports_dir,
        "tp",
        "images/dissertation_v1/ood/sensory_artifact/text_watermark/a.png",
        "sensory",
    )
    _write_fake_heatmap(
        reports_dir,
        "tp",
        "images/dissertation_v1/ood/modality_shift/colour_fundus/b.png",
        "modality",
    )
    return reports_dir


def test_generate_dissertation_figure_suite_writes_priority_outputs(tmp_path: Path):
    reports_dir = _write_report_fixture(tmp_path)
    out_dir = tmp_path / "reports" / "dissertation_figures"

    outputs = generate_dissertation_figure_suite(reports_dir, out_dir, dpi=80)

    expected = {
        "figure_roc_overall_model_comparison.png",
        "figure_pr_overall_model_comparison.png",
        "figure_layer_ablation_patchcore.png",
        "figure_per_ood_type_comparison.png",
        "figure_per_ood_subtype_comparison.png",
        "figure_score_distribution_with_threshold.png",
        "figure_heatmaps_sensory_artifact_examples.png",
        "figure_heatmaps_modality_examples.png",
        "figure_dataset_taxonomy.png",
        "figure_system_pipeline_overview.png",
        "figure_patchcore_method.png",
        "figure_experiment_workflow.png",
        "caption_suggestions.md",
        "figure_index.md",
    }
    output_names = {path.name for path in outputs.files.values()}
    assert expected <= output_names
    for name in expected - {"caption_suggestions.md", "figure_index.md"}:
        path = out_dir / name
        assert path.exists()
        assert Image.open(path).size[0] > 0

    index_text = (out_dir / "figure_index.md").read_text(encoding="utf-8")
    captions_text = (out_dir / "caption_suggestions.md").read_text(encoding="utf-8")
    assert "overall ROC comparison" in index_text
    assert "PatchCore method" in index_text
    assert "synthetic fallback" in captions_text
    assert "patient_id" not in index_text + captions_text
    sensory_pixels = np.asarray(
        Image.open(out_dir / "figure_heatmaps_sensory_artifact_examples.png").convert("RGB")
    )
    assert (
        (sensory_pixels[:, :, 0] < 80)
        & (sensory_pixels[:, :, 1] > 140)
        & (sensory_pixels[:, :, 2] < 90)
    ).any()
