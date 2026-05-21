import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class MeanIntensityDetector:
    def predict_scores(self, dataloader, *, return_patch_maps=False, **_kwargs):
        scores: list[float] = []
        patch_maps: list[np.ndarray] = []
        for images, _labels, _metadata in dataloader:
            batch_scores = images.mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(float)
            scores.extend(batch_scores.tolist())
            if return_patch_maps:
                patch_maps.extend(
                    np.full((2, 2), float(score), dtype=np.float32)
                    for score in batch_scores
                )
        scores_array = np.asarray(scores, dtype=float)
        if return_patch_maps:
            return scores_array, patch_maps
        return scores_array


def _load_evaluate_module():
    script_path = Path("scripts/evaluate.py")
    spec = importlib.util.spec_from_file_location("evaluate_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_toy_image(path: Path, value: int) -> None:
    Image.new("RGB", (16, 16), color=(value, value, value)).save(path)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_evaluation_writes_scores_metrics_and_plots(tmp_path: Path):
    evaluate = _load_evaluate_module()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name, value in [
        ("id_0.png", 0),
        ("id_1.png", 0),
        ("val_0.png", 20),
        ("val_1.png", 20),
        ("ood_0.png", 255),
        ("ood_1.png", 220),
    ]:
        _write_toy_image(image_dir / name, value)

    test_id_manifest = tmp_path / "test_id.csv"
    val_manifest = tmp_path / "val_id.csv"
    test_ood_manifest = tmp_path / "test_ood.csv"
    _write_manifest(
        test_id_manifest,
        [
            {
                "image_path": "images/id_0.png",
                "label": 0,
                "split": "test",
                "source": "toy",
                "ood_type": "id",
                "patient_id": "should-not-be-written",
            },
            {
                "image_path": "images/id_1.png",
                "label": 0,
                "split": "test",
                "source": "toy",
                "ood_type": "id",
            },
        ],
    )
    _write_manifest(
        val_manifest,
        [
            {
                "image_path": "images/val_0.png",
                "label": 0,
                "split": "val",
                "source": "toy",
                "ood_type": "id",
            },
            {
                "image_path": "images/val_1.png",
                "label": 0,
                "split": "val",
                "source": "toy",
                "ood_type": "id",
            },
        ],
    )
    _write_manifest(
        test_ood_manifest,
        [
            {
                "image_path": "images/ood_0.png",
                "label": 1,
                "split": "test",
                "source": "toy",
                "ood_type": "modality_shift",
            },
            {
                "image_path": "images/ood_1.png",
                "label": 1,
                "split": "test",
                "source": "toy",
                "ood_type": "semantic_outlier",
            },
        ],
    )
    config = {
        "project": {"run_name": "toy_eval"},
        "data": {
            "root_dir": str(tmp_path),
            "test_id_manifest": str(test_id_manifest),
            "val_manifest": str(val_manifest),
            "test_ood_manifest": str(test_ood_manifest),
            "image_size": 16,
            "grayscale_to_rgb": False,
            "normalize": "minmax",
            "batch_size": 2,
        },
        "evaluation": {
            "validation_quantile_for_id_threshold": 0.95,
            "save_score_csv": True,
            "save_plots": True,
            "save_heatmaps": True,
            "heatmap_top_k": 1,
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }

    out_dir = evaluate.run_evaluation(
        config,
        checkpoint=tmp_path / "unused.npz",
        detector=MeanIntensityDetector(),
    )

    scores_path = out_dir / "scores.csv"
    metrics_path = out_dir / "metrics.json"
    assert scores_path.exists()
    assert metrics_path.exists()
    assert (out_dir / "roc_curve.png").exists()
    assert (out_dir / "pr_curve.png").exists()

    scores_df = pd.read_csv(scores_path)
    assert len(scores_df) == 4
    assert "patient_id" not in scores_df.columns
    assert set(scores_df["ood_type"]) == {"id", "modality_shift", "semantic_outlier"}
    assert scores_df["prediction"].tolist() == [0, 0, 1, 1]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["global"]["auroc"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert metrics["threshold"]["source"] == "validation_id_quantile"
    assert set(metrics["per_ood_type"]) == {"modality_shift", "semantic_outlier"}
    assert metrics["heatmaps"] == {"directory": "heatmaps", "top_k": 1, "selected_count": 2}

    heatmap_dir = out_dir / "heatmaps"
    assert (heatmap_dir / "colorbar.png").exists()
    assert (heatmap_dir / "heatmap_normalization.json").exists()
    heatmap_manifest = pd.read_csv(heatmap_dir / "heatmap_manifest.csv")
    assert len(heatmap_manifest) == 2
    assert "patient_id" not in heatmap_manifest.columns
    assert set(heatmap_manifest["outcome"]) == {"tp", "tn"}
    for overlay_file in heatmap_manifest["overlay_file"]:
        assert (heatmap_dir / overlay_file).exists()
