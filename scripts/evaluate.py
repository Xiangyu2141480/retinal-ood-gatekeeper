#!/usr/bin/env python
"""Evaluate a trained PatchCore OOD detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.dataset import ManifestImageDataset
from retinal_ood.data.transforms import build_transforms
from retinal_ood.evaluation.reporting import (
    build_metrics_report,
    build_score_rows,
    save_roc_pr_plots,
    save_scores_csv,
)
from retinal_ood.evaluation.thresholds import threshold_from_id_quantile
from retinal_ood.models.patchcore import PatchCoreDetector
from retinal_ood.utils.io import read_yaml, write_json
from retinal_ood.visualization.heatmaps import save_topk_heatmaps


class ScoreDetector(Protocol):
    def predict_scores(
        self,
        dataloader: DataLoader,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        ...


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _require_config_value(config: dict[str, Any], section: str, key: str) -> Any:
    value = config.get(section, {}).get(key)
    if value in (None, ""):
        raise ValueError(f"{section}.{key} must be set")
    return value


def _make_dataset(config: dict[str, Any], manifest_key: str) -> ManifestImageDataset:
    data_cfg = config.get("data", {})
    transform = build_transforms(
        image_size=int(data_cfg.get("image_size", 224)),
        grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", True)),
        normalize=data_cfg.get("normalize", "imagenet"),
    )
    return ManifestImageDataset(
        _require_config_value(config, "data", manifest_key),
        root_dir=data_cfg.get("root_dir"),
        transform=transform,
    )


def _make_loader(config: dict[str, Any], dataset: ManifestImageDataset) -> DataLoader:
    data_cfg = config.get("data", {})
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )


def _dataset_metadata(dataset: ManifestImageDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in dataset.df.to_dict(orient="records"):
        metadata = dict(row)
        metadata["resolved_image_path"] = str(dataset._resolve_path(row["image_path"]))
        rows.append(metadata)
    return rows


def _score_dataset(
    detector: ScoreDetector,
    config: dict[str, Any],
    dataset: ManifestImageDataset,
    *,
    return_patch_maps: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[np.ndarray] | None]:
    labels = dataset.df["label"].astype(int).to_numpy()
    try:
        result = detector.predict_scores(
            _make_loader(config, dataset),
            return_patch_maps=return_patch_maps,
        )
    except TypeError as exc:
        if return_patch_maps:
            raise TypeError(
                "Heatmap export requires detector.predict_scores(..., return_patch_maps=True)"
            ) from exc
        raise

    patch_maps: list[np.ndarray] | None = None
    raw_scores: Any = result
    if return_patch_maps:
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "Heatmap export requires detector.predict_scores(..., return_patch_maps=True) "
                "to return (scores, patch_maps)"
            )
        raw_scores, raw_patch_maps = result
        patch_maps = list(raw_patch_maps)
        if len(patch_maps) != labels.shape[0]:
            raise ValueError(
                f"Detector returned {len(patch_maps)} patch maps for {labels.shape[0]} images"
            )

    scores = np.asarray(raw_scores, dtype=float)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"Detector returned {scores.shape[0]} scores for {labels.shape[0]} images")
    return labels, scores, _dataset_metadata(dataset), patch_maps


def _score_optional_validation_id(
    detector: ScoreDetector,
    config: dict[str, Any],
) -> tuple[np.ndarray | None, str]:
    data_cfg = config.get("data", {})
    if data_cfg.get("val_manifest"):
        val_dataset = _make_dataset(config, "val_manifest").id_subset()
        if len(val_dataset) == 0:
            raise ValueError("data.val_manifest has no ID rows for threshold calibration")
        _, val_scores, _, _ = _score_dataset(detector, config, val_dataset)
        return val_scores, "validation_id_quantile"
    return None, "test_id_quantile"


def _load_patchcore_detector(config: dict[str, Any], checkpoint: str | Path) -> PatchCoreDetector:
    model_cfg = config.get("model", {})
    detector = PatchCoreDetector.load(checkpoint)
    device = _resolve_device(str(model_cfg.get("device", detector.config.device)))
    detector.config.device = device
    detector.device = torch.device(device)
    if detector.feature_extractor is not None:
        detector.feature_extractor.to(detector.device)
        detector.feature_extractor.eval()
    return detector


def run_evaluation(
    config: dict[str, Any],
    checkpoint: str | Path,
    *,
    detector: ScoreDetector | None = None,
) -> Path:
    """Run evaluation and return the output directory."""
    project_cfg = config.get("project", {})
    eval_cfg = config.get("evaluation", {})
    output_cfg = config.get("output", {})

    detector = detector or _load_patchcore_detector(config, checkpoint)
    id_dataset = _make_dataset(config, "test_id_manifest")
    ood_dataset = _make_dataset(config, "test_ood_manifest")
    save_heatmaps = bool(eval_cfg.get("save_heatmaps", output_cfg.get("save_heatmaps", False)))

    id_labels, id_scores, id_metadata, id_patch_maps = _score_dataset(
        detector,
        config,
        id_dataset,
        return_patch_maps=save_heatmaps,
    )
    ood_labels, ood_scores, ood_metadata, ood_patch_maps = _score_dataset(
        detector,
        config,
        ood_dataset,
        return_patch_maps=save_heatmaps,
    )
    labels = np.concatenate([id_labels, ood_labels])
    scores = np.concatenate([id_scores, ood_scores])
    metadata_rows = id_metadata + ood_metadata

    val_scores, threshold_source = _score_optional_validation_id(detector, config)
    threshold_scores = val_scores if val_scores is not None else id_scores
    quantile = float(eval_cfg.get("validation_quantile_for_id_threshold", 0.95))
    if eval_cfg.get("threshold") is not None:
        threshold = float(eval_cfg["threshold"])
        threshold_source = "configured_threshold"
    else:
        threshold = threshold_from_id_quantile(threshold_scores, quantile)

    ood_types = [
        str(row.get("ood_type", row.get("category", "unknown")))
        for row in metadata_rows
    ]
    metrics = build_metrics_report(
        labels,
        scores,
        ood_types,
        threshold=threshold,
        threshold_source=threshold_source,
    )
    metrics["threshold"]["quantile"] = quantile

    run_name = str(project_cfg.get("run_name", "evaluation"))
    out_dir = Path(output_cfg.get("runs_dir", "runs")) / run_name / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(eval_cfg.get("save_score_csv", True)):
        scores_df = build_score_rows(metadata_rows, labels, scores, threshold=threshold)
        save_scores_csv(out_dir / "scores.csv", scores_df)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "resolved_evaluation_config.json", config)

    if bool(eval_cfg.get("save_plots", True)):
        save_roc_pr_plots(labels, scores, out_dir)

    if save_heatmaps:
        patch_maps = (id_patch_maps or []) + (ood_patch_maps or [])
        heatmap_rows = save_topk_heatmaps(
            metadata_rows,
            patch_maps,
            labels,
            scores,
            threshold=threshold,
            out_dir=out_dir / "heatmaps",
            top_k=int(eval_cfg.get("heatmap_top_k", 5)),
            alpha=float(eval_cfg.get("heatmap_alpha", 0.45)),
            cmap=str(eval_cfg.get("heatmap_cmap", "magma")),
        )
        metrics["heatmaps"] = {
            "directory": "heatmaps",
            "top_k": int(eval_cfg.get("heatmap_top_k", 5)),
            "selected_count": len(heatmap_rows),
        }
        write_json(out_dir / "metrics.json", metrics)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-heatmaps", action="store_true", help="Save top-K heatmaps")
    parser.add_argument("--no-heatmaps", action="store_true", help="Disable heatmap export")
    parser.add_argument("--heatmap-top-k", type=int, help="Examples per FP/FN/TP/TN bucket")
    args = parser.parse_args()
    if args.save_heatmaps and args.no_heatmaps:
        parser.error("--save-heatmaps and --no-heatmaps are mutually exclusive")

    config = read_yaml(args.config)
    eval_cfg = config.setdefault("evaluation", {})
    if args.save_heatmaps:
        eval_cfg["save_heatmaps"] = True
    if args.no_heatmaps:
        eval_cfg["save_heatmaps"] = False
    if args.heatmap_top_k is not None:
        eval_cfg["heatmap_top_k"] = args.heatmap_top_k

    out_dir = run_evaluation(config, args.checkpoint)
    print(f"Saved evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
