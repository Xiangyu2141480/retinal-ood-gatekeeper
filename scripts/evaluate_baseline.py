#!/usr/bin/env python
"""Evaluate lightweight baselines with the shared OOD reporting pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import run_evaluation
from retinal_ood.baselines.feature_distance import FeatureDistanceDetector
from retinal_ood.baselines.image_statistics import ImageStatisticsDetector
from retinal_ood.utils.io import read_yaml


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def evaluate_from_config(config: dict[str, Any], checkpoint: str | Path) -> Path:
    """Evaluate a baseline checkpoint and return the evaluation output directory."""
    model_cfg = config.get("model", {})
    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg["save_heatmaps"] = False
    detector = _load_detector(model_cfg, checkpoint)
    return run_evaluation(config, checkpoint, detector=detector)


def _load_detector(model_cfg: dict[str, Any], checkpoint: str | Path) -> ImageStatisticsDetector | FeatureDistanceDetector:
    model_name = str(model_cfg.get("name", "")).lower()
    if model_name == "image_statistics":
        return ImageStatisticsDetector.load(checkpoint)
    if model_name in {"global_feature_knn", "mahalanobis_feature"}:
        return FeatureDistanceDetector.load(
            checkpoint,
            device=_resolve_device(str(model_cfg.get("device", "auto"))),
        )
    raise ValueError(f"Unsupported baseline model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Baseline YAML config")
    parser.add_argument("--checkpoint", required=True, help="Baseline .npz checkpoint")
    args = parser.parse_args()

    out_dir = evaluate_from_config(read_yaml(args.config), args.checkpoint)
    print(f"Saved baseline evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
