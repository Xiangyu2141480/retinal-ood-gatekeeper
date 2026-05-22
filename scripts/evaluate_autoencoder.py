#!/usr/bin/env python
"""Evaluate a trained autoencoder baseline with the shared OOD reporting pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import run_evaluation
from retinal_ood.models.autoencoder import (
    ConvAutoEncoder,
    load_autoencoder_checkpoint,
    predict_reconstruction_scores,
)
from retinal_ood.utils.io import read_yaml


class AutoencoderScoreDetector:
    """Adapter that exposes autoencoder reconstruction error through predict_scores()."""

    def __init__(self, model: ConvAutoEncoder, *, device: str | torch.device) -> None:
        self.model = model
        self.device = device

    def predict_scores(
        self,
        dataloader: DataLoader,
        **kwargs: Any,
    ) -> np.ndarray:
        if kwargs.get("return_patch_maps"):
            raise TypeError("Autoencoder evaluation does not provide PatchCore patch heatmaps")
        return predict_reconstruction_scores(self.model, dataloader, device=self.device)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def evaluate_from_config(config: dict[str, Any], checkpoint: str | Path) -> Path:
    """Evaluate an autoencoder checkpoint and return the evaluation output directory."""
    model_cfg = config.get("model", {})
    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg["save_heatmaps"] = False
    device = _resolve_device(str(model_cfg.get("device", "auto")))
    model = load_autoencoder_checkpoint(checkpoint, map_location=device)
    detector = AutoencoderScoreDetector(model, device=device)
    return run_evaluation(config, checkpoint, detector=detector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Autoencoder YAML config")
    parser.add_argument("--checkpoint", required=True, help="Autoencoder .pt checkpoint")
    args = parser.parse_args()

    out_dir = evaluate_from_config(read_yaml(args.config), args.checkpoint)
    print(f"Saved autoencoder evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
