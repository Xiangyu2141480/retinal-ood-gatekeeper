"""Single-image PatchCore gatekeeper inference.

The gatekeeper is intentionally binary: it accepts likely valid FAF images and rejects
OOD/invalid inputs. It does not diagnose retinal disease or assign disease classes.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader

from retinal_ood.data.transforms import build_transforms
from retinal_ood.models.patchcore import PatchCoreDetector
from retinal_ood.visualization.heatmaps import compute_heatmap_normalization, overlay_heatmap

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ImageScoreDetector(Protocol):
    """Protocol for detectors that can score a DataLoader and optionally return patch maps."""

    def predict_scores(
        self,
        dataloader: DataLoader,
        *,
        return_patch_maps: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        ...


@dataclass(frozen=True)
class GatekeeperResult:
    filename: str
    score: float
    threshold: float | None
    prediction: int | None
    decision: str
    verdict: str
    overlay_png_base64: str | None
    heatmap_png_base64: str | None
    warning: str | None = None


class SingleImageGatekeeper:
    """Run one uploaded image through a fitted PatchCore detector."""

    def __init__(
        self,
        config: dict[str, Any],
        detector: ImageScoreDetector,
        *,
        threshold: float | None,
        heatmap_alpha: float = 0.45,
        heatmap_cmap: str = "magma",
        display_max_side: int = 1024,
    ) -> None:
        self.config = config
        self.detector = detector
        self.threshold = threshold
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_cmap = heatmap_cmap
        self.display_max_side = display_max_side
        data_cfg = config.get("data", {})
        self.transform = build_transforms(
            image_size=int(data_cfg.get("image_size", 224)),
            grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", True)),
            normalize=data_cfg.get("normalize", "imagenet"),
        )

    def predict_bytes(self, filename: str, image_bytes: bytes) -> GatekeeperResult:
        """Score an uploaded image file in memory."""
        image = _decode_supported_image(filename, image_bytes)
        tensor = self.transform(image)
        dataloader = DataLoader([(tensor, torch.tensor(0))], batch_size=1, shuffle=False)
        result = self.detector.predict_scores(dataloader, return_patch_maps=True)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Detector must return (scores, patch_maps) for the drag-drop app")

        scores, patch_maps = result
        scores = np.asarray(scores, dtype=float)
        if scores.shape[0] != 1 or len(patch_maps) != 1:
            raise ValueError("Detector must return exactly one score and one patch map")

        score = float(scores[0])
        prediction = int(score >= self.threshold) if self.threshold is not None else None
        decision, verdict, warning = _decision_from_score(score, self.threshold)
        heatmap_b64, overlay_b64 = _render_heatmap_pair(
            image,
            patch_maps[0],
            alpha=self.heatmap_alpha,
            cmap=self.heatmap_cmap,
            max_side=self.display_max_side,
        )
        return GatekeeperResult(
            filename=filename,
            score=score,
            threshold=self.threshold,
            prediction=prediction,
            decision=decision,
            verdict=verdict,
            overlay_png_base64=overlay_b64,
            heatmap_png_base64=heatmap_b64,
            warning=warning,
        )


def load_patchcore_gatekeeper(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    threshold: float | None = None,
    metrics_path: str | Path | None = None,
) -> SingleImageGatekeeper:
    """Load a PatchCore detector and resolve its operating threshold."""
    checkpoint_path = Path(checkpoint_path)
    detector = PatchCoreDetector.load(checkpoint_path)
    device = _resolve_device(str(config.get("model", {}).get("device", detector.config.device)))
    detector.config.device = device
    detector.device = torch.device(device)
    if detector.feature_extractor is not None:
        detector.feature_extractor.to(detector.device)
        detector.feature_extractor.eval()

    resolved_threshold = resolve_threshold(
        config,
        threshold=threshold,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
    )
    eval_cfg = config.get("evaluation", {})
    return SingleImageGatekeeper(
        config,
        detector,
        threshold=resolved_threshold,
        heatmap_alpha=float(eval_cfg.get("heatmap_alpha", 0.45)),
        heatmap_cmap=str(eval_cfg.get("heatmap_cmap", "magma")),
    )


def resolve_threshold(
    config: dict[str, Any],
    *,
    threshold: float | None = None,
    metrics_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> float | None:
    """Resolve the operating threshold from CLI, metrics.json, or config."""
    if threshold is not None:
        return float(threshold)

    for candidate in _threshold_metric_candidates(metrics_path, checkpoint_path):
        if candidate.exists():
            metrics = json.loads(candidate.read_text(encoding="utf-8"))
            value = metrics.get("threshold", {}).get("value")
            if value is None:
                raise ValueError(f"metrics file has no threshold.value: {candidate}")
            return float(value)

    config_threshold = config.get("evaluation", {}).get("threshold")
    if config_threshold is not None:
        return float(config_threshold)
    return None


def _threshold_metric_candidates(
    metrics_path: str | Path | None,
    checkpoint_path: str | Path | None,
) -> list[Path]:
    if metrics_path is not None:
        return [Path(metrics_path)]
    if checkpoint_path is None:
        return []
    checkpoint_path = Path(checkpoint_path)
    return [checkpoint_path.parent / "evaluation" / "metrics.json"]


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _decode_supported_image(filename: str, image_bytes: bytes) -> Image.Image:
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported image extension {suffix!r}; expected one of {sorted(SUPPORTED_IMAGE_SUFFIXES)}"
        )
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a supported image") from exc


def _decision_from_score(score: float, threshold: float | None) -> tuple[str, str, str | None]:
    if threshold is None:
        return (
            "score_only_no_threshold",
            "SCORE ONLY",
            "No threshold was provided; run evaluation first or pass --threshold/--metrics.",
        )
    if score >= threshold:
        return "reject_ood_or_invalid", "REJECT: OOD / invalid input", None
    return "accept_valid_faf", "ACCEPT: likely valid FAF", None


def _render_heatmap_pair(
    image: Image.Image,
    anomaly_map: np.ndarray,
    *,
    alpha: float,
    cmap: str,
    max_side: int,
) -> tuple[str, str]:
    display_image = _resize_for_display(image, max_side=max_side)
    normalization = compute_heatmap_normalization([anomaly_map])
    heatmap, overlay = overlay_heatmap(
        display_image,
        anomaly_map,
        normalization=normalization,
        alpha=alpha,
        cmap=cmap,
    )
    return _image_to_base64(heatmap), _image_to_base64(overlay)


def _resize_for_display(image: Image.Image, *, max_side: int) -> Image.Image:
    if max_side <= 0:
        raise ValueError("display_max_side must be positive")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.copy()
    scale = max_side / longest
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(resized, Image.Resampling.LANCZOS)


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
