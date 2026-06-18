#!/usr/bin/env python
"""Train the optional Stage 2 rejected-input reason attribution module."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.reason_attribution.classifier import compute_classification_outputs, fit_reason_classifier


def train_reason_attribution(
    *,
    train_manifest: str | Path,
    val_manifest: str | Path,
    root_dir: str | Path,
    out_dir: str | Path,
    reports_dir: str | Path = "reports/dissertation_results/reason_attribution",
    seed: int = 42,
    train_subtype_classifier: bool = False,
    feature_config: Any | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, Path]:
    """Train Stage 2 reason classifiers from OOD-only reason manifests."""
    from retinal_ood.reason_attribution.features import FeatureExtractionConfig, extract_features_from_manifest

    config = feature_config or FeatureExtractionConfig()
    train_table = extract_features_from_manifest(
        train_manifest,
        root_dir=root_dir,
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_table = extract_features_from_manifest(
        val_manifest,
        root_dir=root_dir,
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = fit_reason_classifier(
        train_table.features,
        train_table.metadata["ood_type"].astype(str).to_numpy(),
        subtype_labels=train_table.metadata["ood_subtype"].astype(str).to_numpy(),
        train_subtype_classifier=train_subtype_classifier,
        seed=seed,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model.save(output_dir / "reason_model.npz")
    metadata_path = output_dir / "reason_model_metadata.json"
    metadata = {
        "model_path": model_path.name,
        "seed": seed,
        "train_rows": int(len(train_table.metadata)),
        "val_rows": int(len(val_table.metadata)),
        "feature_dim": int(train_table.features.shape[1]),
        "feature_config": _json_ready_feature_config(config),
        "train_subtype_classifier": bool(train_subtype_classifier),
        "stage_boundary": (
            "Stage 2 reason attribution is trained on OOD taxonomy labels only after "
            "Stage 1 rejection; Stage 1 remains ID-only and unchanged."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    val_predictions = model.predict(val_table.features, unknown_threshold=0.5)
    validation_dir = output_dir / "validation"
    compute_classification_outputs(
        true_family=val_table.metadata["ood_type"].astype(str).to_numpy(),
        pred_family=val_predictions.family,
        family_argmax=val_predictions.family_argmax,
        family_confidence=val_predictions.family_confidence,
        metadata=val_table.metadata,
        out_dir=validation_dir,
        true_subtype=val_table.metadata["ood_subtype"].astype(str).to_numpy()
        if train_subtype_classifier
        else None,
        pred_subtype=val_predictions.subtype,
        subtype_confidence=val_predictions.subtype_confidence,
        unknown_threshold=0.5,
    )
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    training_summary = reports_path / "training_summary.md"
    _write_training_summary(
        training_summary,
        metadata=metadata,
        val_metrics_path=validation_dir / "reason_family_metrics.csv",
        val_predictions_unknown=int((val_predictions.family == "unknown_ood").sum()),
    )
    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "training_summary": training_summary,
    }


def _json_ready_feature_config(config: Any) -> dict[str, Any]:
    raw = asdict(config)
    raw["layers"] = list(config.layers)
    return raw


def _write_training_summary(
    path: Path,
    *,
    metadata: dict[str, Any],
    val_metrics_path: Path,
    val_predictions_unknown: int,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Reason Attribution Training Summary",
                "",
                "This trains only the optional Stage 2 explanation module for already rejected OOD inputs.",
                "Stage 1 remains the ID-only unsupervised binary OOD gatekeeper and is not retrained here.",
                "",
                f"- train rows: {metadata['train_rows']}",
                f"- validation rows: {metadata['val_rows']}",
                f"- feature mode: {metadata['feature_config']['feature_mode']}",
                f"- feature dimension: {metadata['feature_dim']}",
                f"- subtype classifier trained: {metadata['train_subtype_classifier']}",
                f"- validation unknown predictions at gamma=0.5: {val_predictions_unknown}",
                f"- validation metrics: `{val_metrics_path.as_posix()}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _feature_config_from_args(args: argparse.Namespace) -> Any:
    from retinal_ood.reason_attribution.features import FeatureExtractionConfig

    return FeatureExtractionConfig(
        feature_mode=args.feature_mode,
        image_size=args.image_size,
        backbone=args.backbone,
        layers=tuple(args.layers),
        pretrained=not args.no_pretrained,
        feature_backend=args.feature_backend,
        device=args.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optional Stage 2 reason attribution classifiers.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reports-dir", default="reports/dissertation_results/reason_attribution")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-subtype-classifier", action="store_true")
    parser.add_argument("--feature-mode", choices=["image_statistics", "cnn"], default="image_statistics")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--layers", nargs="+", default=["layer4"])
    parser.add_argument("--feature-backend", default="torchvision", choices=["torchvision", "timm"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained CNN weights.")
    args = parser.parse_args()

    outputs = train_reason_attribution(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
        seed=args.seed,
        train_subtype_classifier=args.train_subtype_classifier,
        feature_config=_feature_config_from_args(args),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
