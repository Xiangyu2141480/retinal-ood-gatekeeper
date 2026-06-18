#!/usr/bin/env python
"""Run a compact two-stage gatekeeper demo with optional reason attribution."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.reason_attribution.classifier import ReasonAttributionModel


def run_demo(
    *,
    root_dir: str | Path,
    model_dir: str | Path,
    image_path: str | Path | None = None,
    manifest: str | Path | None = None,
    stage1_decision: str = "REJECT",
    unknown_threshold: float = 0.5,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run Stage 2 only for rows whose Stage 1 decision is REJECT."""
    if image_path is None and manifest is None:
        raise ValueError("Provide either --image-path or --manifest")
    if image_path is not None and manifest is not None:
        raise ValueError("Provide only one of --image-path or --manifest")
    model_path, feature_config = _load_model_config(model_dir)
    model = ReasonAttributionModel.load(model_path)
    if manifest is not None:
        frame = pd.read_csv(manifest)
        if limit is not None:
            frame = frame.head(limit)
        return [
            _run_one_row(
                row=row,
                root_dir=root_dir,
                model=model,
                feature_config=feature_config,
                stage1_decision=_row_stage1_decision(row, stage1_decision),
                unknown_threshold=unknown_threshold,
            )
            for _, row in frame.iterrows()
        ]
    row = pd.Series(
        {
            "image_path": str(image_path),
            "label": 1,
            "split": "demo",
            "source": "demo_input",
            "ood_type": "",
            "ood_subtype": "",
        }
    )
    return [
        _run_one_row(
            row=row,
            root_dir=root_dir,
            model=model,
            feature_config=feature_config,
            stage1_decision=stage1_decision,
            unknown_threshold=unknown_threshold,
        )
    ]


def _run_one_row(
    *,
    row: pd.Series,
    root_dir: str | Path,
    model: ReasonAttributionModel,
    feature_config: Any,
    stage1_decision: str,
    unknown_threshold: float,
) -> dict[str, Any]:
    decision = stage1_decision.upper()
    result: dict[str, Any] = {
        "image_path": str(row["image_path"]),
        "stage1_decision": decision,
        "anomaly_score": _optional_float(row.get("score")),
        "note": (
            "Stage 2 is post-hoc explanation only. Stage 1 remains an ID-only "
            "unsupervised binary gatekeeper."
        ),
    }
    if decision == "ACCEPT":
        result["stage1_interpretation"] = "valid FAF-like input"
        return result
    if decision != "REJECT":
        raise ValueError("stage1_decision must resolve to ACCEPT or REJECT")
    from retinal_ood.reason_attribution.features import extract_features_from_manifest

    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "single_image_manifest.csv"
        pd.DataFrame([_manifest_row_for_feature_extraction(row)]).to_csv(manifest, index=False)
        table = extract_features_from_manifest(
            manifest,
            root_dir=root_dir,
            config=feature_config,
            batch_size=1,
        )
    prediction = model.predict(table.features, unknown_threshold=unknown_threshold)
    result.update(
        {
            "stage2_reason_family": str(prediction.family[0]),
            "reason_confidence": float(prediction.family_confidence[0]),
            "unknown_threshold": float(unknown_threshold),
            "stage2_subtype": str(prediction.subtype[0]) if prediction.subtype is not None else None,
            "subtype_confidence": float(prediction.subtype_confidence[0])
            if prediction.subtype_confidence is not None
            else None,
            "heatmap_path": _optional_text(row.get("heatmap_path")),
        }
    )
    return result


def _manifest_row_for_feature_extraction(row: pd.Series) -> dict[str, Any]:
    return {
        "image_path": str(row["image_path"]),
        "label": int(row.get("label", 1) if str(row.get("label", "")).strip() else 1),
        "split": str(row.get("split", "demo") or "demo"),
        "source": str(row.get("source", "demo_input") or "demo_input"),
        "ood_type": str(row.get("ood_type", "demo") or "demo"),
        "ood_subtype": str(row.get("ood_subtype", "demo") or "demo"),
    }


def _row_stage1_decision(row: pd.Series, requested: str) -> str:
    requested = requested.upper()
    if requested in {"ACCEPT", "REJECT"}:
        return requested
    if requested != "AUTO":
        raise ValueError("--stage1-decision must be ACCEPT, REJECT, or auto")
    if "prediction" in row and str(row["prediction"]).strip() != "":
        return "REJECT" if int(row["prediction"]) == 1 else "ACCEPT"
    if "label" in row and str(row["label"]).strip() != "":
        return "REJECT" if int(row["label"]) == 1 else "ACCEPT"
    return "REJECT"


def _load_model_config(model_dir: str | Path) -> tuple[Path, Any]:
    from retinal_ood.reason_attribution.features import FeatureExtractionConfig

    model_root = Path(model_dir)
    metadata_path = model_root / "reason_model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = metadata.get("feature_config", {})
    return (
        model_root / metadata.get("model_path", "reason_model.npz"),
        FeatureExtractionConfig(
            feature_mode=cfg.get("feature_mode", "image_statistics"),
            image_size=int(cfg.get("image_size", 224)),
            backbone=cfg.get("backbone", "resnet50"),
            layers=tuple(cfg.get("layers", ["layer4"])),
            pretrained=bool(cfg.get("pretrained", True)),
            feature_backend=cfg.get("feature_backend", "torchvision"),
            grayscale_to_rgb=bool(cfg.get("grayscale_to_rgb", True)),
            normalize=cfg.get("normalize"),
            device=cfg.get("device", "cpu"),
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "" or str(value).lower() == "nan":
        return None
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None or str(value).strip() == "" or str(value).lower() == "nan":
        return None
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the two-stage rejected-input explanation demo.")
    parser.add_argument("--image-path", help="Single image path, relative to --root-dir or absolute")
    parser.add_argument("--manifest", help="Manifest containing image_path rows")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--stage1-decision", default="REJECT", choices=["ACCEPT", "REJECT", "auto"])
    parser.add_argument("--unknown-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="Maximum manifest rows to process")
    parser.add_argument("--out-json", help="Optional output JSON file")
    args = parser.parse_args()
    result = run_demo(
        root_dir=args.root_dir,
        model_dir=args.model_dir,
        image_path=args.image_path,
        manifest=args.manifest,
        stage1_decision=args.stage1_decision,
        unknown_threshold=args.unknown_threshold,
        limit=args.limit,
    )
    text = json.dumps(result[0] if args.image_path else result, indent=2)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
