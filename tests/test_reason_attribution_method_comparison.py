from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from retinal_ood.reason_attribution.comparison import (
    ComparisonConfig,
    compute_unknown_threshold_curve,
    run_reason_attribution_method_comparison,
    select_best_method,
    snapshot_file_hashes,
)
from retinal_ood.reason_attribution.methods import (
    HierarchicalReasonClassifier,
    METADATA_EXCLUDED_COLUMNS,
    extract_global_pooled_features_from_manifest,
)


def _write_pattern_image(path: Path, *, family: str, subtype_index: int, variant: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 24
    yy, xx = np.mgrid[0:size, 0:size]
    if family == "modality_shift":
        channel = np.full((size, size), 35 + 8 * variant + subtype_index * 18, dtype=np.uint8)
    elif family == "sensory_artifact":
        channel = np.where((xx + subtype_index * 3) % 5 < 2, 210, 40).astype(np.uint8)
        channel = np.clip(channel + variant * 4, 0, 255).astype(np.uint8)
    else:
        channel = np.where((xx - yy + variant) % 7 < 3, 150, 235).astype(np.uint8)
        channel[:4, :] = 10 + subtype_index
    rgb = np.dstack([channel, np.roll(channel, 1, axis=0), np.roll(channel, 1, axis=1)])
    Image.fromarray(rgb).save(path)


def _write_reason_manifests(root: Path, *, include_id: bool = False) -> tuple[Path, Path, Path]:
    subtype_by_family = {
        "modality_shift": ["colour_fundus", "oct_screenshot"],
        "sensory_artifact": ["text_watermark", "rectangle_annotation"],
        "semantic_outlier": ["cifar10_natural", "cifar10_natural"],
    }
    rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for split, variants in {"train": range(4), "val": range(4, 6), "test": range(6, 8)}.items():
        for family, subtypes in subtype_by_family.items():
            for subtype_index, subtype in enumerate(subtypes):
                for variant in variants:
                    image_path = f"images/{split}_{family}_{subtype_index}_{variant}.png"
                    _write_pattern_image(
                        root / image_path,
                        family=family,
                        subtype_index=subtype_index,
                        variant=variant,
                    )
                    rows_by_split[split].append(
                        {
                            "image_path": image_path,
                            "label": 1,
                            "split": split,
                            "filename": f"{split}_{family}_{subtype_index}_{variant}.png",
                            "source": f"source_{split}_{variant}",
                            "source_dataset": f"dataset_{family}",
                            "source_url": f"https://example.invalid/{split}/{variant}",
                            "license_status": "public",
                            "ood_type": family,
                            "ood_subtype": subtype,
                            "synthetic_transform": f"transform_{subtype_index}",
                            "severity": variant,
                            "notes": f"note with metadata {variant}",
                        }
                    )
    if include_id:
        rows_by_split["train"].append(
            {
                "image_path": rows_by_split["train"][0]["image_path"],
                "label": 0,
                "split": "train",
                "filename": "id.png",
                "source": "forbidden_id_source",
                "source_dataset": "id_dataset",
                "source_url": "https://example.invalid/id",
                "license_status": "public",
                "ood_type": "id",
                "ood_subtype": "",
                "synthetic_transform": "",
                "severity": "",
                "notes": "this ID row must be rejected before Stage 2 training",
            }
        )
    paths = []
    for split in ("train", "val", "test"):
        path = root / f"reason_{split}.csv"
        pd.DataFrame(rows_by_split[split]).to_csv(path, index=False)
        paths.append(path)
    return tuple(paths)  # type: ignore[return-value]


def test_global_pooled_features_are_image_only_and_ignore_metadata(tmp_path: Path):
    train_manifest, _, _ = _write_reason_manifests(tmp_path)
    metadata_changed = pd.read_csv(train_manifest)
    metadata_changed["filename"] = "changed_filename.png"
    metadata_changed["source"] = "changed_source"
    metadata_changed["source_dataset"] = "changed_dataset"
    metadata_changed["source_url"] = "https://changed.invalid/source"
    metadata_changed["license_status"] = "changed_license"
    metadata_changed["label"] = 99
    metadata_changed["notes"] = "changed note text"
    metadata_changed["ood_type"] = metadata_changed["ood_type"].sample(frac=1.0, random_state=3).to_numpy()
    metadata_changed["ood_subtype"] = metadata_changed["ood_subtype"].sample(frac=1.0, random_state=4).to_numpy()
    metadata_changed["synthetic_transform"] = "changed_transform"
    metadata_changed["severity"] = "changed_severity"
    changed_manifest = tmp_path / "reason_train_metadata_changed.csv"
    metadata_changed.to_csv(changed_manifest, index=False)

    original = extract_global_pooled_features_from_manifest(train_manifest, root_dir=tmp_path, image_size=12)
    changed = extract_global_pooled_features_from_manifest(changed_manifest, root_dir=tmp_path, image_size=12)

    np.testing.assert_allclose(original.features, changed.features, atol=1e-6)
    assert original.metadata["source"].iloc[0] != changed.metadata["source"].iloc[0]
    assert original.feature_names[0].startswith("pooled_pixel_")


def test_declared_metadata_exclusion_list_covers_leakage_gate_fields():
    required_excluded = {
        "image_path",
        "filename",
        "source",
        "source_dataset",
        "source_url",
        "license_status",
        "notes",
        "label",
        "ood_type",
        "ood_subtype",
        "synthetic_transform",
        "severity",
    }

    assert required_excluded.issubset(set(METADATA_EXCLUDED_COLUMNS))


def test_comparison_generates_required_tables_and_uses_same_splits(tmp_path: Path):
    train_manifest, val_manifest, test_manifest = _write_reason_manifests(tmp_path)
    stage1_guard = tmp_path / "stage1_gatekeeper_config.yaml"
    stage1_guard.write_text("stage1: id_only_unsupervised\n", encoding="utf-8")
    before_hashes = snapshot_file_hashes([stage1_guard])

    result = run_reason_attribution_method_comparison(
        ComparisonConfig(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            root_dir=tmp_path,
            out_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
            seed=11,
            unknown_threshold=0.5,
            include_subtype=True,
            include_hierarchical=True,
            method_names=("image_statistics_logreg", "nearest_centroid", "hierarchical_classifier"),
            stage1_guard_paths=(stage1_guard,),
            global_image_size=12,
            statistics_image_size=16,
        )
    )

    assert result.split_sizes == {"train": 24, "val": 12, "test": 12}
    assert result.stage1_file_hashes_before == before_hashes
    assert snapshot_file_hashes([stage1_guard]) == before_hashes
    for name in (
        "method_overview.csv",
        "family_metrics_by_method.csv",
        "subtype_metrics_by_method.csv",
        "per_family_f1_by_method.csv",
        "per_subtype_f1_by_method.csv",
        "unknown_threshold_by_method.csv",
        "best_method_summary.csv",
        "skipped_methods.md",
    ):
        assert (tmp_path / "results" / name).exists()
    assert (tmp_path / "figures" / "figure_reason_method_family_macro_f1.png").exists()
    assert result.best_family_method in {"image_statistics_logreg", "nearest_centroid", "hierarchical_classifier"}


def test_comparison_rejects_id_rows_before_stage2_training(tmp_path: Path):
    train_manifest, val_manifest, test_manifest = _write_reason_manifests(tmp_path, include_id=True)

    try:
        run_reason_attribution_method_comparison(
            ComparisonConfig(
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                test_manifest=test_manifest,
                root_dir=tmp_path,
                out_dir=tmp_path / "results",
                figures_dir=tmp_path / "figures",
                method_names=("image_statistics_logreg",),
            )
        )
    except ValueError as exc:
        assert "Stage 2 manifests must be OOD-only" in str(exc)
    else:
        raise AssertionError("ID rows must not enter Stage 2 comparison training")


def test_best_method_selection_and_unknown_threshold_are_deterministic():
    metrics = pd.DataFrame(
        [
            {"method": "complex_method", "split": "val", "family_macro_f1": 0.91, "complexity": 4},
            {"method": "simple_method", "split": "val", "family_macro_f1": 0.91, "complexity": 1},
            {"method": "lower_method", "split": "val", "family_macro_f1": 0.90, "complexity": 0},
        ]
    )

    assert select_best_method(metrics, split="val") == "simple_method"

    curve = compute_unknown_threshold_curve(
        true_family=np.asarray(["a", "a", "b", "b"]),
        predicted_family=np.asarray(["a", "b", "b", "a"]),
        confidence=np.asarray([0.95, 0.4, 0.7, 0.2]),
        thresholds=np.asarray([0.0, 0.5, 0.9]),
    )

    assert curve["coverage"].tolist() == [1.0, 0.5, 0.25]
    assert curve["known_accuracy"].tolist() == [0.5, 1.0, 1.0]


def test_subtype_can_be_disabled_and_hierarchy_uses_predicted_family(tmp_path: Path):
    train_manifest, val_manifest, test_manifest = _write_reason_manifests(tmp_path)
    result = run_reason_attribution_method_comparison(
        ComparisonConfig(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            root_dir=tmp_path,
            out_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
            seed=5,
            include_subtype=False,
            include_hierarchical=False,
            method_names=("image_statistics_logreg",),
            global_image_size=12,
            statistics_image_size=16,
        )
    )

    subtype_metrics = pd.read_csv(tmp_path / "results" / "subtype_metrics_by_method.csv")
    assert set(subtype_metrics["split"]) == {"val", "test"}
    assert set(subtype_metrics["status"]) == {"not_requested"}
    assert result.best_subtype_method == "not_requested"

    hierarchy = HierarchicalReasonClassifier(
        family_model=_FixedEstimator("modality_shift", ["modality_shift", "sensory_artifact"]),
        subtype_models={
            "modality_shift": _FixedEstimator("colour_fundus", ["colour_fundus"]),
            "sensory_artifact": _FixedEstimator("text_watermark", ["text_watermark"]),
        },
        fallback_subtype_model=_FixedEstimator("cifar10_natural", ["cifar10_natural"]),
    )
    predictions = hierarchy.predict(np.ones((3, 2), dtype=np.float32), unknown_threshold=0.5)

    assert predictions.family_argmax.tolist() == ["modality_shift", "modality_shift", "modality_shift"]
    assert predictions.subtype is not None
    assert predictions.subtype.tolist() == ["colour_fundus", "colour_fundus", "colour_fundus"]


class _FixedEstimator:
    def __init__(self, label: str, classes: list[str]) -> None:
        self.label = label
        self.classes_ = np.asarray(classes)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.label, dtype=object)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((features.shape[0], len(self.classes_)), dtype=float)
        probabilities[:, list(self.classes_).index(self.label)] = 1.0
        return probabilities
