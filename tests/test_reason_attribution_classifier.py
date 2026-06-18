from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from retinal_ood.reason_attribution.classifier import (
    FAMILY_CLASSES,
    SUBTYPE_CLASSES,
    ReasonAttributionModel,
    compute_classification_outputs,
    fit_reason_classifier,
)
from retinal_ood.reason_attribution.features import (
    FeatureExtractionConfig,
    extract_features_from_manifest,
)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_feature_manifest(root: Path) -> Path:
    rows = []
    for index, (name, value) in enumerate(
        [
            ("first", 32),
            ("second", 32),
            ("third", 220),
        ]
    ):
        image_path = f"images/{name}.png"
        _write_image(root / image_path, value)
        rows.append(
            {
                "image_path": image_path,
                "label": 1,
                "split": "test",
                "source": f"metadata_source_{index}",
                "ood_type": FAMILY_CLASSES[index % len(FAMILY_CLASSES)],
                "ood_subtype": SUBTYPE_CLASSES[index],
                "notes": f"metadata note {index}",
            }
        )
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _toy_reason_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [4.0, 4.0],
            [4.2, 4.1],
            [8.0, 0.0],
            [8.2, 0.1],
        ],
        dtype=np.float32,
    )
    family = np.asarray(
        [
            "modality_shift",
            "modality_shift",
            "sensory_artifact",
            "sensory_artifact",
            "semantic_outlier",
            "semantic_outlier",
        ]
    )
    subtype = np.asarray(
        [
            "colour_fundus",
            "oct_screenshot",
            "text_watermark",
            "rectangle_annotation",
            "cifar10_natural",
            "cifar10_natural",
        ]
    )
    return features, family, subtype


def test_feature_extraction_uses_image_pixels_not_metadata(tmp_path: Path):
    manifest = _write_feature_manifest(tmp_path)
    table = extract_features_from_manifest(
        manifest,
        root_dir=tmp_path,
        config=FeatureExtractionConfig(feature_mode="image_statistics", image_size=16),
        batch_size=2,
    )

    assert table.features.shape[0] == 3
    assert table.metadata["source"].tolist() == [
        "metadata_source_0",
        "metadata_source_1",
        "metadata_source_2",
    ]
    np.testing.assert_allclose(table.features[0], table.features[1], atol=1e-6)
    assert not np.allclose(table.features[0], table.features[2])


def test_classifier_trains_family_and_optional_subtype_with_unknown_threshold():
    features, family, subtype = _toy_reason_data()

    model = fit_reason_classifier(
        features,
        family,
        subtype_labels=subtype,
        train_subtype_classifier=True,
        seed=7,
    )

    predictions = model.predict(features, unknown_threshold=0.5)
    assert set(predictions.family).issubset(set(FAMILY_CLASSES))
    assert predictions.family_confidence.shape == (features.shape[0],)
    assert predictions.subtype is not None
    assert set(predictions.subtype).issubset(set(SUBTYPE_CLASSES))

    unknown = model.predict(np.asarray([[3.9, 0.1]], dtype=np.float32), unknown_threshold=0.99)
    assert unknown.family.tolist() == ["unknown_ood"]


def test_subtype_classifier_is_optional_and_serializes_compact_npz(tmp_path: Path):
    features, family, _ = _toy_reason_data()
    model = fit_reason_classifier(features, family, train_subtype_classifier=False, seed=7)

    assert model.subtype_classifier is None

    path = tmp_path / "reason_model.npz"
    model.save(path)
    loaded = ReasonAttributionModel.load(path)
    predictions = loaded.predict(features, unknown_threshold=0.5)

    assert predictions.subtype is None
    assert predictions.family.shape == (features.shape[0],)


def test_metrics_files_and_confusion_matrix_shape_are_generated(tmp_path: Path):
    true_family = np.asarray(["modality_shift", "sensory_artifact", "semantic_outlier"])
    pred_family = np.asarray(["modality_shift", "sensory_artifact", "modality_shift"])
    true_subtype = np.asarray(["colour_fundus", "text_watermark", "cifar10_natural"])
    pred_subtype = np.asarray(["colour_fundus", "rectangle_annotation", "cifar10_natural"])
    metadata = pd.DataFrame(
        {
            "image_path": ["a.png", "b.png", "c.png"],
            "ood_type": true_family,
            "ood_subtype": true_subtype,
        }
    )

    outputs = compute_classification_outputs(
        true_family=true_family,
        pred_family=pred_family,
        family_confidence=np.asarray([0.9, 0.8, 0.7]),
        metadata=metadata,
        out_dir=tmp_path,
        true_subtype=true_subtype,
        pred_subtype=pred_subtype,
        subtype_confidence=np.asarray([0.9, 0.5, 0.7]),
        unknown_threshold=0.5,
    )

    family_metrics = pd.read_csv(outputs["family_metrics_csv"])
    family_confusion = pd.read_csv(outputs["family_confusion_csv"], index_col=0)
    subtype_metrics = pd.read_csv(outputs["subtype_metrics_csv"])
    subtype_confusion = pd.read_csv(outputs["subtype_confusion_csv"], index_col=0)

    assert {"accuracy", "macro_f1"}.issubset(family_metrics["metric"])
    assert family_confusion.shape == (3, 3)
    assert {"accuracy", "macro_f1"}.issubset(subtype_metrics["metric"])
    assert subtype_confusion.shape == (len(SUBTYPE_CLASSES), len(SUBTYPE_CLASSES))
    assert outputs["predictions_csv"].exists()
