"""Method definitions for Stage 2 rejected-input reason attribution.

All feature extractors in this module are image-derived. Manifest metadata is kept
only for labels, grouping, and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from retinal_ood.reason_attribution.classifier import (
    FAMILY_CLASSES,
    UNKNOWN_LABEL,
    ReasonPredictions,
)
from retinal_ood.reason_attribution.features import FeatureTable

FeatureSetName = Literal["statistics", "global", "fusion"]

METADATA_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "image_path",
    "file_name",
    "filename",
    "source",
    "source_dataset",
    "source_url",
    "license_status",
    "source_split",
    "source_image_hash",
    "parent_image_hash",
    "ood_type",
    "ood_subtype",
    "synthetic_transform",
    "severity",
    "notes",
    "label",
    "split",
)


@dataclass(frozen=True)
class ReasonMethodSpec:
    """Configuration for a lightweight Stage 2 reason-attribution method."""

    name: str
    feature_set: FeatureSetName
    estimator_kind: str
    description: str
    complexity: int
    is_hierarchical: bool = False


@dataclass(frozen=True)
class ReasonMethodModel:
    """Fitted method with a uniform prediction interface."""

    spec: ReasonMethodSpec
    family_model: Any
    subtype_model: Any | None
    train_seconds: float

    def predict(self, features: np.ndarray, *, unknown_threshold: float) -> ReasonPredictions:
        if isinstance(self.family_model, HierarchicalReasonClassifier):
            return self.family_model.predict(features, unknown_threshold=unknown_threshold)

        family_argmax, family_confidence = predict_labels_and_confidence(
            self.family_model,
            features,
        )
        family = family_argmax.copy()
        family[family_confidence < unknown_threshold] = UNKNOWN_LABEL
        subtype = None
        subtype_confidence = None
        if self.subtype_model is not None:
            subtype, subtype_confidence = predict_labels_and_confidence(self.subtype_model, features)

        return ReasonPredictions(
            family=family.astype(str),
            family_confidence=family_confidence.astype(float),
            family_argmax=family_argmax.astype(str),
            subtype=subtype.astype(str) if subtype is not None else None,
            subtype_confidence=(
                subtype_confidence.astype(float) if subtype_confidence is not None else None
            ),
        )


@dataclass(frozen=True)
class HierarchicalReasonClassifier:
    """Family-first reason attribution without oracle family leakage at test time."""

    family_model: Any
    subtype_models: dict[str, Any]
    fallback_subtype_model: Any | None = None

    def predict(self, features: np.ndarray, *, unknown_threshold: float) -> ReasonPredictions:
        matrix = _validate_matrix(features)
        family_argmax, family_confidence = predict_labels_and_confidence(
            self.family_model,
            matrix,
        )
        family = family_argmax.copy()
        family[family_confidence < unknown_threshold] = UNKNOWN_LABEL

        subtype = np.empty(matrix.shape[0], dtype=object)
        subtype_confidence = np.zeros(matrix.shape[0], dtype=float)
        for index, predicted_family in enumerate(family_argmax.astype(str)):
            subtype_model = self.subtype_models.get(predicted_family, self.fallback_subtype_model)
            if subtype_model is None:
                subtype[index] = UNKNOWN_LABEL
                subtype_confidence[index] = 0.0
                continue
            label, confidence = predict_labels_and_confidence(
                subtype_model,
                matrix[index : index + 1],
            )
            subtype[index] = str(label[0])
            subtype_confidence[index] = float(confidence[0])

        return ReasonPredictions(
            family=family.astype(str),
            family_confidence=family_confidence.astype(float),
            family_argmax=family_argmax.astype(str),
            subtype=subtype.astype(str),
            subtype_confidence=subtype_confidence.astype(float),
        )


class ConstantLabelEstimator:
    """Estimator used when a family contains only one subtype."""

    def __init__(self, label: str) -> None:
        self.label = str(label)
        self.classes_ = np.asarray([self.label])

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ConstantLabelEstimator":
        _ = _validate_matrix(features)
        labels = np.asarray(labels).astype(str)
        if set(labels) != {self.label}:
            raise ValueError("ConstantLabelEstimator received labels outside its single class")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = _validate_matrix(features)
        return np.full(matrix.shape[0], self.label, dtype=object)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = _validate_matrix(features)
        return np.ones((matrix.shape[0], 1), dtype=float)


def available_reason_method_specs(
    *,
    include_hierarchical: bool = False,
    include_rbf_svm: bool = False,
) -> tuple[ReasonMethodSpec, ...]:
    """Return the ordered set of supported comparison methods."""
    specs: list[ReasonMethodSpec] = [
        ReasonMethodSpec(
            name="image_statistics_logreg",
            feature_set="statistics",
            estimator_kind="logistic_regression",
            description="Balanced logistic regression on low-level image statistics h(x).",
            complexity=1,
        ),
        ReasonMethodSpec(
            name="global_feature_knn",
            feature_set="global",
            estimator_kind="knn",
            description="K-nearest neighbours on deterministic pooled image features z(x).",
            complexity=3,
        ),
        ReasonMethodSpec(
            name="nearest_centroid",
            feature_set="global",
            estimator_kind="nearest_centroid",
            description="Class prototype baseline in pooled image-feature space.",
            complexity=2,
        ),
        ReasonMethodSpec(
            name="logistic_regression",
            feature_set="global",
            estimator_kind="logistic_regression",
            description="Balanced multinomial logistic regression on pooled image features z(x).",
            complexity=3,
        ),
        ReasonMethodSpec(
            name="linear_svm",
            feature_set="global",
            estimator_kind="linear_svm",
            description="Balanced linear SVM margin baseline on pooled image features z(x).",
            complexity=4,
        ),
        ReasonMethodSpec(
            name="random_forest_or_gradient_boosting",
            feature_set="global",
            estimator_kind="random_forest",
            description="Balanced random forest non-linear classical ML baseline.",
            complexity=5,
        ),
        ReasonMethodSpec(
            name="feature_statistics_fusion",
            feature_set="fusion",
            estimator_kind="logistic_regression",
            description="Balanced logistic regression on concatenated z(x) and h(x).",
            complexity=4,
        ),
    ]
    if include_hierarchical:
        specs.append(
            ReasonMethodSpec(
                name="hierarchical_classifier",
                feature_set="fusion",
                estimator_kind="hierarchical_logistic",
                description=(
                    "Family-first classifier; subtype prediction is routed by predicted family."
                ),
                complexity=6,
                is_hierarchical=True,
            )
        )
    if include_rbf_svm:
        specs.append(
            ReasonMethodSpec(
                name="rbf_svm_optional",
                feature_set="global",
                estimator_kind="rbf_svm",
                description="Optional RBF SVM baseline; disabled by default for runtime.",
                complexity=8,
            )
        )
    return tuple(specs)


def resolve_method_specs(
    *,
    method_names: tuple[str, ...] | None,
    include_hierarchical: bool,
    include_rbf_svm: bool,
) -> tuple[ReasonMethodSpec, ...]:
    """Resolve requested method names to method specs."""
    include_hierarchy = include_hierarchical or (
        method_names is not None and "hierarchical_classifier" in method_names
    )
    specs = available_reason_method_specs(
        include_hierarchical=include_hierarchy,
        include_rbf_svm=include_rbf_svm,
    )
    if method_names is None:
        return specs
    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(method_names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown reason attribution methods requested: {unknown}")
    return tuple(by_name[name] for name in method_names)


def fit_reason_method(
    spec: ReasonMethodSpec,
    train_features: np.ndarray,
    family_labels: np.ndarray,
    *,
    subtype_labels: np.ndarray | None,
    include_subtype: bool,
    seed: int,
) -> ReasonMethodModel:
    """Fit a configured Stage 2 reason-attribution method."""
    start = perf_counter()
    matrix = _validate_matrix(train_features)
    family = np.asarray(family_labels).astype(str)
    if spec.is_hierarchical:
        model = _fit_hierarchical_classifier(
            matrix,
            family,
            subtype_labels=subtype_labels,
            seed=seed,
            include_subtype=include_subtype,
        )
        return ReasonMethodModel(
            spec=spec,
            family_model=model,
            subtype_model=None,
            train_seconds=perf_counter() - start,
        )

    family_model = _fit_estimator(
        spec.estimator_kind,
        matrix,
        family,
        seed=seed,
    )
    subtype_model = None
    if include_subtype and subtype_labels is not None:
        subtype_model = _fit_estimator(
            spec.estimator_kind,
            matrix,
            np.asarray(subtype_labels).astype(str),
            seed=seed,
        )
    return ReasonMethodModel(
        spec=spec,
        family_model=family_model,
        subtype_model=subtype_model,
        train_seconds=perf_counter() - start,
    )


def extract_global_pooled_features_from_manifest(
    manifest_path: str | Path,
    *,
    root_dir: str | Path,
    image_size: int = 24,
) -> FeatureTable:
    """Extract deterministic pooled image features used as z(x).

    The manifest is used only to locate image files and return reporting metadata. No
    metadata values are encoded into the feature matrix.
    """
    metadata = pd.read_csv(manifest_path)
    if metadata.empty:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    if "image_path" not in metadata.columns:
        raise ValueError(f"Manifest missing image_path column: {manifest_path}")
    features = np.vstack(
        [
            _pooled_pixels_from_image(
                _resolve_image_path(image_path, root_dir=Path(root_dir)),
                image_size=image_size,
            )
            for image_path in metadata["image_path"].tolist()
        ]
    ).astype(np.float32)
    feature_names = tuple(f"pooled_pixel_{index:04d}" for index in range(features.shape[1]))
    return FeatureTable(
        features=features,
        metadata=metadata.reset_index(drop=True).copy(),
        feature_names=feature_names,
    )


def predict_labels_and_confidence(estimator: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and a normalized confidence score for sklearn-like estimators."""
    probabilities = predict_probabilities(estimator, features)
    classes = _estimator_classes(estimator)
    indices = probabilities.argmax(axis=1)
    labels = classes[indices]
    confidence = probabilities[np.arange(len(indices)), indices]
    return labels.astype(str), confidence.astype(float)


def predict_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Return probability-like scores for calibrated and non-calibrated estimators."""
    matrix = _validate_matrix(features)
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(matrix)
        return _normalize_probabilities(np.asarray(probabilities, dtype=float))

    if isinstance(estimator, Pipeline):
        classifier = estimator.steps[-1][1]
        if isinstance(classifier, NearestCentroid):
            transformed = estimator[:-1].transform(matrix)
            distances = _pairwise_euclidean_distances(transformed, classifier.centroids_)
            return _softmax(-distances)

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(matrix)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return _softmax(scores)

    predictions = estimator.predict(matrix).astype(str)
    classes = _estimator_classes(estimator)
    probabilities = np.zeros((matrix.shape[0], len(classes)), dtype=float)
    class_to_index = {label: index for index, label in enumerate(classes)}
    for row_index, label in enumerate(predictions):
        probabilities[row_index, class_to_index[str(label)]] = 1.0
    return probabilities


def _fit_hierarchical_classifier(
    features: np.ndarray,
    family_labels: np.ndarray,
    *,
    subtype_labels: np.ndarray | None,
    seed: int,
    include_subtype: bool,
) -> HierarchicalReasonClassifier:
    family_model = _fit_estimator(
        "logistic_regression",
        features,
        family_labels,
        seed=seed,
    )
    if not include_subtype or subtype_labels is None:
        return HierarchicalReasonClassifier(
            family_model=family_model,
            subtype_models={},
            fallback_subtype_model=None,
        )

    subtype = np.asarray(subtype_labels).astype(str)
    subtype_models: dict[str, Any] = {}
    for family in FAMILY_CLASSES:
        mask = family_labels.astype(str) == family
        if not mask.any():
            continue
        subtype_models[family] = _fit_estimator(
            "logistic_regression",
            features[mask],
            subtype[mask],
            seed=seed,
        )
    fallback = _fit_estimator(
        "logistic_regression",
        features,
        subtype,
        seed=seed,
    )
    return HierarchicalReasonClassifier(
        family_model=family_model,
        subtype_models=subtype_models,
        fallback_subtype_model=fallback,
    )


def _fit_estimator(kind: str, features: np.ndarray, labels: np.ndarray, *, seed: int) -> Any:
    matrix = _validate_matrix(features)
    labels = np.asarray(labels).astype(str)
    unique = np.unique(labels)
    if len(unique) == 1:
        return ConstantLabelEstimator(str(unique[0])).fit(matrix, labels)

    if kind == "logistic_regression":
        estimator: Any = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif kind == "knn":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=min(5, len(labels)))),
            ]
        )
    elif kind == "nearest_centroid":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", NearestCentroid()),
            ]
        )
    elif kind == "linear_svm":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        dual="auto",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    elif kind == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced",
            min_samples_leaf=1,
            n_estimators=160,
            n_jobs=1,
            random_state=seed,
        )
    elif kind == "rbf_svm":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        class_weight="balanced",
                        gamma="scale",
                        kernel="rbf",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported estimator kind: {kind}")

    estimator.fit(matrix, labels)
    return estimator


def _pooled_pixels_from_image(path: Path, *, image_size: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")
    image = Image.open(path).convert("L")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((image_size, image_size), resample=resampling)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(-1).astype(np.float32)


def _resolve_image_path(image_path: str | Path, *, root_dir: Path) -> Path:
    path = Path(str(image_path))
    if path.is_absolute():
        return path
    return root_dir / path


def _estimator_classes(estimator: Any) -> np.ndarray:
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_).astype(str)
    if isinstance(estimator, Pipeline):
        return np.asarray(estimator.steps[-1][1].classes_).astype(str)
    raise ValueError("Estimator does not expose classes_")


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Probabilities must be a 2D matrix")
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return matrix / row_sums


def _validate_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"features must be a 2D matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain NaN or infinite values")
    return matrix


def _pairwise_euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    squared = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return np.sqrt(squared).astype(np.float32)


def _softmax(scores: np.ndarray) -> np.ndarray:
    logits = np.asarray(scores, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
