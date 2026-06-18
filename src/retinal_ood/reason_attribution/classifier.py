"""Lightweight classifiers for post-rejection OOD reason attribution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from retinal_ood.evaluation.report_tables import dataframe_to_markdown

FAMILY_CLASSES: tuple[str, ...] = (
    "modality_shift",
    "sensory_artifact",
    "semantic_outlier",
)

SUBTYPE_CLASSES: tuple[str, ...] = (
    "colour_fundus",
    "oct_screenshot",
    "text_watermark",
    "rectangle_annotation",
    "arrow_annotation",
    "composite_layout",
    "blur_artifact",
    "border_crop",
    "gaussian_noise",
    "jpeg_compression",
    "cifar10_natural",
)

UNKNOWN_LABEL = "unknown_ood"


@dataclass(frozen=True)
class ReasonPredictions:
    """Predicted Stage 2 reason labels and confidences."""

    family: np.ndarray
    family_confidence: np.ndarray
    family_argmax: np.ndarray
    subtype: np.ndarray | None = None
    subtype_confidence: np.ndarray | None = None


@dataclass
class _ProbabilisticClassifier:
    method: str
    classes: np.ndarray
    coef: np.ndarray | None = None
    intercept: np.ndarray | None = None
    centroids: np.ndarray | None = None

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = _validate_feature_matrix(features)
        if self.method == "logistic":
            if self.coef is None or self.intercept is None:
                raise ValueError("Logistic classifier is missing parameters")
            return _logistic_proba(matrix, self.coef, self.intercept, len(self.classes))
        if self.method == "nearest_centroid":
            if self.centroids is None:
                raise ValueError("Nearest-centroid classifier is missing centroids")
            distances = _pairwise_euclidean_distances(matrix, self.centroids)
            return _softmax(-distances)
        raise ValueError(f"Unsupported classifier method: {self.method}")

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(features)
        indices = probabilities.argmax(axis=1)
        labels = self.classes[indices]
        confidence = probabilities[np.arange(len(indices)), indices]
        return labels.astype(str), confidence.astype(float)


@dataclass
class ReasonAttributionModel:
    """Family classifier with optional subtype classifier."""

    family_classifier: _ProbabilisticClassifier
    subtype_classifier: _ProbabilisticClassifier | None = None

    def predict(self, features: np.ndarray, *, unknown_threshold: float = 0.5) -> ReasonPredictions:
        if not 0.0 <= unknown_threshold <= 1.0:
            raise ValueError("unknown_threshold must be between 0 and 1")
        family_argmax, family_confidence = self.family_classifier.predict(features)
        family = family_argmax.copy()
        family[family_confidence < unknown_threshold] = UNKNOWN_LABEL
        subtype = None
        subtype_confidence = None
        if self.subtype_classifier is not None:
            subtype, subtype_confidence = self.subtype_classifier.predict(features)
        return ReasonPredictions(
            family=family.astype(str),
            family_confidence=family_confidence.astype(float),
            family_argmax=family_argmax.astype(str),
            subtype=subtype.astype(str) if subtype is not None else None,
            subtype_confidence=subtype_confidence.astype(float) if subtype_confidence is not None else None,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _classifier_payload("family", self.family_classifier)
        has_subtype = self.subtype_classifier is not None
        payload["has_subtype"] = np.asarray(has_subtype)
        if self.subtype_classifier is not None:
            payload.update(_classifier_payload("subtype", self.subtype_classifier))
        else:
            payload.update(_empty_classifier_payload("subtype"))
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReasonAttributionModel":
        data = np.load(path, allow_pickle=False)
        family = _classifier_from_payload(data, "family")
        subtype = _classifier_from_payload(data, "subtype") if bool(data["has_subtype"]) else None
        return cls(family_classifier=family, subtype_classifier=subtype)


def fit_reason_classifier(
    features: np.ndarray,
    family_labels: np.ndarray | list[str] | pd.Series,
    *,
    subtype_labels: np.ndarray | list[str] | pd.Series | None = None,
    train_subtype_classifier: bool = False,
    seed: int = 42,
) -> ReasonAttributionModel:
    """Fit Stage 2 reason family and optional subtype classifiers."""
    matrix = _validate_feature_matrix(features)
    family = _validate_labels(family_labels, allowed=FAMILY_CLASSES, target_name="ood_type")
    family_classifier = _fit_classifier(matrix, family, seed=seed)
    subtype_classifier = None
    if train_subtype_classifier:
        if subtype_labels is None:
            raise ValueError("subtype_labels are required when train_subtype_classifier=True")
        subtype = _validate_labels(subtype_labels, allowed=SUBTYPE_CLASSES, target_name="ood_subtype")
        subtype_classifier = _fit_classifier(matrix, subtype, seed=seed)
    return ReasonAttributionModel(
        family_classifier=family_classifier,
        subtype_classifier=subtype_classifier,
    )


def compute_classification_outputs(
    *,
    true_family: np.ndarray,
    pred_family: np.ndarray,
    family_confidence: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: str | Path,
    unknown_threshold: float,
    true_subtype: np.ndarray | None = None,
    pred_subtype: np.ndarray | None = None,
    subtype_confidence: np.ndarray | None = None,
    family_argmax: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write reason-attribution metrics, confusion matrices, and predictions."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    true_family = np.asarray(true_family).astype(str)
    pred_family = np.asarray(pred_family).astype(str)
    family_argmax = np.asarray(family_argmax).astype(str) if family_argmax is not None else pred_family

    family_metrics = _metrics_table(true_family, pred_family, labels=FAMILY_CLASSES)
    family_confusion = _confusion_table(true_family, family_argmax, labels=FAMILY_CLASSES)
    predictions = metadata.reset_index(drop=True).copy()
    predictions["true_reason_family"] = true_family
    predictions["pred_reason_family"] = pred_family
    predictions["argmax_reason_family"] = family_argmax
    predictions["reason_confidence"] = np.asarray(family_confidence, dtype=float)
    predictions["unknown_threshold"] = float(unknown_threshold)

    outputs = {
        "family_metrics_csv": out_path / "reason_family_metrics.csv",
        "family_metrics_md": out_path / "reason_family_metrics.md",
        "family_confusion_csv": out_path / "reason_family_confusion_matrix.csv",
        "predictions_csv": out_path / "reason_predictions.csv",
    }
    _write_csv_and_markdown(family_metrics, outputs["family_metrics_csv"], outputs["family_metrics_md"], "Reason Family Metrics")
    family_confusion.to_csv(outputs["family_confusion_csv"])

    if true_subtype is not None and pred_subtype is not None:
        true_subtype = np.asarray(true_subtype).astype(str)
        pred_subtype = np.asarray(pred_subtype).astype(str)
        subtype_metrics = _metrics_table(true_subtype, pred_subtype, labels=SUBTYPE_CLASSES)
        subtype_confusion = _confusion_table(true_subtype, pred_subtype, labels=SUBTYPE_CLASSES)
        outputs.update(
            {
                "subtype_metrics_csv": out_path / "reason_subtype_metrics.csv",
                "subtype_metrics_md": out_path / "reason_subtype_metrics.md",
                "subtype_confusion_csv": out_path / "reason_subtype_confusion_matrix.csv",
            }
        )
        _write_csv_and_markdown(
            subtype_metrics,
            outputs["subtype_metrics_csv"],
            outputs["subtype_metrics_md"],
            "Reason Subtype Metrics",
        )
        subtype_confusion.to_csv(outputs["subtype_confusion_csv"])
        predictions["true_reason_subtype"] = true_subtype
        predictions["pred_reason_subtype"] = pred_subtype
        predictions["subtype_confidence"] = (
            np.asarray(subtype_confidence, dtype=float)
            if subtype_confidence is not None
            else np.full(len(pred_subtype), np.nan)
        )

    predictions.to_csv(outputs["predictions_csv"], index=False)
    return outputs


def unknown_threshold_sweep(
    *,
    true_family: np.ndarray,
    family_argmax: np.ndarray,
    family_confidence: np.ndarray,
    thresholds: list[float] | tuple[float, ...] | np.ndarray,
) -> pd.DataFrame:
    """Summarize coverage and known-prediction accuracy over unknown thresholds."""
    true_family = np.asarray(true_family).astype(str)
    family_argmax = np.asarray(family_argmax).astype(str)
    confidence = np.asarray(family_confidence, dtype=float)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        known = confidence >= float(threshold)
        coverage = float(known.mean()) if known.size else 0.0
        known_accuracy = float((family_argmax[known] == true_family[known]).mean()) if known.any() else float("nan")
        overall_accuracy = float(((family_argmax == true_family) & known).mean()) if known.size else 0.0
        rows.append(
            {
                "unknown_threshold": float(threshold),
                "coverage": coverage,
                "known_accuracy": known_accuracy,
                "overall_accuracy_with_unknown_as_incorrect": overall_accuracy,
                "unknown_rate": 1.0 - coverage,
            }
        )
    return pd.DataFrame(rows)


def _fit_classifier(features: np.ndarray, labels: np.ndarray, *, seed: int) -> _ProbabilisticClassifier:
    if len(np.unique(labels)) < 2:
        raise ValueError("Reason attribution classifier requires at least two classes")
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return _fit_nearest_centroid(features, labels)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    classifier.fit(features, labels)
    return _ProbabilisticClassifier(
        method="logistic",
        classes=classifier.classes_.astype(str),
        coef=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
    )


def _fit_nearest_centroid(features: np.ndarray, labels: np.ndarray) -> _ProbabilisticClassifier:
    classes = np.asarray(sorted(np.unique(labels))).astype(str)
    centroids = np.vstack([features[labels == cls].mean(axis=0) for cls in classes]).astype(np.float32)
    return _ProbabilisticClassifier(
        method="nearest_centroid",
        classes=classes,
        centroids=centroids,
    )


def _validate_feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"features must be a 2D numeric matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain NaN or infinite values")
    return matrix


def _validate_labels(
    labels: np.ndarray | list[str] | pd.Series,
    *,
    allowed: tuple[str, ...],
    target_name: str,
) -> np.ndarray:
    array = np.asarray(labels).astype(str)
    if array.size == 0:
        raise ValueError(f"{target_name} labels are empty")
    empty = pd.Series(array).astype(str).str.strip().eq("").to_numpy()
    if empty.any():
        raise ValueError(f"{target_name} labels contain empty values")
    unknown = sorted(set(array) - set(allowed))
    if unknown:
        raise ValueError(f"{target_name} labels contain unsupported values: {unknown}")
    return array


def _logistic_proba(features: np.ndarray, coef: np.ndarray, intercept: np.ndarray, n_classes: int) -> np.ndarray:
    logits = features @ coef.T + intercept
    if n_classes == 2 and logits.shape[1] == 1:
        prob_positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        probabilities = np.column_stack([1.0 - prob_positive, prob_positive])
        return probabilities.astype(float)
    return _softmax(logits)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(float)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _pairwise_euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    squared = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return np.sqrt(squared).astype(np.float32)


def _metrics_table(y_true: np.ndarray, y_pred: np.ndarray, *, labels: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "accuracy", "class": "overall", "value": accuracy_score(y_true, y_pred)},
        {
            "metric": "macro_f1",
            "class": "overall",
            "value": f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0),
        },
    ]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )
    for index, label in enumerate(labels):
        rows.extend(
            [
                {"metric": "precision", "class": label, "value": precision[index]},
                {"metric": "recall", "class": label, "value": recall[index]},
                {"metric": "f1", "class": label, "value": f1[index]},
                {"metric": "support", "class": label, "value": support[index]},
            ]
        )
    return pd.DataFrame(rows)


def _confusion_table(y_true: np.ndarray, y_pred: np.ndarray, *, labels: tuple[str, ...]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))


def _write_csv_and_markdown(dataframe: pd.DataFrame, csv_path: Path, markdown_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)
    markdown_path.write_text(f"# {title}\n\n{dataframe_to_markdown(dataframe)}\n", encoding="utf-8")


def _classifier_payload(prefix: str, classifier: _ProbabilisticClassifier) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_method": np.asarray(classifier.method),
        f"{prefix}_classes": classifier.classes.astype(str),
        f"{prefix}_coef": _or_empty(classifier.coef),
        f"{prefix}_intercept": _or_empty(classifier.intercept),
        f"{prefix}_centroids": _or_empty(classifier.centroids),
    }


def _empty_classifier_payload(prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_method": np.asarray(""),
        f"{prefix}_classes": np.asarray([], dtype=str),
        f"{prefix}_coef": np.empty((0, 0), dtype=np.float32),
        f"{prefix}_intercept": np.empty((0,), dtype=np.float32),
        f"{prefix}_centroids": np.empty((0, 0), dtype=np.float32),
    }


def _classifier_from_payload(data: Any, prefix: str) -> _ProbabilisticClassifier:
    method = str(data[f"{prefix}_method"])
    coef = data[f"{prefix}_coef"].astype(np.float32)
    intercept = data[f"{prefix}_intercept"].astype(np.float32)
    centroids = data[f"{prefix}_centroids"].astype(np.float32)
    return _ProbabilisticClassifier(
        method=method,
        classes=data[f"{prefix}_classes"].astype(str),
        coef=None if coef.size == 0 else coef,
        intercept=None if intercept.size == 0 else intercept,
        centroids=None if centroids.size == 0 else centroids,
    )


def _or_empty(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=np.float32)
    return value.astype(np.float32)
