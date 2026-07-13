"""Systematic comparison for optional Stage 2 reason attribution methods."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

from retinal_ood.evaluation.report_tables import dataframe_to_markdown
from retinal_ood.reason_attribution.classifier import (
    FAMILY_CLASSES,
    SUBTYPE_CLASSES,
    UNKNOWN_LABEL,
    ReasonPredictions,
)
from retinal_ood.reason_attribution.features import (
    FeatureExtractionConfig,
    FeatureTable,
    extract_features_from_manifest,
)
from retinal_ood.reason_attribution.methods import (
    FeatureSetName,
    ReasonMethodModel,
    ReasonMethodSpec,
    extract_global_pooled_features_from_manifest,
    fit_reason_method,
    resolve_method_specs,
)

BASELINE_PR24_FAMILY_MACRO_F1 = 0.8947
BASELINE_PR24_SUBTYPE_MACRO_F1 = 0.6724
DEFAULT_THRESHOLDS = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for a Stage 2 method-comparison run."""

    train_manifest: str | Path
    val_manifest: str | Path
    test_manifest: str | Path
    root_dir: str | Path
    out_dir: str | Path
    figures_dir: str | Path
    seed: int = 42
    unknown_threshold: float = 0.5
    include_subtype: bool = False
    include_hierarchical: bool = False
    include_rbf_svm: bool = False
    method_names: tuple[str, ...] | None = None
    stage1_guard_paths: tuple[str | Path, ...] = ()
    global_image_size: int = 24
    statistics_image_size: int = 224
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class ComparisonResult:
    """Summary of generated comparison artifacts."""

    out_dir: Path
    figures_dir: Path
    split_sizes: dict[str, int]
    best_family_method: str
    best_subtype_method: str
    selected_method: str
    stage1_file_hashes_before: dict[str, str]
    tables: dict[str, Path]
    figures: dict[str, Path]


@dataclass(frozen=True)
class _SplitFeatures:
    metadata: pd.DataFrame
    statistics: np.ndarray
    global_features: np.ndarray
    fusion: np.ndarray

    def matrix(self, feature_set: FeatureSetName) -> np.ndarray:
        if feature_set == "statistics":
            return self.statistics
        if feature_set == "global":
            return self.global_features
        if feature_set == "fusion":
            return self.fusion
        raise ValueError(f"Unsupported feature set: {feature_set}")


def run_reason_attribution_method_comparison(config: ComparisonConfig) -> ComparisonResult:
    """Train, evaluate, and report multiple post-rejection reason methods."""
    out_dir = Path(config.out_dir)
    figures_dir = Path(config.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stage1_hashes_before = snapshot_file_hashes(config.stage1_guard_paths)
    manifests = _load_and_validate_manifests(config)
    split_sizes = {split: len(metadata) for split, metadata in manifests.items()}
    features = _extract_feature_sets(config)
    specs = resolve_method_specs(
        method_names=config.method_names,
        include_hierarchical=config.include_hierarchical,
        include_rbf_svm=config.include_rbf_svm,
    )

    family_records: list[dict[str, Any]] = []
    subtype_records: list[dict[str, Any]] = []
    per_family_records: list[dict[str, Any]] = []
    per_subtype_records: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    fitted: dict[str, ReasonMethodModel] = {}
    predictions: dict[tuple[str, str], ReasonPredictions] = {}

    train_split = features["train"]
    for spec in specs:
        try:
            model = fit_reason_method(
                spec,
                train_split.matrix(spec.feature_set),
                train_split.metadata["ood_type"].astype(str).to_numpy(),
                subtype_labels=(
                    train_split.metadata["ood_subtype"].astype(str).to_numpy()
                    if config.include_subtype
                    else None
                ),
                include_subtype=config.include_subtype,
                seed=config.seed,
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            skipped.append({"method": spec.name, "reason": str(exc)})
            continue
        fitted[spec.name] = model

    if not fitted:
        raise RuntimeError("No reason attribution methods were fitted successfully")

    fitted_specs = [spec for spec in specs if spec.name in fitted]
    for spec in fitted_specs:
        split_records = _evaluate_method_on_split(
            spec=spec,
            model=fitted[spec.name],
            split_name="val",
            split_features=features["val"],
            include_subtype=config.include_subtype,
            unknown_threshold=config.unknown_threshold,
            thresholds=np.asarray(config.thresholds, dtype=float),
        )
        predictions[(spec.name, "val")] = split_records["predictions"]
        family_records.extend(split_records["family_records"])
        subtype_records.extend(split_records["subtype_records"])
        per_family_records.extend(split_records["per_family_records"])
        per_subtype_records.extend(split_records["per_subtype_records"])
        threshold_records.extend(split_records["threshold_records"])

    family_metrics = pd.DataFrame(family_records)
    subtype_metrics = pd.DataFrame(subtype_records)

    best_family_method = select_best_method(family_metrics, split="val")
    best_subtype_method = _select_best_subtype_method(subtype_metrics, include_subtype=config.include_subtype)
    selected_method = best_family_method

    selected_models = _selected_models_payload(
        config=config,
        split_sizes=split_sizes,
        family_metrics=family_metrics,
        subtype_metrics=subtype_metrics,
        best_family_method=best_family_method,
        best_subtype_method=best_subtype_method,
    )

    for spec in fitted_specs:
        split_records = _evaluate_method_on_split(
            spec=spec,
            model=fitted[spec.name],
            split_name="test",
            split_features=features["test"],
            include_subtype=config.include_subtype,
            unknown_threshold=config.unknown_threshold,
            thresholds=np.asarray(config.thresholds, dtype=float),
        )
        predictions[(spec.name, "test")] = split_records["predictions"]
        family_records.extend(split_records["family_records"])
        subtype_records.extend(split_records["subtype_records"])
        per_family_records.extend(split_records["per_family_records"])
        per_subtype_records.extend(split_records["per_subtype_records"])
        threshold_records.extend(split_records["threshold_records"])

    family_metrics = pd.DataFrame(family_records)
    subtype_metrics = pd.DataFrame(subtype_records)
    per_family = pd.DataFrame(per_family_records)
    per_subtype = pd.DataFrame(per_subtype_records)
    threshold_metrics = pd.DataFrame(threshold_records)

    confusion_tables = _write_best_confusion_tables(
        out_dir=out_dir,
        features=features,
        predictions=predictions,
        best_family_method=best_family_method,
        best_subtype_method=best_subtype_method,
        include_subtype=config.include_subtype,
    )
    selected_predictions = _write_selected_predictions_table(
        out_dir=out_dir,
        test_features=features["test"],
        predictions=predictions,
        best_family_method=best_family_method,
        best_subtype_method=best_subtype_method,
        unknown_threshold=config.unknown_threshold,
    )
    best_summary = _best_summary_table(
        family_metrics=family_metrics,
        subtype_metrics=subtype_metrics,
        per_family=per_family,
        per_subtype=per_subtype,
        best_family_method=best_family_method,
        best_subtype_method=best_subtype_method,
        selected_method=selected_method,
        unknown_threshold=config.unknown_threshold,
    )
    overview = _method_overview_table(
        specs=specs,
        skipped=skipped,
        family_metrics=family_metrics,
        subtype_metrics=subtype_metrics,
    )

    tables = _write_required_tables(
        out_dir=out_dir,
        overview=overview,
        family_metrics=family_metrics,
        subtype_metrics=subtype_metrics,
        per_family=per_family,
        per_subtype=per_subtype,
        threshold_metrics=threshold_metrics,
        best_summary=best_summary,
        skipped=skipped,
        method_subset=config.method_names,
    )
    tables.update(confusion_tables)
    tables["selected_models"] = _write_selected_models_json(
        out_dir / "selected_models.json",
        payload=selected_models,
    )
    tables["predictions_test"] = selected_predictions

    figures = _write_required_figures(
        figures_dir=figures_dir,
        family_metrics=family_metrics,
        subtype_metrics=subtype_metrics,
        per_family=per_family,
        per_subtype=per_subtype,
        threshold_metrics=threshold_metrics,
        confusion_tables=confusion_tables,
        best_summary=best_summary,
        best_family_method=best_family_method,
        best_subtype_method=best_subtype_method,
        selected_method=selected_method,
        include_subtype=config.include_subtype,
    )

    if snapshot_file_hashes(config.stage1_guard_paths) != stage1_hashes_before:
        raise RuntimeError("Stage 1 guard files changed during Stage 2 comparison")

    return ComparisonResult(
        out_dir=out_dir,
        figures_dir=figures_dir,
        split_sizes=split_sizes,
        best_family_method=best_family_method,
        best_subtype_method=best_subtype_method,
        selected_method=selected_method,
        stage1_file_hashes_before=stage1_hashes_before,
        tables=tables,
        figures=figures,
    )


def select_best_method(
    metrics: pd.DataFrame,
    *,
    split: str = "val",
    metric_column: str = "family_macro_f1",
    tolerance: float = 1e-6,
) -> str:
    """Select a method by validation metric, then simplicity and name."""
    candidates = metrics[metrics["split"].astype(str) == split].copy()
    if "status" in candidates.columns:
        candidates = candidates[candidates["status"].astype(str) == "ok"]
    candidates = candidates.dropna(subset=[metric_column])
    if candidates.empty:
        raise ValueError(f"No candidates available for {metric_column} on split={split}")
    if "complexity" not in candidates.columns:
        candidates["complexity"] = 999
    best_score = float(candidates[metric_column].max())
    tied = candidates[candidates[metric_column] >= best_score - tolerance].copy()
    tied = tied.sort_values(
        [metric_column, "complexity", "method"],
        ascending=[False, True, True],
        kind="stable",
    )
    return str(tied.iloc[0]["method"])


def compute_unknown_threshold_curve(
    *,
    true_family: np.ndarray,
    predicted_family: np.ndarray,
    confidence: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Compute coverage and known-output accuracy over unknown thresholds."""
    true = np.asarray(true_family).astype(str)
    predicted = np.asarray(predicted_family).astype(str)
    score = np.asarray(confidence, dtype=float)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        known = score >= float(threshold)
        coverage = float(known.mean()) if known.size else 0.0
        known_accuracy = float((predicted[known] == true[known]).mean()) if known.any() else np.nan
        overall = float(((predicted == true) & known).mean()) if known.size else 0.0
        rows.append(
            {
                "unknown_threshold": float(threshold),
                "coverage": coverage,
                "known_accuracy": known_accuracy,
                "overall_accuracy_with_unknown_as_incorrect": overall,
                "unknown_rate": 1.0 - coverage,
            }
        )
    return pd.DataFrame(rows)


def snapshot_file_hashes(paths: tuple[str | Path, ...] | list[str | Path]) -> dict[str, str]:
    """Hash files that should not be modified by Stage 2 comparison code."""
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path)] = digest
    return hashes


def _load_and_validate_manifests(config: ComparisonConfig) -> dict[str, pd.DataFrame]:
    manifests = {
        "train": pd.read_csv(config.train_manifest),
        "val": pd.read_csv(config.val_manifest),
        "test": pd.read_csv(config.test_manifest),
    }
    for split, metadata in manifests.items():
        _validate_stage2_manifest(metadata, split=split)
    return manifests


def _validate_stage2_manifest(metadata: pd.DataFrame, *, split: str) -> None:
    required = {"image_path", "ood_type", "ood_subtype"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Reason {split} manifest missing required columns: {missing}")
    if metadata.empty:
        raise ValueError(f"Reason {split} manifest is empty")
    if "label" in metadata.columns:
        labels = pd.to_numeric(metadata["label"], errors="coerce")
        if not labels.eq(1).all():
            raise ValueError("Stage 2 manifests must be OOD-only; found non-OOD label rows")
    family = metadata["ood_type"].astype(str)
    if family.str.lower().eq("id").any() or not set(family).issubset(set(FAMILY_CLASSES)):
        raise ValueError("Stage 2 manifests must be OOD-only with supported reason families")
    subtype = metadata["ood_subtype"].astype(str)
    empty_subtype = subtype.str.strip().eq("") | subtype.str.lower().eq("nan")
    if empty_subtype.any() or not set(subtype).issubset(set(SUBTYPE_CLASSES)):
        raise ValueError("Stage 2 manifests must contain supported reason subtypes")


def _extract_feature_sets(config: ComparisonConfig) -> dict[str, _SplitFeatures]:
    root_dir = Path(config.root_dir)
    manifest_paths = {
        "train": config.train_manifest,
        "val": config.val_manifest,
        "test": config.test_manifest,
    }
    extracted: dict[str, _SplitFeatures] = {}
    for split, manifest_path in manifest_paths.items():
        statistics = extract_features_from_manifest(
            manifest_path,
            root_dir=root_dir,
            config=FeatureExtractionConfig(
                feature_mode="image_statistics",
                image_size=config.statistics_image_size,
            ),
        )
        global_features = extract_global_pooled_features_from_manifest(
            manifest_path,
            root_dir=root_dir,
            image_size=config.global_image_size,
        )
        _assert_same_manifest_order(statistics, global_features, split=split)
        fusion = np.concatenate([global_features.features, statistics.features], axis=1).astype(np.float32)
        extracted[split] = _SplitFeatures(
            metadata=statistics.metadata,
            statistics=statistics.features.astype(np.float32),
            global_features=global_features.features.astype(np.float32),
            fusion=fusion,
        )
    return extracted


def _assert_same_manifest_order(left: FeatureTable, right: FeatureTable, *, split: str) -> None:
    left_paths = left.metadata["image_path"].astype(str).tolist()
    right_paths = right.metadata["image_path"].astype(str).tolist()
    if left_paths != right_paths:
        raise ValueError(f"Feature extractors produced different row order for {split}")


def _evaluate_method_on_split(
    *,
    spec: ReasonMethodSpec,
    model: ReasonMethodModel,
    split_name: str,
    split_features: _SplitFeatures,
    include_subtype: bool,
    unknown_threshold: float,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    split_predictions = model.predict(
        split_features.matrix(spec.feature_set),
        unknown_threshold=unknown_threshold,
    )
    threshold = compute_unknown_threshold_curve(
        true_family=split_features.metadata["ood_type"].astype(str).to_numpy(),
        predicted_family=split_predictions.family_argmax,
        confidence=split_predictions.family_confidence,
        thresholds=thresholds,
    )
    threshold["method"] = spec.name
    threshold["split"] = split_name
    subtype_rows, per_subtype_rows = _subtype_metric_rows(
        spec=spec,
        split=split_name,
        true_subtype=split_features.metadata["ood_subtype"].astype(str).to_numpy(),
        predictions=split_predictions,
        include_subtype=include_subtype,
    )
    return {
        "predictions": split_predictions,
        "family_records": _family_metric_rows(
            spec=spec,
            split=split_name,
            true_family=split_features.metadata["ood_type"].astype(str).to_numpy(),
            predictions=split_predictions,
            unknown_threshold=unknown_threshold,
            train_seconds=model.train_seconds,
        ),
        "per_family_records": _per_family_rows(
            spec=spec,
            split=split_name,
            true_family=split_features.metadata["ood_type"].astype(str).to_numpy(),
            predictions=split_predictions,
        ),
        "subtype_records": subtype_rows,
        "per_subtype_records": per_subtype_rows,
        "threshold_records": threshold.to_dict(orient="records"),
    }


def _family_metric_rows(
    *,
    spec: ReasonMethodSpec,
    split: str,
    true_family: np.ndarray,
    predictions: ReasonPredictions,
    unknown_threshold: float,
    train_seconds: float,
) -> list[dict[str, Any]]:
    true = np.asarray(true_family).astype(str)
    predicted = predictions.family.astype(str)
    argmax = predictions.family_argmax.astype(str)
    known = predicted != UNKNOWN_LABEL
    known_accuracy = float((argmax[known] == true[known]).mean()) if known.any() else np.nan
    return [
        {
            "method": spec.name,
            "split": split,
            "status": "ok",
            "feature_set": spec.feature_set,
            "estimator": spec.estimator_kind,
            "complexity": spec.complexity,
            "family_accuracy": accuracy_score(true, predicted),
            "family_argmax_accuracy": accuracy_score(true, argmax),
            "family_macro_f1": f1_score(
                true,
                predicted,
                labels=list(FAMILY_CLASSES),
                average="macro",
                zero_division=0,
            ),
            "family_balanced_accuracy": recall_score(
                true,
                predicted,
                labels=list(FAMILY_CLASSES),
                average="macro",
                zero_division=0,
            ),
            "unknown_threshold": unknown_threshold,
            "known_coverage_at_gamma": float(known.mean()) if known.size else 0.0,
            "unknown_rate_at_gamma": float((~known).mean()) if known.size else 0.0,
            "accuracy_excluding_unknown": known_accuracy,
            "train_seconds": float(train_seconds),
        }
    ]


def _per_family_rows(
    *,
    spec: ReasonMethodSpec,
    split: str,
    true_family: np.ndarray,
    predictions: ReasonPredictions,
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        np.asarray(true_family).astype(str),
        predictions.family.astype(str),
        labels=list(FAMILY_CLASSES),
        zero_division=0,
    )
    rows = []
    for index, label in enumerate(FAMILY_CLASSES):
        rows.append(
            {
                "method": spec.name,
                "split": split,
                "family": label,
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
        )
    return rows


def _subtype_metric_rows(
    *,
    spec: ReasonMethodSpec,
    split: str,
    true_subtype: np.ndarray,
    predictions: ReasonPredictions,
    include_subtype: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not include_subtype:
        return (
            [
                {
                    "method": spec.name,
                    "split": split,
                    "status": "not_requested",
                    "feature_set": spec.feature_set,
                    "estimator": spec.estimator_kind,
                    "complexity": spec.complexity,
                    "subtype_accuracy": np.nan,
                    "subtype_macro_f1": np.nan,
                }
            ],
            [],
        )
    if predictions.subtype is None:
        return (
            [
                {
                    "method": spec.name,
                    "split": split,
                    "status": "skipped",
                    "feature_set": spec.feature_set,
                    "estimator": spec.estimator_kind,
                    "complexity": spec.complexity,
                    "subtype_accuracy": np.nan,
                    "subtype_macro_f1": np.nan,
                }
            ],
            [],
        )
    true = np.asarray(true_subtype).astype(str)
    predicted = predictions.subtype.astype(str)
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=list(SUBTYPE_CLASSES),
        zero_division=0,
    )
    per_subtype = [
        {
            "method": spec.name,
            "split": split,
            "subtype": label,
            "precision": precision[index],
            "recall": recall[index],
            "f1": f1[index],
            "support": int(support[index]),
        }
        for index, label in enumerate(SUBTYPE_CLASSES)
    ]
    return (
        [
            {
                "method": spec.name,
                "split": split,
                "status": "ok",
                "feature_set": spec.feature_set,
                "estimator": spec.estimator_kind,
                "complexity": spec.complexity,
                "subtype_accuracy": accuracy_score(true, predicted),
                "subtype_macro_f1": f1_score(
                    true,
                    predicted,
                    labels=list(SUBTYPE_CLASSES),
                    average="macro",
                    zero_division=0,
                ),
            }
        ],
        per_subtype,
    )


def _select_best_subtype_method(subtype_metrics: pd.DataFrame, *, include_subtype: bool) -> str:
    if not include_subtype:
        return "not_requested"
    ok = subtype_metrics[subtype_metrics["status"].astype(str) == "ok"]
    if ok.empty:
        return "not_available"
    return select_best_method(ok, split="val", metric_column="subtype_macro_f1")


def _selected_models_payload(
    *,
    config: ComparisonConfig,
    split_sizes: dict[str, int],
    family_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame,
    best_family_method: str,
    best_subtype_method: str,
) -> dict[str, Any]:
    family_val = _row_for_method(family_metrics, method=best_family_method, split="val")
    subtype_val = None
    if best_subtype_method not in {"not_requested", "not_available"}:
        subtype_val = _row_for_method(subtype_metrics, method=best_subtype_method, split="val")
    return {
        "selected_family_method": best_family_method,
        "selected_subtype_method": best_subtype_method,
        "selected_family_validation_value": float(family_val["family_macro_f1"]),
        "selected_subtype_validation_value": (
            None if subtype_val is None else float(subtype_val["subtype_macro_f1"])
        ),
        "selection_metric_family": "family_macro_f1",
        "selection_metric_subtype": "subtype_macro_f1",
        "tie_break_rule": "highest validation macro-F1, then lower complexity, then method name",
        "seed": int(config.seed),
        "unknown_threshold": float(config.unknown_threshold),
        "threshold_policy": (
            f"Fixed unknown threshold {float(config.unknown_threshold):.1f}; "
            "validation and test curves are descriptive only."
        ),
        "manifests": {
            "train": str(Path(config.train_manifest)),
            "val": str(Path(config.val_manifest)),
            "test": str(Path(config.test_manifest)),
        },
        "split_sizes": {split: int(size) for split, size in split_sizes.items()},
        "test_evaluation_started_after_selection": True,
    }


def _write_best_confusion_tables(
    *,
    out_dir: Path,
    features: dict[str, _SplitFeatures],
    predictions: dict[tuple[str, str], ReasonPredictions],
    best_family_method: str,
    best_subtype_method: str,
    include_subtype: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    test = features["test"]
    family_predictions = predictions[(best_family_method, "test")]
    family_confusion = pd.DataFrame(
        confusion_matrix(
            test.metadata["ood_type"].astype(str).to_numpy(),
            family_predictions.family_argmax.astype(str),
            labels=list(FAMILY_CLASSES),
        ),
        index=list(FAMILY_CLASSES),
        columns=list(FAMILY_CLASSES),
    )
    family_path = out_dir / "best_reason_family_confusion_matrix.csv"
    family_confusion.to_csv(family_path)
    outputs["best_family_confusion_csv"] = family_path
    canonical_family_path = out_dir / "family_confusion_matrix.csv"
    family_confusion.to_csv(canonical_family_path)
    outputs["family_confusion_csv"] = canonical_family_path

    subtype_path = out_dir / "best_subtype_confusion_matrix.csv"
    if include_subtype and best_subtype_method in {method for method, split in predictions if split == "test"}:
        subtype_predictions = predictions[(best_subtype_method, "test")]
        if subtype_predictions.subtype is not None:
            subtype_confusion = pd.DataFrame(
                confusion_matrix(
                    test.metadata["ood_subtype"].astype(str).to_numpy(),
                    subtype_predictions.subtype.astype(str),
                    labels=list(SUBTYPE_CLASSES),
                ),
                index=list(SUBTYPE_CLASSES),
                columns=list(SUBTYPE_CLASSES),
            )
            subtype_confusion.to_csv(subtype_path)
            outputs["best_subtype_confusion_csv"] = subtype_path
            canonical_subtype_path = out_dir / "subtype_confusion_matrix.csv"
            subtype_confusion.to_csv(canonical_subtype_path)
            outputs["subtype_confusion_csv"] = canonical_subtype_path
    return outputs


def _write_selected_predictions_table(
    *,
    out_dir: Path,
    test_features: _SplitFeatures,
    predictions: dict[tuple[str, str], ReasonPredictions],
    best_family_method: str,
    best_subtype_method: str,
    unknown_threshold: float,
) -> Path:
    family_predictions = predictions[(best_family_method, "test")]
    subtype_predictions = predictions.get((best_subtype_method, "test"))
    subtype_values = (
        subtype_predictions.subtype.astype(str)
        if subtype_predictions is not None and subtype_predictions.subtype is not None
        else np.full(len(test_features.metadata), "not_available", dtype=object)
    )
    subtype_confidence = (
        subtype_predictions.subtype_confidence.astype(float)
        if subtype_predictions is not None and subtype_predictions.subtype_confidence is not None
        else np.full(len(test_features.metadata), np.nan, dtype=float)
    )
    predictions_table = pd.DataFrame(
        {
            "image_path": test_features.metadata["image_path"].astype(str).to_numpy(),
            "true_family": test_features.metadata["ood_type"].astype(str).to_numpy(),
            "true_subtype": test_features.metadata["ood_subtype"].astype(str).to_numpy(),
            "predicted_family": family_predictions.family.astype(str),
            "argmax_family": family_predictions.family_argmax.astype(str),
            "family_confidence": family_predictions.family_confidence.astype(float),
            "predicted_subtype": subtype_values,
            "subtype_confidence": subtype_confidence,
            "selected_family_method": best_family_method,
            "selected_subtype_method": best_subtype_method,
            "unknown_threshold": float(unknown_threshold),
        }
    )
    path = out_dir / "predictions_test.csv"
    predictions_table.to_csv(path, index=False)
    return path


def _best_summary_table(
    *,
    family_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame,
    per_family: pd.DataFrame,
    per_subtype: pd.DataFrame,
    best_family_method: str,
    best_subtype_method: str,
    selected_method: str,
    unknown_threshold: float,
) -> pd.DataFrame:
    family_test = _row_for_method(family_metrics, method=best_family_method, split="test")
    family_val = _row_for_method(family_metrics, method=best_family_method, split="val")
    hardest_family = _hardest_label(
        per_family[(per_family["method"] == best_family_method) & (per_family["split"] == "test")],
        label_column="family",
    )

    subtype_val_macro = np.nan
    subtype_test_macro = np.nan
    hardest_subtype = "not_requested"
    if best_subtype_method not in {"not_requested", "not_available"}:
        subtype_val = _row_for_method(subtype_metrics, method=best_subtype_method, split="val")
        subtype_test = _row_for_method(subtype_metrics, method=best_subtype_method, split="test")
        subtype_val_macro = float(subtype_val.get("subtype_macro_f1", np.nan))
        subtype_test_macro = float(subtype_test.get("subtype_macro_f1", np.nan))
        hardest_subtype = _hardest_label(
            per_subtype[
                (per_subtype["method"] == best_subtype_method) & (per_subtype["split"] == "test")
            ],
            label_column="subtype",
        )

    family_test_macro = float(family_test["family_macro_f1"])
    subtype_delta = subtype_test_macro - BASELINE_PR24_SUBTYPE_MACRO_F1
    rows = [
        ("best_reason_family_method", best_family_method),
        ("best_reason_family_validation_macro_f1", f"{float(family_val['family_macro_f1']):.4f}"),
        ("best_reason_family_test_accuracy", f"{float(family_test['family_accuracy']):.4f}"),
        ("best_reason_family_test_macro_f1", f"{family_test_macro:.4f}"),
        ("best_reason_family_test_balanced_accuracy", f"{float(family_test['family_balanced_accuracy']):.4f}"),
        ("family_macro_f1_delta_vs_pr24_baseline", f"{family_test_macro - BASELINE_PR24_FAMILY_MACRO_F1:+.4f}"),
        ("family_improves_over_pr24_baseline", str(family_test_macro > BASELINE_PR24_FAMILY_MACRO_F1)),
        ("unknown_threshold_gamma", f"{unknown_threshold:.2f}"),
        ("known_coverage_at_gamma", f"{float(family_test['known_coverage_at_gamma']):.4f}"),
        ("unknown_rate_at_gamma", f"{float(family_test['unknown_rate_at_gamma']):.4f}"),
        ("accuracy_excluding_unknown", f"{float(family_test['accuracy_excluding_unknown']):.4f}"),
        ("best_subtype_method", best_subtype_method),
        ("best_subtype_validation_macro_f1", _format_optional_float(subtype_val_macro)),
        ("best_subtype_test_macro_f1", _format_optional_float(subtype_test_macro)),
        ("subtype_macro_f1_delta_vs_pr24_baseline", _format_optional_float(subtype_delta)),
        ("hardest_family", hardest_family),
        ("hardest_subtype", hardest_subtype),
        ("selected_final_method", selected_method),
        (
            "selection_rationale",
            "Selected by validation reason-family macro-F1, with simpler methods preferred on ties.",
        ),
        (
            "conceptual_boundary",
            "Stage 1 remains ID-only unsupervised OOD detection; Stage 2 is post-hoc explanation.",
        ),
        ("clinical_scope", "Reason labels are likely rejection explanations, not disease predictions."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _method_overview_table(
    *,
    specs: tuple[ReasonMethodSpec, ...],
    skipped: list[dict[str, str]],
    family_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame,
) -> pd.DataFrame:
    skipped_by_method = {row["method"]: row["reason"] for row in skipped}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        status = "skipped" if spec.name in skipped_by_method else "ok"
        rows.append(
            {
                "method": spec.name,
                "status": status,
                "feature_set": spec.feature_set,
                "estimator": spec.estimator_kind,
                "complexity": spec.complexity,
                "description": spec.description,
                "skip_reason": skipped_by_method.get(spec.name, ""),
                "validation_family_macro_f1": _metric_lookup(
                    family_metrics,
                    spec.name,
                    "val",
                    "family_macro_f1",
                ),
                "test_family_macro_f1": _metric_lookup(
                    family_metrics,
                    spec.name,
                    "test",
                    "family_macro_f1",
                ),
                "validation_subtype_macro_f1": _metric_lookup(
                    subtype_metrics,
                    spec.name,
                    "val",
                    "subtype_macro_f1",
                ),
                "test_subtype_macro_f1": _metric_lookup(
                    subtype_metrics,
                    spec.name,
                    "test",
                    "subtype_macro_f1",
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_required_tables(
    *,
    out_dir: Path,
    overview: pd.DataFrame,
    family_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame,
    per_family: pd.DataFrame,
    per_subtype: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    best_summary: pd.DataFrame,
    skipped: list[dict[str, str]],
    method_subset: tuple[str, ...] | None,
) -> dict[str, Path]:
    tables = {
        "method_overview": _write_table(
            overview,
            out_dir / "method_overview.csv",
            out_dir / "method_overview.md",
            "Reason Attribution Method Overview",
        ),
        "family_metrics_by_method": _write_table(
            family_metrics,
            out_dir / "family_metrics_by_method.csv",
            out_dir / "family_metrics_by_method.md",
            "Family Metrics by Method",
        ),
        "subtype_metrics_by_method": _write_table(
            subtype_metrics,
            out_dir / "subtype_metrics_by_method.csv",
            out_dir / "subtype_metrics_by_method.md",
            "Subtype Metrics by Method",
        ),
        "per_family_f1_by_method": _write_table(
            per_family,
            out_dir / "per_family_f1_by_method.csv",
            out_dir / "per_family_f1_by_method.md",
            "Per-Family F1 by Method",
        ),
        "per_subtype_f1_by_method": _write_table(
            per_subtype,
            out_dir / "per_subtype_f1_by_method.csv",
            out_dir / "per_subtype_f1_by_method.md",
            "Per-Subtype F1 by Method",
        ),
        "unknown_threshold_by_method": _write_table(
            threshold_metrics,
            out_dir / "unknown_threshold_by_method.csv",
            out_dir / "unknown_threshold_by_method.md",
            "Unknown Threshold by Method",
        ),
        "best_method_summary": _write_table(
            best_summary,
            out_dir / "best_method_summary.csv",
            out_dir / "best_method_summary.md",
            "Best Method Summary",
        ),
        "method_comparison": _write_csv(overview, out_dir / "method_comparison.csv"),
        "family_metrics": _write_csv(family_metrics, out_dir / "family_metrics.csv"),
        "subtype_metrics": _write_csv(subtype_metrics, out_dir / "subtype_metrics.csv"),
    }
    tables["skipped_methods"] = _write_skipped_methods(
        out_dir / "skipped_methods.md",
        skipped=skipped,
        method_subset=method_subset,
    )
    return tables


def _write_required_figures(
    *,
    figures_dir: Path,
    family_metrics: pd.DataFrame,
    subtype_metrics: pd.DataFrame,
    per_family: pd.DataFrame,
    per_subtype: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    confusion_tables: dict[str, Path],
    best_summary: pd.DataFrame,
    best_family_method: str,
    best_subtype_method: str,
    selected_method: str,
    include_subtype: bool,
) -> dict[str, Path]:
    figures = {
        "family_macro_f1": _plot_family_macro_f1(
            family_metrics,
            figures_dir / "figure_reason_method_family_macro_f1.png",
        ),
        "accuracy_macro_f1": _plot_accuracy_macro_f1(
            family_metrics,
            figures_dir / "figure_reason_method_accuracy_macro_f1.png",
        ),
        "per_family_f1": _plot_per_class_heatmap(
            per_family[per_family["split"] == "test"],
            path=figures_dir / "figure_reason_method_per_family_f1.png",
            class_column="family",
            title="Per-family F1 by Stage 2 method",
        ),
        "per_subtype_f1": _plot_per_class_heatmap(
            per_subtype[per_subtype["split"] == "test"] if not per_subtype.empty else per_subtype,
            path=figures_dir / "figure_reason_method_per_subtype_f1.png",
            class_column="subtype",
            title="Per-subtype F1 by Stage 2 method",
        ),
        "best_family_confusion": _plot_confusion_matrix(
            confusion_tables["best_family_confusion_csv"],
            figures_dir / "figure_best_reason_family_confusion_matrix.png",
            title=f"Best family method: {best_family_method}",
        ),
        "unknown_threshold_tradeoff": _plot_threshold_tradeoff(
            threshold_metrics,
            figures_dir / "figure_unknown_threshold_tradeoff.png",
            method=best_family_method,
        ),
        "selection_summary": _plot_selection_summary(
            best_summary,
            figures_dir / "figure_reason_method_selection_summary.png",
        ),
        "two_stage_pipeline": _plot_two_stage_pipeline(
            figures_dir / "figure_two_stage_updated_pipeline.png",
            selected_method=selected_method,
        ),
    }
    subtype_confusion = confusion_tables.get("best_subtype_confusion_csv")
    if include_subtype and subtype_confusion is not None:
        figures["best_subtype_confusion"] = _plot_confusion_matrix(
            subtype_confusion,
            figures_dir / "figure_best_subtype_confusion_matrix.png",
            title=f"Best subtype method: {best_subtype_method}",
        )
    else:
        figures["best_subtype_confusion"] = _plot_placeholder(
            figures_dir / "figure_best_subtype_confusion_matrix.png",
            "Subtype comparison was not requested for this run.",
        )
    figures.update(_write_figure_docs(figures_dir))
    return figures


def _write_table(dataframe: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)
    md_path.write_text(f"# {title}\n\n{dataframe_to_markdown(dataframe)}\n", encoding="utf-8")
    return csv_path


def _write_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def _write_selected_models_json(path: Path, *, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_skipped_methods(
    path: Path,
    *,
    skipped: list[dict[str, str]],
    method_subset: tuple[str, ...] | None,
) -> Path:
    lines = ["# Skipped Reason Attribution Methods", ""]
    if skipped:
        for row in skipped:
            lines.append(f"- `{row['method']}`: {row['reason']}")
    else:
        lines.append("- No required comparison methods were skipped.")
    if method_subset is not None:
        lines.append("")
        lines.append(
            "This run used an explicit method subset for testing or development; "
            "unrequested methods are not counted as skipped."
        )
    lines.append("")
    lines.append(
        "Optional `rbf_svm_optional` is disabled by default to keep the dissertation "
        "comparison command bounded and reproducible."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _plot_family_macro_f1(metrics: pd.DataFrame, path: Path) -> Path:
    data = metrics[(metrics["split"] == "test") & (metrics["status"] == "ok")].copy()
    data = data.sort_values("family_macro_f1", ascending=False, kind="stable")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    ax.bar(data["method"], data["family_macro_f1"], color="#4c78a8")
    ax.axhline(BASELINE_PR24_FAMILY_MACRO_F1, color="#d62728", linestyle="--", linewidth=1.2)
    ax.text(
        0.99,
        BASELINE_PR24_FAMILY_MACRO_F1 + 0.01,
        "PR #24 baseline",
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        color="#7f1d1d",
    )
    ax.set_ylabel("Test family macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Reason family macro-F1 by Stage 2 method")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_accuracy_macro_f1(metrics: pd.DataFrame, path: Path) -> Path:
    data = metrics[(metrics["split"] == "test") & (metrics["status"] == "ok")].copy()
    data = data.sort_values("family_macro_f1", ascending=False, kind="stable")
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    ax.bar(x - width / 2, data["family_accuracy"], width, label="accuracy", color="#59a14f")
    ax.bar(x + width / 2, data["family_macro_f1"], width, label="macro-F1", color="#4c78a8")
    ax.set_xticks(x, labels=data["method"], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Reason family accuracy and macro-F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_per_class_heatmap(
    data: pd.DataFrame,
    *,
    path: Path,
    class_column: str,
    title: str,
) -> Path:
    if data.empty:
        return _plot_placeholder(path, "No subtype metrics were generated for this run.")
    pivot = data.pivot(index="method", columns=class_column, values="f1").fillna(0.0)
    fig, ax = plt.subplots(
        figsize=(max(7, 0.65 * len(pivot.columns)), max(3.5, 0.45 * len(pivot.index) + 1.5)),
        dpi=200,
    )
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="F1")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_confusion_matrix(csv_path: Path, path: Path, *, title: str) -> Path:
    matrix = pd.read_csv(csv_path, index_col=0)
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(5, 0.6 * len(matrix.columns)), max(4, 0.45 * len(matrix.index))),
        dpi=200,
    )
    im = ax.imshow(values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, str(int(values[row, col])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_threshold_tradeoff(metrics: pd.DataFrame, path: Path, *, method: str) -> Path:
    data = metrics[(metrics["split"] == "test") & (metrics["method"] == method)].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=200)
    ax.plot(data["unknown_threshold"], data["coverage"], marker="o", label="coverage")
    ax.plot(
        data["unknown_threshold"],
        data["known_accuracy"],
        marker="s",
        label="accuracy excluding unknown",
    )
    ax.plot(
        data["unknown_threshold"],
        data["overall_accuracy_with_unknown_as_incorrect"],
        marker="^",
        label="overall with unknown incorrect",
    )
    ax.set_xlabel("Unknown threshold gamma")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Unknown-threshold trade-off: {method}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_selection_summary(summary: pd.DataFrame, path: Path) -> Path:
    lookup = dict(zip(summary["item"].astype(str), summary["value"].astype(str)))
    lines = [
        "Stage 2 method selection",
        f"Selected method: {lookup.get('selected_final_method', 'not_available')}",
        f"Family macro-F1: {lookup.get('best_reason_family_test_macro_f1', 'n/a')}",
        f"Family accuracy: {lookup.get('best_reason_family_test_accuracy', 'n/a')}",
        f"Known coverage at gamma: {lookup.get('known_coverage_at_gamma', 'n/a')}",
        f"Best subtype method: {lookup.get('best_subtype_method', 'n/a')}",
        f"Subtype macro-F1: {lookup.get('best_subtype_test_macro_f1', 'n/a')}",
        f"Hardest family: {lookup.get('hardest_family', 'n/a')}",
        f"Hardest subtype: {lookup.get('hardest_subtype', 'n/a')}",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    ax.axis("off")
    ax.text(0.02, 0.94, lines[0], fontsize=16, weight="bold", transform=ax.transAxes)
    for index, line in enumerate(lines[1:]):
        ax.text(0.04, 0.82 - index * 0.085, line, fontsize=11, transform=ax.transAxes)
    ax.text(
        0.04,
        0.08,
        "Stage 1 remains ID-only unsupervised OOD detection; Stage 2 explains rejected inputs only.",
        fontsize=9,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_two_stage_pipeline(path: Path, *, selected_method: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=200)
    ax.axis("off")
    boxes = [
        (0.08, 0.57, "Input image"),
        (0.29, 0.57, "Stage 1\nID-only unsupervised\nOOD gatekeeper"),
        (0.50, 0.75, "ACCEPT\nvalid FAF"),
        (0.50, 0.37, "REJECT\nOOD / invalid"),
        (0.72, 0.37, f"Stage 2\npost-hoc reason attribution\n{selected_method}"),
        (0.91, 0.37, "reason family\noptional subtype\nunknown_ood"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#2f4858"},
            transform=ax.transAxes,
        )
    for start, end in [
        ((0.14, 0.57), (0.21, 0.57)),
        ((0.39, 0.63), (0.45, 0.73)),
        ((0.39, 0.51), (0.45, 0.39)),
        ((0.56, 0.37), (0.64, 0.37)),
        ((0.81, 0.37), (0.86, 0.37)),
    ]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.6}, xycoords=ax.transAxes)
    ax.text(
        0.42,
        0.09,
        "OOD taxonomy labels are used only to train/evaluate the optional Stage 2 explanation layer.",
        ha="center",
        fontsize=8.5,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_placeholder(path: Path, text: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _write_figure_docs(figures_dir: Path) -> dict[str, Path]:
    rows = [
        ("figure_reason_method_family_macro_f1.png", "Family macro-F1 comparison"),
        ("figure_reason_method_accuracy_macro_f1.png", "Accuracy and macro-F1 comparison"),
        ("figure_reason_method_per_family_f1.png", "Per-family F1 heatmap"),
        ("figure_reason_method_per_subtype_f1.png", "Per-subtype F1 heatmap"),
        ("figure_best_reason_family_confusion_matrix.png", "Best family method confusion matrix"),
        ("figure_best_subtype_confusion_matrix.png", "Best subtype method confusion matrix"),
        ("figure_unknown_threshold_tradeoff.png", "Unknown-threshold coverage trade-off"),
        ("figure_reason_method_selection_summary.png", "Stage 2 method selection summary"),
        ("figure_two_stage_updated_pipeline.png", "Two-stage pipeline with selected Stage 2 method"),
    ]
    index_path = figures_dir / "figure_index.md"
    index_table = pd.DataFrame(rows, columns=["path", "title"])
    index_path.write_text(
        "# Reason Attribution Method Comparison Figure Index\n\n"
        + dataframe_to_markdown(index_table)
        + "\n",
        encoding="utf-8",
    )
    captions_path = figures_dir / "caption_suggestions.md"
    captions_path.write_text(
        "\n".join(
            [
                "# Caption Suggestions",
                "",
                "- `figure_reason_method_family_macro_f1.png`: Test macro-F1 for each optional Stage 2 reason-attribution method, with the PR #24 baseline indicated for comparison.",
                "- `figure_reason_method_accuracy_macro_f1.png`: Test accuracy and macro-F1 for reason family attribution across methods.",
                "- `figure_reason_method_per_family_f1.png`: Per-family F1 scores for modality shift, sensory artifact, and semantic outlier explanations.",
                "- `figure_reason_method_per_subtype_f1.png`: Fine-grained subtype F1 scores, showing where subtype attribution is strongest and weakest.",
                "- `figure_best_reason_family_confusion_matrix.png`: Confusion matrix for the selected family-level reason attribution method.",
                "- `figure_best_subtype_confusion_matrix.png`: Confusion matrix for the best subtype attribution method.",
                "- `figure_unknown_threshold_tradeoff.png`: Coverage and accuracy trade-off as low-confidence predictions are mapped to `unknown_ood`.",
                "- `figure_reason_method_selection_summary.png`: Dissertation-ready summary of the selected Stage 2 method and its main metrics.",
                "- `figure_two_stage_updated_pipeline.png`: Updated pipeline emphasizing that Stage 1 remains ID-only and unsupervised while Stage 2 explains rejected inputs.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    guide_path = figures_dir / "figure_selection_guide.md"
    guide_path.write_text(
        "\n".join(
            [
                "# Figure Selection Guide",
                "",
                "| figure | include | suggested chapter | notes |",
                "| --- | --- | --- | --- |",
                "| figure_reason_method_family_macro_f1.png | must_include | Results | Main method-comparison result. |",
                "| figure_reason_method_accuracy_macro_f1.png | recommended | Results | Pairs accuracy with macro-F1 for readability. |",
                "| figure_reason_method_per_family_f1.png | recommended | Results | Shows hardest reason family. |",
                "| figure_reason_method_per_subtype_f1.png | appendix | Appendix | Dense but useful for fine-grained error analysis. |",
                "| figure_best_reason_family_confusion_matrix.png | must_include | Results | Best family model error structure. |",
                "| figure_best_subtype_confusion_matrix.png | appendix | Appendix | Best subtype model error structure. |",
                "| figure_unknown_threshold_tradeoff.png | recommended | Discussion | Supports unknown_ood threshold discussion. |",
                "| figure_reason_method_selection_summary.png | optional | Results | Compact summary slide-style figure. |",
                "| figure_two_stage_updated_pipeline.png | must_include | Methodology | Clarifies the Stage 1/Stage 2 boundary. |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "figure_index": index_path,
        "caption_suggestions": captions_path,
        "figure_selection_guide": guide_path,
    }


def _row_for_method(dataframe: pd.DataFrame, *, method: str, split: str) -> pd.Series:
    rows = dataframe[(dataframe["method"] == method) & (dataframe["split"] == split)]
    if rows.empty:
        raise ValueError(f"No metrics row found for method={method}, split={split}")
    return rows.iloc[0]


def _hardest_label(dataframe: pd.DataFrame, *, label_column: str) -> str:
    if dataframe.empty:
        return "not_available"
    row = dataframe.sort_values("f1", ascending=True, kind="stable").iloc[0]
    return f"{row[label_column]} (F1={float(row['f1']):.4f})"


def _metric_lookup(dataframe: pd.DataFrame, method: str, split: str, column: str) -> float:
    if dataframe.empty or column not in dataframe.columns:
        return np.nan
    rows = dataframe[(dataframe["method"] == method) & (dataframe["split"] == split)]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0][column])


def _format_optional_float(value: float) -> str:
    if not np.isfinite(value):
        return "not_available"
    return f"{float(value):.4f}"
