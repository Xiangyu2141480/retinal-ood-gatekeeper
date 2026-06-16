"""Reusable robustness-analysis helpers for dissertation OOD experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from retinal_ood.evaluation.metrics import compute_ood_metrics, confusion_at_threshold


def deterministic_id_subset(manifest: pd.DataFrame, *, size: int, seed: int) -> pd.DataFrame:
    """Return a deterministic subset of ID-only manifest rows."""
    if size <= 0:
        raise ValueError("size must be positive")
    table = manifest.copy()
    if "label" in table.columns:
        table = table[table["label"].astype(int) == 0]
    if "ood_type" in table.columns:
        table = table[table["ood_type"].astype(str) == "id"]
    if table.empty:
        raise ValueError("manifest contains no ID rows")
    actual_size = min(size, len(table))
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(table), size=actual_size, replace=False))
    return table.iloc[positions].reset_index(drop=True)


def threshold_policy_sweep(
    scores: pd.DataFrame,
    *,
    val_id_scores: np.ndarray,
    policies: Sequence[str],
) -> pd.DataFrame:
    """Evaluate threshold policies against one score table."""
    table = _prepare_score_table(scores)
    labels = table["label"].astype(int).to_numpy()
    score_values = table["score"].astype(float).to_numpy()
    val_id_scores = np.asarray(val_id_scores, dtype=float)
    if val_id_scores.size == 0:
        raise ValueError("val_id_scores must not be empty")

    rows: list[dict[str, Any]] = []
    for policy in policies:
        threshold, policy_kind = _threshold_for_policy(
            policy,
            labels=labels,
            scores=score_values,
            val_id_scores=val_id_scores,
        )
        confusion = confusion_at_threshold(labels, score_values, threshold)
        id_count = int((labels == 0).sum())
        ood_count = int((labels == 1).sum())
        rows.append(
            {
                "policy": policy,
                "policy_kind": policy_kind,
                "threshold": threshold,
                "id_false_rejection_count": confusion["fp"],
                "id_false_rejection_rate": confusion["fp"] / id_count if id_count else np.nan,
                "ood_recall": confusion["tp"] / ood_count if ood_count else np.nan,
                "per_ood_type_recall": json.dumps(_recall_by_group(table, threshold, "ood_type"), sort_keys=True),
                "per_ood_subtype_recall": json.dumps(
                    _recall_by_group(table, threshold, "ood_subtype"),
                    sort_keys=True,
                ),
                "id_count": id_count,
                "ood_count": ood_count,
            }
        )
    return pd.DataFrame(rows)


def artifact_severity_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate artifact severity scores and reject rates."""
    required = {"scheme", "artifact_type", "severity_value", "score", "deployment_threshold", "research_threshold"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"artifact severity results missing columns: {missing}")
    table = results.copy()
    rows: list[dict[str, Any]] = []
    for (scheme, artifact_type), artifact_rows in table.groupby(["scheme", "artifact_type"], sort=True):
        severity_score_spearman = float(
            artifact_rows["severity_value"].rank().corr(artifact_rows["score"].rank(), method="pearson")
        )
        for severity_value, severity_rows in artifact_rows.groupby("severity_value", sort=True):
            rows.append(
                {
                    "scheme": scheme,
                    "artifact_type": artifact_type,
                    "severity_value": float(severity_value),
                    "mean_score": float(severity_rows["score"].mean()),
                    "median_score": float(severity_rows["score"].median()),
                    "deployment_reject_rate": float(
                        (severity_rows["score"] >= severity_rows["deployment_threshold"]).mean()
                    ),
                    "research_reject_rate": float(
                        (severity_rows["score"] >= severity_rows["research_threshold"]).mean()
                    ),
                    "n_images": int(len(severity_rows)),
                    "severity_score_spearman": severity_score_spearman,
                }
            )
    return pd.DataFrame(rows)


def method_disagreement_table(
    scores_by_method: Mapping[str, pd.DataFrame],
    *,
    thresholds: Mapping[str, float],
    primary_method: str,
    secondary_method: str,
) -> pd.DataFrame:
    """Merge per-method scores and label common disagreement/failure categories."""
    if primary_method not in scores_by_method or secondary_method not in scores_by_method:
        raise ValueError("primary_method and secondary_method must both be present")
    merged: pd.DataFrame | None = None
    metadata_columns = ["image_path", "label", "ood_type", "ood_subtype"]
    for method, raw_scores in scores_by_method.items():
        threshold = float(thresholds[method])
        scores = _prepare_score_table(raw_scores)
        method_frame = scores[metadata_columns + ["score"]].copy()
        method_frame[f"score_{method}"] = method_frame.pop("score")
        method_frame[f"prediction_{method}"] = (method_frame[f"score_{method}"] >= threshold).astype(int)
        method_frame[f"threshold_{method}"] = threshold
        if merged is None:
            merged = method_frame
        else:
            merged = merged.merge(
                method_frame[["image_path", f"score_{method}", f"prediction_{method}", f"threshold_{method}"]],
                on="image_path",
                how="inner",
            )
    if merged is None:
        raise ValueError("scores_by_method must not be empty")
    merged["case_type"] = [
        _case_type(row, primary_method=primary_method, secondary_method=secondary_method)
        for _, row in merged.iterrows()
    ]
    return merged


def pca_projection(features: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    """Project features to two PCA dimensions and append selected metadata."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if features.shape[0] != len(metadata):
        raise ValueError("features and metadata must contain the same number of rows")
    if features.shape[0] < 2:
        raise ValueError("at least two samples are required for PCA")
    projection = PCA(n_components=2, random_state=0).fit_transform(features)
    columns = [column for column in ["image_path", "label", "ood_type", "ood_subtype", "score"] if column in metadata]
    output = metadata.loc[:, columns].copy().reset_index(drop=True)
    output["pc1"] = projection[:, 0]
    output["pc2"] = projection[:, 1]
    return output


def subtype_influence_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute overall metrics with each subtype isolated and removed."""
    table = _prepare_score_table(scores)
    rows: list[dict[str, Any]] = []
    for subtype in sorted(str(value) for value in table["ood_subtype"].dropna().unique() if value != "id"):
        only = pd.concat(
            [
                table[table["label"].astype(int) == 0],
                table[(table["label"].astype(int) == 1) & (table["ood_subtype"].astype(str) == subtype)],
            ],
            ignore_index=True,
        )
        removed = table[
            (table["label"].astype(int) == 0)
            | ((table["label"].astype(int) == 1) & (table["ood_subtype"].astype(str) != subtype))
        ]
        for mode, subset in [("only_subtype", only), ("removed_subtype", removed)]:
            metrics = compute_ood_metrics(subset["label"].to_numpy(), subset["score"].to_numpy())
            rows.append(
                {
                    "ood_subtype": subtype,
                    "analysis": mode,
                    "auroc": metrics.auroc,
                    "auprc": metrics.auprc,
                    "fpr_at_95_tpr": metrics.fpr_at_95_tpr,
                    "id_count": int((subset["label"].astype(int) == 0).sum()),
                    "ood_count": int((subset["label"].astype(int) == 1).sum()),
                }
            )
    return pd.DataFrame(rows)


def _prepare_score_table(scores: pd.DataFrame) -> pd.DataFrame:
    table = scores.copy()
    for column in ["label", "score"]:
        if column not in table.columns:
            raise ValueError(f"scores table missing required column: {column}")
        table[column] = pd.to_numeric(table[column], errors="raise")
    if "image_path" not in table.columns:
        table["image_path"] = [f"row_{index}" for index in range(len(table))]
    if "ood_type" not in table.columns:
        table["ood_type"] = np.where(table["label"].astype(int) == 0, "id", "unknown")
    if "ood_subtype" not in table.columns:
        table["ood_subtype"] = np.where(table["label"].astype(int) == 0, "id", "unknown")
    return table


def _threshold_for_policy(
    policy: str,
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    val_id_scores: np.ndarray,
) -> tuple[float, str]:
    if policy == "research_95_tpr":
        return _threshold_at_tpr(labels, scores, 0.95), "research"
    if policy.startswith("val_id_quantile_"):
        quantile_text = policy.removeprefix("val_id_quantile_").replace("_", ".")
        quantile = float(quantile_text) / 100.0
        return float(np.quantile(val_id_scores, quantile)), "deployment"
    if policy.startswith("fixed_id_rejection_"):
        budget = float(policy.removeprefix("fixed_id_rejection_")) / 100.0
        return float(np.quantile(val_id_scores, 1.0 - budget)), "deployment"
    raise ValueError(f"Unsupported threshold policy: {policy}")


def _threshold_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    eligible = np.where(tpr >= target_tpr)[0]
    if eligible.size == 0:
        return float(np.nanmax(scores))
    best_fpr = np.min(fpr[eligible])
    best = eligible[fpr[eligible] == best_fpr]
    finite = thresholds[best][np.isfinite(thresholds[best])]
    if finite.size:
        return float(finite[0])
    return float(np.nanmax(scores))


def _recall_by_group(table: pd.DataFrame, threshold: float, group_column: str) -> dict[str, float]:
    if group_column not in table.columns:
        return {}
    output: dict[str, float] = {}
    ood = table[table["label"].astype(int) == 1]
    for group, rows in ood.groupby(group_column, sort=True):
        output[str(group)] = float((rows["score"].astype(float) >= threshold).mean())
    return output


def _case_type(row: pd.Series, *, primary_method: str, secondary_method: str) -> str:
    label = int(row["label"])
    primary_prediction = int(row[f"prediction_{primary_method}"])
    secondary_prediction = int(row[f"prediction_{secondary_method}"])
    primary_correct = primary_prediction == label
    secondary_correct = secondary_prediction == label
    if label == 0 and primary_prediction == 1:
        return "primary_false_positive_id"
    if label == 1 and primary_prediction == 0:
        return "primary_false_negative_ood"
    if primary_correct and secondary_correct:
        return "all_methods_correct"
    if primary_correct and not secondary_correct:
        return "primary_correct_secondary_wrong"
    if not primary_correct and secondary_correct:
        return "secondary_correct_primary_wrong"
    return "all_methods_wrong"
