#!/usr/bin/env python
"""Generate robustness, safety, failure-analysis, and runtime dissertation outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_autoencoder import AutoencoderScoreDetector  # noqa: E402
from retinal_ood.baselines.feature_distance import (  # noqa: E402
    FeatureDistanceDetector,
    _knn_scores,
    _mahalanobis_scores,
    _regularized_inverse_covariance,
)
from retinal_ood.baselines.image_statistics import (  # noqa: E402
    ImageStatisticsDetector,
    extract_image_statistics_features,
)
from retinal_ood.data.dataset import ManifestImageDataset, collate_manifest_batch  # noqa: E402
from retinal_ood.data.transforms import build_transforms  # noqa: E402
from retinal_ood.evaluation.bootstrap import bootstrap_metric_ci, metric_snapshot  # noqa: E402
from retinal_ood.evaluation.report_tables import dataframe_to_markdown  # noqa: E402
from retinal_ood.evaluation.robustness import (  # noqa: E402
    artifact_severity_summary,
    method_disagreement_table,
    pca_projection,
    subtype_influence_table,
    threshold_policy_sweep,
)
from retinal_ood.models.autoencoder import load_autoencoder_checkpoint  # noqa: E402
from retinal_ood.models.patchcore import PatchCoreDetector  # noqa: E402
from retinal_ood.utils.seed import set_seed  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


SCHEMES = {
    "image_statistics": {
        "label": "Image statistics",
        "kind": "image_statistics",
        "score_csv": "reports/generated/dissertation_runs/multi_scheme_primary/runs/image_statistics/evaluation/scores.csv",
        "checkpoint": "reports/generated/dissertation_runs/multi_scheme_primary/runs/image_statistics/baseline_model.npz",
        "config": "reports/generated/dissertation_runs/multi_scheme_primary/configs/image_statistics.yaml",
        "complexity": "very low",
        "localization": "none",
    },
    "autoencoder": {
        "label": "Autoencoder",
        "kind": "autoencoder",
        "score_csv": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/autoencoder_baseline/evaluation/scores.csv",
        "checkpoint": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/autoencoder_baseline/model.pt",
        "config": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/configs/autoencoder_baseline.yaml",
        "complexity": "low",
        "localization": "weak reconstruction",
    },
    "global_feature_knn": {
        "label": "Global feature kNN",
        "kind": "global_feature_knn",
        "score_csv": "reports/generated/dissertation_runs/multi_scheme_primary/runs/global_feature_knn/evaluation/scores.csv",
        "checkpoint": "reports/generated/dissertation_runs/multi_scheme_primary/runs/global_feature_knn/baseline_model.npz",
        "config": "reports/generated/dissertation_runs/multi_scheme_primary/configs/global_feature_knn.yaml",
        "complexity": "medium",
        "localization": "none",
    },
    "mahalanobis_feature": {
        "label": "Mahalanobis feature",
        "kind": "mahalanobis_feature",
        "score_csv": "reports/generated/dissertation_runs/multi_scheme_primary/runs/mahalanobis_feature/evaluation/scores.csv",
        "checkpoint": "reports/generated/dissertation_runs/multi_scheme_primary/runs/mahalanobis_feature/baseline_model.npz",
        "config": "reports/generated/dissertation_runs/multi_scheme_primary/configs/mahalanobis_feature.yaml",
        "complexity": "medium",
        "localization": "none",
    },
    "patchcore_l3": {
        "label": "PatchCore L3",
        "kind": "patchcore",
        "score_csv": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer3/evaluation/scores.csv",
        "checkpoint": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer3/patchcore_memory.npz",
        "config": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/configs/patchcore_layer3.yaml",
        "complexity": "high",
        "localization": "patch heatmap",
    },
    "patchcore_l2_l3": {
        "label": "PatchCore L2+L3",
        "kind": "patchcore",
        "score_csv": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer2_layer3/evaluation/scores.csv",
        "checkpoint": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/runs/patchcore_layer2_layer3/patchcore_memory.npz",
        "config": "reports/generated/dissertation_runs/primary_balanced_by_subtype_10k/configs/patchcore_layer2_layer3.yaml",
        "complexity": "very high",
        "localization": "patch heatmap",
    },
}

BOOTSTRAP_SCHEMES = [
    "mahalanobis_feature",
    "patchcore_l3",
    "patchcore_l2_l3",
    "autoencoder",
    "global_feature_knn",
    "image_statistics",
]
THRESHOLD_SCHEMES = ["mahalanobis_feature", "patchcore_l3", "autoencoder", "image_statistics"]
SEVERITY_SCHEMES = ["mahalanobis_feature", "patchcore_l3", "autoencoder", "image_statistics"]
DISAGREEMENT_SCHEMES = ["mahalanobis_feature", "patchcore_l3", "autoencoder", "image_statistics"]
TRAIN_SIZE_SCHEMES = ["mahalanobis_feature", "global_feature_knn", "image_statistics"]
TRAIN_SIZES = [50, 100, 250, 500, 700]
THRESHOLD_POLICIES = [
    "research_95_tpr",
    "val_id_quantile_95",
    "val_id_quantile_97_5",
    "val_id_quantile_99",
    "val_id_quantile_99_5",
    "fixed_id_rejection_5",
    "fixed_id_rejection_10",
    "fixed_id_rejection_20",
]
PALETTE = {
    "blue": "#2F6F9F",
    "green": "#3B7F5C",
    "red": "#B85C5C",
    "gold": "#B38B2E",
    "gray": "#5B6470",
}


def _parse_score_root(value: str) -> dict[str, Path | str]:
    if "=" not in value:
        raise ValueError("--score-root values must use name=path")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError("--score-root values must use name=path")
    return {"name": name, "path": Path(raw_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dissertation robustness/failure-analysis package.")
    parser.add_argument("--results-dir", default="reports/dissertation_results/robustness_analysis")
    parser.add_argument("--figures-dir", default="reports/dissertation_figures/robustness")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--severity-sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--score-root",
        action="append",
        help="Optional extra score root formatted as name=path; retained for reproducibility notes.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = _resolve_device(args.device)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for value in args.score_root or []:
        _parse_score_root(value)

    repo = Path.cwd()
    scores = _load_primary_scores(repo)
    available = sorted(scores)
    _write_existing_artifact_inventory(results_dir / "existing_result_inventory.md", repo, available)

    bootstrap = _build_bootstrap_ci(scores, n_bootstrap=args.bootstrap_samples, seed=args.seed)
    _write_table(results_dir, "bootstrap_ci", bootstrap, "Bootstrap Confidence Intervals")

    val_scores = _score_validation_id(repo, device=device)
    threshold_sweep = _build_threshold_policy_sweep(scores, val_scores)
    _write_table(results_dir, "threshold_policy_sweep", threshold_sweep, "Threshold Policy Sweep")

    train_size = _build_train_size_sensitivity(repo, device=device, seed=args.seed)
    _write_table(results_dir, "train_size_sensitivity", train_size, "Train-Size Sensitivity")

    severity_raw, severity_summary, examples = _build_artifact_severity_stress(
        repo,
        scores,
        sample_size=args.severity_sample_size,
        seed=args.seed,
        device=device,
    )
    _write_table(results_dir, "artifact_severity_stress", severity_summary, "Artifact Severity Stress Test")
    severity_raw.to_csv(results_dir / "artifact_severity_stress_raw_scores.csv", index=False)

    disagreement = _build_method_disagreement(scores)
    _write_table(results_dir, "method_disagreement_cases", disagreement, "Method Disagreement Cases")

    projection = _build_feature_space_projection(repo, scores["mahalanobis_feature"], device=device)
    projection.to_csv(results_dir / "feature_space_projection.csv", index=False)

    runtime = _build_runtime_summary(repo, scores, device=device)
    _write_table(results_dir, "runtime_resource_summary", runtime, "Runtime and Resource Summary")

    subtype_rows = []
    for scheme in ["mahalanobis_feature", "patchcore_l3", "autoencoder", "image_statistics"]:
        table = subtype_influence_table(scores[scheme])
        table.insert(0, "scheme", scheme)
        table.insert(1, "scheme_label", SCHEMES[scheme]["label"])
        subtype_rows.append(table)
    subtype_influence = pd.concat(subtype_rows, ignore_index=True)
    _write_table(results_dir, "subtype_influence", subtype_influence, "Subtype Influence Diagnostics")

    _write_figures(
        figures_dir,
        bootstrap,
        threshold_sweep,
        train_size,
        severity_summary,
        examples,
        disagreement,
        projection,
        runtime,
        subtype_influence,
    )
    _write_interpretation_docs(results_dir, figures_dir, bootstrap, threshold_sweep, train_size, severity_summary, runtime)
    print(f"Wrote robustness tables to {results_dir}")
    print(f"Wrote robustness figures to {figures_dir}")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _load_primary_scores(repo: Path) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for scheme, spec in SCHEMES.items():
        path = repo / str(spec["score_csv"])
        if not path.exists():
            missing.append(f"{scheme}: {path}")
            continue
        table = pd.read_csv(path)
        for column in ["label", "score", "threshold", "prediction"]:
            if column in table.columns:
                table[column] = pd.to_numeric(table[column], errors="coerce")
        if "ood_subtype" not in table.columns:
            table["ood_subtype"] = np.where(table["label"].astype(int) == 0, "id", "unknown")
        output[scheme] = table
    if missing:
        raise FileNotFoundError("Missing required primary score CSVs:\n" + "\n".join(missing))
    return output


def _write_existing_artifact_inventory(path: Path, repo: Path, available: list[str]) -> None:
    committed_paths = [
        "reports/dissertation_results/multi_scheme_comparison/",
        "reports/dissertation_results/primary_balanced_by_subtype_10k/",
        "reports/dissertation_figures/",
        "docs/experiments/dissertation_results_interpretation.md",
    ]
    generated_paths = sorted(
        str(candidate.relative_to(repo)).replace("\\", "/")
        for candidate in (repo / "reports/generated/dissertation_runs").glob("*/runs/*/evaluation/scores.csv")
    )
    lines = [
        "# Existing Result Inventory",
        "",
        "This PR reuses merged PR #21 result summaries and local generated score artifacts.",
        "",
        "## Committed PR #21 artifacts",
        "",
        *[f"- `{item}`" for item in committed_paths if (repo / item).exists()],
        "",
        "## Score artifacts reused locally",
        "",
        *[f"- `{item}`" for item in generated_paths],
        "",
        "## Schemes available for robustness analysis",
        "",
        *[f"- `{scheme}`" for scheme in available],
        "",
        "No completed PR #21 experiment is duplicated in the committed outputs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_bootstrap_ci(scores: dict[str, pd.DataFrame], *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scheme in BOOTSTRAP_SCHEMES:
        table = scores[scheme]
        threshold = _deployment_threshold(table)
        result = bootstrap_metric_ci(
            table["label"].astype(int).to_numpy(),
            table["score"].astype(float).to_numpy(),
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        result.insert(0, "scheme", scheme)
        result.insert(1, "scheme_label", SCHEMES[scheme]["label"])
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _score_validation_id(repo: Path, *, device: str) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for scheme in THRESHOLD_SCHEMES:
        spec = SCHEMES[scheme]
        config = _read_yaml(repo / str(spec["config"]))
        checkpoint = repo / str(spec["checkpoint"])
        detector = _load_detector(scheme, config, checkpoint, device=device)
        val_manifest = config.get("data", {}).get("val_manifest")
        if not val_manifest:
            raise ValueError(f"{scheme} config does not define data.val_manifest")
        scores[scheme] = _score_manifest(detector, config, val_manifest)
    return scores


def _build_threshold_policy_sweep(scores: dict[str, pd.DataFrame], val_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scheme in THRESHOLD_SCHEMES:
        table = threshold_policy_sweep(
            scores[scheme],
            val_id_scores=val_scores[scheme],
            policies=THRESHOLD_POLICIES,
        )
        table.insert(0, "scheme", scheme)
        table.insert(1, "scheme_label", SCHEMES[scheme]["label"])
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _build_train_size_sensitivity(repo: Path, *, device: str, seed: int) -> pd.DataFrame:
    root_dir = repo / "data"
    train_manifest = repo / "datasets/dissertation_v1/manifests/train_id.csv"
    val_manifest = repo / "datasets/dissertation_v1/manifests/val_id.csv"
    test_id_manifest = repo / "datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv"
    test_ood_manifest = repo / "datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv"

    train_df = pd.read_csv(train_manifest)
    test_df = pd.concat([pd.read_csv(test_id_manifest), pd.read_csv(test_ood_manifest)], ignore_index=True)
    test_labels = test_df["label"].astype(int).to_numpy()

    image_train = _image_stat_features(train_manifest, root_dir)
    image_val = _image_stat_features(val_manifest, root_dir)
    image_test = _image_stat_features_for_combined(test_id_manifest, test_ood_manifest, root_dir)

    feature_extractor = FeatureDistanceDetector(
        mode="knn",
        backbone="resnet50",
        layers=("layer3",),
        pretrained=True,
        device=device,
    )
    train_features = _feature_distance_features(feature_extractor, train_manifest, root_dir)
    val_features = _feature_distance_features(feature_extractor, val_manifest, root_dir)
    test_features = _feature_distance_features_for_combined(feature_extractor, test_id_manifest, test_ood_manifest, root_dir)

    rows: list[dict[str, Any]] = []
    for size in TRAIN_SIZES:
        indices = _deterministic_positions(len(train_df), size=size, seed=seed)
        rows.append(
            _train_size_row(
                "image_statistics",
                size,
                image_train[indices],
                image_val,
                image_test,
                test_labels,
                score_mode="image_statistics",
            )
        )
        rows.append(
            _train_size_row(
                "global_feature_knn",
                size,
                train_features[indices],
                val_features,
                test_features,
                test_labels,
                score_mode="knn",
            )
        )
        rows.append(
            _train_size_row(
                "mahalanobis_feature",
                size,
                train_features[indices],
                val_features,
                test_features,
                test_labels,
                score_mode="mahalanobis",
            )
        )
    return pd.DataFrame(rows)


def _train_size_row(
    scheme: str,
    train_size: int,
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    labels: np.ndarray,
    *,
    score_mode: str,
) -> dict[str, Any]:
    if score_mode == "image_statistics":
        mean = train_features.mean(axis=0)
        scale = train_features.std(axis=0)
        scale[scale < 1e-6] = 1.0
        val_scores = np.sqrt(np.mean(((val_features - mean) / scale) ** 2, axis=1))
        test_scores = np.sqrt(np.mean(((test_features - mean) / scale) ** 2, axis=1))
    elif score_mode == "knn":
        val_scores = _knn_scores(val_features.astype(np.float32), train_features.astype(np.float32), 1)
        test_scores = _knn_scores(test_features.astype(np.float32), train_features.astype(np.float32), 1)
    elif score_mode == "mahalanobis":
        mean = train_features.mean(axis=0).astype(np.float32)
        inv_cov = _regularized_inverse_covariance(train_features.astype(np.float32), 1e-3)
        val_scores = _mahalanobis_scores(val_features.astype(np.float32), mean, inv_cov)
        test_scores = _mahalanobis_scores(test_features.astype(np.float32), mean, inv_cov)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    threshold = float(np.quantile(val_scores, 0.95))
    metrics = metric_snapshot(labels, test_scores, threshold=threshold)
    return {
        "scheme": scheme,
        "scheme_label": SCHEMES[scheme]["label"],
        "train_size": train_size,
        "threshold": threshold,
        **metrics,
    }


def _build_artifact_severity_stress(
    repo: Path,
    scores: dict[str, pd.DataFrame],
    *,
    sample_size: int,
    seed: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Image.Image]]:
    id_manifest = pd.read_csv(repo / "datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(id_manifest), size=min(sample_size, len(id_manifest)), replace=False))
    selected_rows = id_manifest.iloc[selected].reset_index(drop=True)
    source_images = [_load_image(repo / "data", row["image_path"]) for _, row in selected_rows.iterrows()]
    artifacts = _generate_artifacts(source_images)
    detectors = {
        scheme: _load_detector(
            scheme,
            _read_yaml(repo / str(SCHEMES[scheme]["config"])),
            repo / str(SCHEMES[scheme]["checkpoint"]),
            device=device,
        )
        for scheme in SEVERITY_SCHEMES
    }

    rows: list[dict[str, Any]] = []
    examples: dict[str, Image.Image] = {}
    for artifact_key, payload in artifacts.items():
        artifact_type, severity_label, severity_value, pil_images = payload
        examples.setdefault(f"{artifact_type}_{severity_label}", pil_images[0])
        for scheme, detector in detectors.items():
            config = _read_yaml(repo / str(SCHEMES[scheme]["config"]))
            artifact_scores = _score_pil_images(detector, config, pil_images)
            deployment_threshold = _deployment_threshold(scores[scheme])
            research_threshold = _research_threshold(scores[scheme])
            for index, score in enumerate(artifact_scores):
                rows.append(
                    {
                        "scheme": scheme,
                        "scheme_label": SCHEMES[scheme]["label"],
                        "artifact_type": artifact_type,
                        "severity_label": severity_label,
                        "severity_value": severity_value,
                        "sample_index": index,
                        "score": float(score),
                        "deployment_threshold": deployment_threshold,
                        "research_threshold": research_threshold,
                    }
                )
    raw = pd.DataFrame(rows)
    summary = artifact_severity_summary(raw)
    summary = summary.merge(raw[["scheme", "scheme_label"]].drop_duplicates(), on="scheme", how="left")
    return raw, summary, examples


def _build_method_disagreement(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected = {scheme: scores[scheme] for scheme in DISAGREEMENT_SCHEMES}
    thresholds = {scheme: _deployment_threshold(scores[scheme]) for scheme in DISAGREEMENT_SCHEMES}
    table = method_disagreement_table(
        selected,
        thresholds=thresholds,
        primary_method="mahalanobis_feature",
        secondary_method="patchcore_l3",
    )
    table["borderline_margin_mahalanobis"] = (
        table["score_mahalanobis_feature"] - table["threshold_mahalanobis_feature"]
    ).abs()
    table["borderline_margin_patchcore_l3"] = (table["score_patchcore_l3"] - table["threshold_patchcore_l3"]).abs()
    return table.sort_values(["case_type", "borderline_margin_mahalanobis"], kind="stable").reset_index(drop=True)


def _build_feature_space_projection(repo: Path, maha_scores: pd.DataFrame, *, device: str) -> pd.DataFrame:
    config = _read_yaml(repo / str(SCHEMES["mahalanobis_feature"]["config"]))
    detector = FeatureDistanceDetector.load(repo / str(SCHEMES["mahalanobis_feature"]["checkpoint"]), device=device)
    id_manifest = repo / "datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv"
    ood_manifest = repo / "datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv"
    features = _feature_distance_features_for_combined(detector, id_manifest, ood_manifest, repo / "data")
    metadata = maha_scores[["image_path", "label", "ood_type", "ood_subtype", "score"]].copy()
    projection = pca_projection(features, metadata)
    projection["mahalanobis_percentile"] = projection["score"].rank(pct=True)
    projection["feature_config"] = "resnet50_layer3_global_pool"
    _ = config
    return projection


def _build_runtime_summary(repo: Path, scores: dict[str, pd.DataFrame], *, device: str) -> pd.DataFrame:
    sample_manifest = repo / "datasets/dissertation_v1/manifests/test_ood_smoke.csv"
    rows: list[dict[str, Any]] = []
    for scheme, spec in SCHEMES.items():
        if scheme == "patchcore_l2_l3":
            # Included in artifact-size/runtime table, but avoid duplicate slow scoring in smoke timing.
            timing_scheme = "patchcore_l3"
        else:
            timing_scheme = scheme
        config = _read_yaml(repo / str(SCHEMES[timing_scheme]["config"]))
        checkpoint = repo / str(spec["checkpoint"])
        timing_checkpoint = repo / str(SCHEMES[timing_scheme]["checkpoint"])
        detector = _load_detector(timing_scheme, config, timing_checkpoint, device=device)
        start = time.perf_counter()
        sample_scores = _score_manifest(detector, config, sample_manifest)
        elapsed = time.perf_counter() - start
        primary = metric_snapshot(
            scores[scheme]["label"].to_numpy(),
            scores[scheme]["score"].to_numpy(),
            threshold=_deployment_threshold(scores[scheme]),
        )
        rows.append(
            {
                "scheme": scheme,
                "scheme_label": spec["label"],
                "artifact_size_mb": checkpoint.stat().st_size / (1024 * 1024) if checkpoint.exists() else np.nan,
                "scoring_time_seconds": elapsed,
                "scoring_images": len(sample_scores),
                "scoring_ms_per_image": 1000 * elapsed / max(len(sample_scores), 1),
                "fit_time_seconds": np.nan,
                "fit_time_note": "not remeasured; existing PR #21 training artifacts reused",
                "localization_support": spec["localization"],
                "implementation_complexity": spec["complexity"],
                "primary_auroc": primary["auroc"],
                "primary_fpr_at_95_tpr": primary["fpr_at_95_tpr"],
            }
        )
    return pd.DataFrame(rows)


def _write_figures(
    figures_dir: Path,
    bootstrap: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    train_size: pd.DataFrame,
    severity: pd.DataFrame,
    examples: dict[str, Image.Image],
    disagreement: pd.DataFrame,
    projection: pd.DataFrame,
    runtime: pd.DataFrame,
    subtype_influence: pd.DataFrame,
) -> None:
    _plot_bootstrap_metric(bootstrap, "auroc", figures_dir / "figure_bootstrap_ci_main_metrics.png")
    _plot_bootstrap_metric(bootstrap, "fpr_at_95_tpr", figures_dir / "figure_bootstrap_ci_fpr95.png")
    _plot_train_size(train_size, "auroc", figures_dir / "figure_train_size_sensitivity_auroc.png")
    _plot_train_size(train_size, "fpr_at_95_tpr", figures_dir / "figure_train_size_sensitivity_fpr95.png")
    _plot_threshold_tradeoff(threshold_sweep, figures_dir / "figure_threshold_policy_tradeoff.png")
    _plot_id_rejection_vs_recall(threshold_sweep, figures_dir / "figure_id_rejection_vs_ood_recall.png")
    _plot_threshold_by_type(threshold_sweep, figures_dir / "figure_threshold_policy_by_ood_type.png")
    _plot_artifact_severity(severity, "mean_score", figures_dir / "figure_artifact_severity_scores.png")
    _plot_artifact_severity(severity, "deployment_reject_rate", figures_dir / "figure_artifact_severity_reject_rate.png")
    _plot_artifact_examples(examples, figures_dir / "figure_artifact_severity_examples.png")
    _plot_disagreement_matrix(disagreement, figures_dir / "figure_method_disagreement_matrix.png")
    _plot_case_examples(disagreement, figures_dir / "figure_method_disagreement_examples.png", title="Method disagreement examples")
    _plot_case_examples(disagreement, figures_dir / "figure_failure_case_grid.png", title="Failure case grid")
    _plot_case_examples(
        disagreement[disagreement["case_type"] == "primary_false_positive_id"],
        figures_dir / "figure_false_positive_id_examples.png",
        title="ID false positives",
    )
    _plot_case_examples(
        disagreement[disagreement["case_type"].str.contains("false_negative|secondary_correct", regex=True)],
        figures_dir / "figure_false_negative_ood_examples.png",
        title="OOD false negatives and disagreements",
    )
    _plot_pca(projection, "ood_type", figures_dir / "figure_feature_space_pca_by_ood_type.png")
    _plot_pca(projection, "ood_subtype", figures_dir / "figure_feature_space_pca_by_subtype.png")
    _plot_pca_score(projection, figures_dir / "figure_feature_space_pca_by_score.png")
    _plot_runtime_vs_performance(runtime, figures_dir / "figure_runtime_vs_performance.png")
    _plot_method_tradeoff_table(runtime, figures_dir / "figure_method_tradeoff_table.png")
    _plot_subtype_influence(subtype_influence, figures_dir / "figure_subtype_influence.png")


def _write_interpretation_docs(
    results_dir: Path,
    figures_dir: Path,
    bootstrap: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    train_size: pd.DataFrame,
    severity: pd.DataFrame,
    runtime: pd.DataFrame,
) -> None:
    best_bootstrap = (
        bootstrap[bootstrap["metric"] == "auroc"].sort_values("estimate", ascending=False).iloc[0]["scheme_label"]
    )
    best_700 = train_size[train_size["train_size"] == train_size["train_size"].max()].sort_values(
        "auroc",
        ascending=False,
    ).iloc[0]
    safest = threshold_sweep[threshold_sweep["policy_kind"] == "deployment"].sort_values(
        ["ood_recall", "id_false_rejection_rate"],
        ascending=[False, True],
    ).iloc[0]
    top_severity = severity.sort_values("severity_score_spearman", ascending=False).iloc[0]
    runtime_best = runtime.sort_values("scoring_ms_per_image").iloc[0]
    interpretation = f"""# Dissertation Robustness Results Interpretation

## Summary of Robustness Findings

The robustness analyses extend the merged multi-scheme comparison without changing the core task. Training remains ID-only, OOD examples remain evaluation-only, and the system remains a binary FAF OOD gatekeeper rather than a disease classifier.

## Bootstrap CI Interpretation

Bootstrap confidence intervals were computed from sample-level primary score outputs. The top AUROC estimate remains `{best_bootstrap}`, supporting the PR #21 finding that feature-distance methods are strong on the current synthetic-backed evaluation.

## Training-Size Sensitivity Interpretation

At the largest tested ID subset ({int(best_700['train_size'])} images), the strongest train-size sensitivity row is `{best_700['scheme_label']}` with AUROC {best_700['auroc']:.4f}. These curves show whether performance saturates before all 700 ID training images are used and provide motivation for future real FAF collection.

## Threshold-Safety Interpretation

Threshold sweeps distinguish research thresholds from deployment-style ID validation quantiles. The strongest deployment-style row by OOD recall in this run is `{safest['scheme_label']}` / `{safest['policy']}` with OOD recall {safest['ood_recall']:.4f} and ID false rejection rate {safest['id_false_rejection_rate']:.4f}. This is threshold analysis, not deployment approval.

## Artifact-Severity Stress-Test Interpretation

Artifact severity tests were generated from ID fallback images for evaluation only. The highest monotonic severity-score relationship in the compact summary is `{top_severity['scheme']}` on `{top_severity['artifact_type']}` with Spearman {top_severity['severity_score_spearman']:.4f}.

## Method Disagreement and Failure Cases

The disagreement tables compare Mahalanobis, PatchCore L3, autoencoder, and image statistics decisions at their selected thresholds. These cases support discussion of why a quantitatively strong global/statistical feature method and a localizable PatchCore method can fail on different images.

## Feature-Space Visualization Interpretation

PCA figures use the same feature family as the Mahalanobis baseline. They are intended to show whether semantic outliers, modality shifts, and sensory artifacts form separable feature-space structure and whether that separation explains Mahalanobis performance.

## Runtime and Deployment Practicality

The fastest smoke scoring row is `{runtime_best['scheme_label']}` at {runtime_best['scoring_ms_per_image']:.2f} ms/image on this local CPU run. Runtime figures should be interpreted as local engineering measurements, not hardware-independent deployment guarantees.

## Limitations

`test_id_synthetic_fallback.csv` is synthetic fallback ID, not real clinical FAF validation. OOD stress tests are not clinical prevalence estimates. Research thresholds use OOD labels for analysis only. Real clinical FAF validation, calibration, and prospective workflow testing remain future work.

## Suggested Dissertation Paragraphs

The follow-up robustness experiments show that the ranking observed in the main comparison is not solely a single point estimate. Bootstrap intervals, train-size curves, threshold sweeps, and severity stress tests provide complementary evidence about uncertainty, calibration sensitivity, and failure modes. The analysis also clarifies a practical distinction: Mahalanobis feature distance is quantitatively strong and compact, while PatchCore L3 remains useful for localizing and discussing image-level artifact evidence.
"""
    (Path("docs/experiments/dissertation_robustness_results_interpretation.md")).write_text(
        interpretation,
        encoding="utf-8",
    )

    captions = _robustness_caption_rows()
    caption_lines = ["# Robustness Figure Caption Suggestions", ""]
    guide_lines = [
        "# Robustness Figure Selection Guide",
        "",
        "| Figure | Purpose | Section | Status |",
        "| --- | --- | --- | --- |",
    ]
    index_lines = [
        "# Robustness Figure Index",
        "",
        "| File | Recommended section | Suggested caption |",
        "| --- | --- | --- |",
    ]
    for row in captions:
        figure_path = figures_dir / row["filename"]
        if not figure_path.exists():
            continue
        caption_lines.extend([f"## {row['filename']}", "", row["caption"], ""])
        guide_lines.append(f"| {row['filename']} | {row['purpose']} | {row['section']} | {row['status']} |")
        index_lines.append(f"| {row['filename']} | {row['section']} | {row['caption']} |")
    (figures_dir / "caption_suggestions.md").write_text("\n".join(caption_lines), encoding="utf-8")
    (figures_dir / "figure_selection_guide.md").write_text("\n".join(guide_lines) + "\n", encoding="utf-8")
    (figures_dir / "figure_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    _ = results_dir


def _robustness_caption_rows() -> list[dict[str, str]]:
    return [
        _caption("figure_bootstrap_ci_main_metrics.png", "Bootstrap AUROC uncertainty", "Results", "must_include"),
        _caption("figure_bootstrap_ci_fpr95.png", "Bootstrap FPR@95%TPR uncertainty", "Safety", "must_include"),
        _caption("figure_train_size_sensitivity_auroc.png", "Train-size AUROC sensitivity", "Results", "must_include"),
        _caption("figure_train_size_sensitivity_fpr95.png", "Train-size FPR@95 sensitivity", "Safety", "optional"),
        _caption("figure_threshold_policy_tradeoff.png", "Threshold policy trade-off", "Safety", "must_include"),
        _caption("figure_id_rejection_vs_ood_recall.png", "ID rejection versus OOD recall", "Safety", "must_include"),
        _caption("figure_threshold_policy_by_ood_type.png", "OOD-type recall by threshold policy", "Safety", "optional"),
        _caption("figure_artifact_severity_scores.png", "Artifact severity score response", "Stress tests", "must_include"),
        _caption("figure_artifact_severity_reject_rate.png", "Artifact severity reject rate", "Stress tests", "must_include"),
        _caption("figure_artifact_severity_examples.png", "Visual examples of generated artifact severities", "Stress tests", "optional"),
        _caption("figure_method_disagreement_matrix.png", "Method disagreement case counts", "Failure analysis", "must_include"),
        _caption("figure_method_disagreement_examples.png", "Representative method disagreement examples", "Failure analysis", "must_include"),
        _caption("figure_failure_case_grid.png", "Representative failure cases", "Failure analysis", "optional"),
        _caption("figure_false_positive_id_examples.png", "ID false-positive examples", "Failure analysis", "optional"),
        _caption("figure_false_negative_ood_examples.png", "OOD false-negative examples", "Failure analysis", "optional"),
        _caption("figure_feature_space_pca_by_ood_type.png", "Feature-space PCA by OOD type", "Feature interpretation", "must_include"),
        _caption("figure_feature_space_pca_by_subtype.png", "Feature-space PCA by subtype", "Feature interpretation", "optional"),
        _caption("figure_feature_space_pca_by_score.png", "Feature-space PCA by Mahalanobis score", "Feature interpretation", "must_include"),
        _caption("figure_runtime_vs_performance.png", "Runtime versus performance", "Deployment practicality", "must_include"),
        _caption("figure_method_tradeoff_table.png", "Method trade-off table", "Deployment practicality", "optional"),
        _caption("figure_subtype_influence.png", "Subtype influence diagnostics", "Results", "optional"),
    ]


def _caption(filename: str, purpose: str, section: str, status: str) -> dict[str, str]:
    return {
        "filename": filename,
        "purpose": purpose,
        "section": section,
        "status": status,
        "caption": (
            f"{purpose}. Generated for the unsupervised FAF OOD gatekeeper; training is ID-only and OOD "
            "labels are used only for evaluation grouping. The ID test set is synthetic fallback, not real clinical FAF validation."
        ),
    }


def _write_table(results_dir: Path, stem: str, table: pd.DataFrame, title: str) -> None:
    csv_path = results_dir / f"{stem}.csv"
    md_path = results_dir / f"{stem}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    md_path.write_text(f"# {title}\n\n{dataframe_to_markdown(table)}\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def _load_detector(scheme: str, config: dict[str, Any], checkpoint: Path, *, device: str):
    kind = str(SCHEMES[scheme]["kind"])
    if kind == "image_statistics":
        return ImageStatisticsDetector.load(checkpoint)
    if kind in {"global_feature_knn", "mahalanobis_feature"}:
        return FeatureDistanceDetector.load(checkpoint, device=device)
    if kind == "autoencoder":
        model = load_autoencoder_checkpoint(checkpoint, map_location=torch.device(device))
        return AutoencoderScoreDetector(model, device=torch.device(device))
    if kind == "patchcore":
        detector = PatchCoreDetector.load(checkpoint)
        detector.config.device = device
        detector.device = torch.device(device)
        if detector.feature_extractor is not None:
            detector.feature_extractor.to(detector.device)
            detector.feature_extractor.eval()
        return detector
    raise ValueError(f"Unsupported detector kind: {kind}")


def _score_manifest(detector: Any, config: dict[str, Any], manifest_path: str | Path) -> np.ndarray:
    data_cfg = config.get("data", {})
    transform = build_transforms(
        image_size=int(data_cfg.get("image_size", 224)),
        grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", True)),
        normalize=data_cfg.get("normalize", "imagenet"),
    )
    dataset = ManifestImageDataset(
        str(manifest_path),
        root_dir=data_cfg.get("root_dir"),
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_manifest_batch,
    )
    return np.asarray(detector.predict_scores(loader), dtype=float)


def _score_pil_images(detector: Any, config: dict[str, Any], images: list[Image.Image]) -> np.ndarray:
    data_cfg = config.get("data", {})
    transform = build_transforms(
        image_size=int(data_cfg.get("image_size", 224)),
        grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", True)),
        normalize=data_cfg.get("normalize", "imagenet"),
    )
    tensors = torch.stack([transform(image.convert("RGB")) for image in images])
    labels = torch.zeros((len(images),), dtype=torch.long)
    loader = DataLoader(TensorDataset(tensors, labels), batch_size=int(data_cfg.get("batch_size", 16)), shuffle=False)
    return np.asarray(detector.predict_scores(loader), dtype=float)


def _deployment_threshold(scores: pd.DataFrame) -> float:
    if "threshold" in scores.columns:
        values = pd.to_numeric(scores["threshold"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    id_scores = scores.loc[scores["label"].astype(int) == 0, "score"].astype(float)
    return float(np.quantile(id_scores, 0.95))


def _research_threshold(scores: pd.DataFrame) -> float:
    labels = scores["label"].astype(int).to_numpy()
    values = scores["score"].astype(float).to_numpy()
    from retinal_ood.evaluation.robustness import _threshold_at_tpr

    return _threshold_at_tpr(labels, values, 0.95)


def _deterministic_positions(count: int, *, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=min(size, count), replace=False))


def _image_stat_features(manifest: Path, root_dir: Path) -> np.ndarray:
    transform = build_transforms(image_size=224, grayscale_to_rgb=False, normalize="minmax")
    dataset = ManifestImageDataset(manifest, root_dir=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_manifest_batch)
    batches = [extract_image_statistics_features(batch[0]) for batch in loader]
    return np.concatenate(batches, axis=0)


def _image_stat_features_for_combined(id_manifest: Path, ood_manifest: Path, root_dir: Path) -> np.ndarray:
    return np.concatenate([_image_stat_features(id_manifest, root_dir), _image_stat_features(ood_manifest, root_dir)])


def _feature_distance_features(detector: FeatureDistanceDetector, manifest: Path, root_dir: Path) -> np.ndarray:
    transform = build_transforms(image_size=224, grayscale_to_rgb=True, normalize="imagenet")
    dataset = ManifestImageDataset(manifest, root_dir=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_manifest_batch)
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for images, *_ in loader:
            batches.append(detector._extract_features(images.to(detector.device, dtype=torch.float32)))
    return np.concatenate(batches, axis=0)


def _feature_distance_features_for_combined(
    detector: FeatureDistanceDetector,
    id_manifest: Path,
    ood_manifest: Path,
    root_dir: Path,
) -> np.ndarray:
    return np.concatenate(
        [
            _feature_distance_features(detector, id_manifest, root_dir),
            _feature_distance_features(detector, ood_manifest, root_dir),
        ]
    )


def _load_image(root_dir: Path, image_path: str) -> Image.Image:
    path = root_dir / image_path
    return Image.open(path).convert("RGB")


def _generate_artifacts(images: list[Image.Image]) -> dict[str, tuple[str, str, float, list[Image.Image]]]:
    artifacts: dict[str, tuple[str, str, float, list[Image.Image]]] = {}
    for opacity in [0.10, 0.25, 0.50, 0.75]:
        label = f"opacity_{opacity:.2f}".replace(".", "_")
        artifacts[f"text_watermark_{label}"] = (
            "text_watermark",
            label,
            opacity,
            [_text_watermark(image, opacity) for image in images],
        )
    for sigma in [5, 10, 20, 40]:
        artifacts[f"gaussian_noise_sigma_{sigma}"] = (
            "gaussian_noise",
            f"sigma_{sigma}",
            float(sigma),
            [_gaussian_noise(image, sigma) for image in images],
        )
    for quality in [90, 70, 50, 30, 10]:
        artifacts[f"jpeg_quality_{quality}"] = (
            "jpeg_compression",
            f"quality_{quality}",
            float(100 - quality),
            [_jpeg_compress(image, quality) for image in images],
        )
    for kernel in [3, 5, 9, 15]:
        artifacts[f"blur_kernel_{kernel}"] = (
            "blur_artifact",
            f"kernel_{kernel}",
            float(kernel),
            [image.filter(ImageFilter.GaussianBlur(radius=kernel / 3.0)) for image in images],
        )
    for fraction in [0.02, 0.05, 0.10, 0.20]:
        label = f"crop_{fraction:.2f}".replace(".", "_")
        artifacts[f"border_crop_{label}"] = (
            "border_crop",
            label,
            fraction,
            [_border_crop(image, fraction) for image in images],
        )
    for width in [2, 5, 9, 15]:
        artifacts[f"rectangle_annotation_width_{width}"] = (
            "rectangle_annotation",
            f"width_{width}",
            float(width),
            [_rectangle_annotation(image, width) for image in images],
        )
    return artifacts


def _text_watermark(image: Image.Image, opacity: float) -> Image.Image:
    output = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (255, 255, 255, int(255 * opacity))
    draw.text((output.width * 0.18, output.height * 0.42), "OOD", fill=fill)
    return Image.alpha_composite(output, overlay).convert("RGB")


def _gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(1234 + int(sigma))
    noisy = np.clip(arr + rng.normal(0, sigma, size=arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _border_crop(image: Image.Image, fraction: float) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width = max(1, int(min(output.size) * fraction))
    draw.rectangle([0, 0, output.width, width], fill=(0, 0, 0))
    draw.rectangle([0, output.height - width, output.width, output.height], fill=(0, 0, 0))
    draw.rectangle([0, 0, width, output.height], fill=(0, 0, 0))
    draw.rectangle([output.width - width, 0, output.width, output.height], fill=(0, 0, 0))
    return output


def _rectangle_annotation(image: Image.Image, width: int) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    margin = max(5, output.width // 8)
    draw.rectangle(
        [margin, margin, output.width - margin, output.height - margin],
        outline=(255, 40, 40),
        width=width,
    )
    return output


def _plot_bootstrap_metric(table: pd.DataFrame, metric: str, path: Path) -> None:
    rows = table[table["metric"] == metric].copy()
    rows = rows.sort_values("estimate", ascending=metric == "fpr_at_95_tpr")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(rows))
    ax.errorbar(
        rows["estimate"],
        y,
        xerr=[rows["estimate"] - rows["ci_low"], rows["ci_high"] - rows["estimate"]],
        fmt="o",
        color=PALETTE["blue"],
        ecolor=PALETTE["gray"],
        capsize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(rows["scheme_label"])
    ax.set_xlabel(metric)
    ax.set_title(f"Bootstrap confidence intervals: {metric}")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, path)


def _plot_train_size(table: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for scheme, rows in table.groupby("scheme", sort=False):
        ax.plot(rows["train_size"], rows[metric], marker="o", label=SCHEMES[scheme]["label"])
    ax.set_xlabel("ID training images")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Train-size sensitivity: {metric}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    _save(fig, path)


def _plot_threshold_tradeoff(table: pd.DataFrame, path: Path) -> None:
    deployment = table[table["policy_kind"] == "deployment"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for scheme, rows in deployment.groupby("scheme", sort=False):
        ax.plot(rows["id_false_rejection_rate"], rows["ood_recall"], marker="o", label=SCHEMES[scheme]["label"])
    ax.set_xlabel("ID false rejection rate")
    ax.set_ylabel("OOD recall")
    ax.set_title("Threshold policy trade-off")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def _plot_id_rejection_vs_recall(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {"research": PALETTE["gold"], "deployment": PALETTE["blue"]}
    for kind, rows in table.groupby("policy_kind"):
        ax.scatter(rows["id_false_rejection_rate"], rows["ood_recall"], label=kind, color=colors[kind], alpha=0.8)
    ax.set_xlabel("ID false rejection rate")
    ax.set_ylabel("OOD recall")
    ax.set_title("ID rejection versus OOD recall")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    _save(fig, path)


def _plot_threshold_by_type(table: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in table.iterrows():
        recalls = json.loads(row["per_ood_type_recall"])
        for ood_type, recall in recalls.items():
            rows.append({"scheme_label": row["scheme_label"], "policy": row["policy"], "ood_type": ood_type, "recall": recall})
    long = pd.DataFrame(rows)
    subset = long[long["policy"].isin(["research_95_tpr", "val_id_quantile_95", "val_id_quantile_99"])]
    pivot = subset.pivot_table(index=["scheme_label", "policy"], columns="ood_type", values="recall", aggfunc="first")
    _heatmap(pivot, path, "OOD-type recall by threshold policy", "Recall")


def _plot_artifact_severity(table: pd.DataFrame, value_column: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), sharey=value_column.endswith("rate"))
    for axis, (artifact, rows) in zip(axes.ravel(), table.groupby("artifact_type", sort=True)):
        for scheme, scheme_rows in rows.groupby("scheme", sort=False):
            axis.plot(
                scheme_rows["severity_value"],
                scheme_rows[value_column],
                marker="o",
                label=SCHEMES.get(scheme, {}).get("label", scheme),
            )
        axis.set_title(str(artifact).replace("_", " "))
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(table["artifact_type"].unique()) :]:
        axis.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=7)
    fig.suptitle(value_column.replace("_", " ").title())
    _save(fig, path)


def _plot_artifact_examples(examples: dict[str, Image.Image], path: Path) -> None:
    selected = list(examples.items())[:12]
    fig, axes = plt.subplots(3, 4, figsize=(9.2, 7.0))
    for axis, (label, image) in zip(axes.ravel(), selected):
        axis.imshow(image)
        axis.set_title(_display_artifact_label(label), fontsize=8)
        axis.axis("off")
    for axis in axes.ravel()[len(selected) :]:
        axis.axis("off")
    fig.suptitle("Artifact severity stress-test examples")
    _save(fig, path)


def _display_artifact_label(label: str) -> str:
    text = label.replace("_", " ")
    return re.sub(r"\b0 (\d{2})\b", r"0.\1", text)


def _plot_disagreement_matrix(table: pd.DataFrame, path: Path) -> None:
    counts = table["case_type"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh([str(item).replace("_", " ") for item in counts.index], counts.values, color=PALETTE["blue"])
    ax.set_xlabel("Samples")
    ax.set_title("Method disagreement case counts")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, path)


def _plot_case_examples(table: pd.DataFrame, path: Path, *, title: str) -> None:
    rows = table.head(12)
    fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.2))
    for axis, (_, row) in zip(axes.ravel(), rows.iterrows()):
        image_path = Path("data") / str(row["image_path"])
        if image_path.exists():
            axis.imshow(Image.open(image_path).convert("RGB"))
        else:
            axis.text(0.5, 0.5, "image unavailable", ha="center", va="center")
        axis.set_title(str(row["case_type"]).replace("_", " "), fontsize=7)
        axis.axis("off")
    for axis in axes.ravel()[len(rows) :]:
        axis.axis("off")
    fig.suptitle(title)
    _save(fig, path)


def _plot_pca(table: pd.DataFrame, color_column: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for value, rows in table.groupby(color_column, sort=True):
        ax.scatter(rows["pc1"], rows["pc2"], s=14, alpha=0.7, label=str(value).replace("_", " "))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Feature-space PCA by {color_column.replace('_', ' ')}")
    ax.legend(frameon=False, fontsize=7, markerscale=1.5)
    ax.grid(alpha=0.18)
    _save(fig, path)


def _plot_pca_score(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    scatter = ax.scatter(table["pc1"], table["pc2"], c=table["score"], cmap="viridis", s=14, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Mahalanobis score")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Feature-space PCA by anomaly score")
    ax.grid(alpha=0.18)
    _save(fig, path)


def _plot_runtime_vs_performance(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(table["scoring_ms_per_image"], table["primary_auroc"], s=80, color=PALETTE["blue"])
    for _, row in table.iterrows():
        ax.text(row["scoring_ms_per_image"], row["primary_auroc"], row["scheme_label"], fontsize=8)
    ax.set_xlabel("Scoring ms/image (local CPU smoke sample)")
    ax.set_ylabel("Primary AUROC")
    ax.set_title("Runtime versus performance")
    ax.grid(alpha=0.22)
    _save(fig, path)


def _plot_method_tradeoff_table(table: pd.DataFrame, path: Path) -> None:
    display = table[
        ["scheme_label", "artifact_size_mb", "scoring_ms_per_image", "primary_auroc", "localization_support", "implementation_complexity"]
    ].copy()
    for column in ["artifact_size_mb", "scoring_ms_per_image", "primary_auroc"]:
        display[column] = display[column].map(lambda value: f"{float(value):.3f}" if pd.notna(value) else "")
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.axis("off")
    tbl = ax.table(cellText=display.values.tolist(), colLabels=display.columns.tolist(), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    ax.set_title("Method trade-off table")
    _save(fig, path)


def _plot_subtype_influence(table: pd.DataFrame, path: Path) -> None:
    only = table[table["analysis"] == "only_subtype"]
    pivot = only.pivot_table(index="scheme_label", columns="ood_subtype", values="auroc", aggfunc="first")
    _heatmap(pivot, path, "Subtype-only AUROC influence", "AUROC")


def _heatmap(pivot: pd.DataFrame, path: Path, title: str, colorbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, len(pivot.columns) * 0.7), max(4.8, len(pivot) * 0.4)))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(item).replace("_", " ") for item in pivot.columns], rotation=28, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(item) for item in pivot.index], fontsize=8)
    ax.set_title(title)
    plt.colorbar(image, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
