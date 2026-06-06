#!/usr/bin/env python
"""Run dissertation-ready OOD experiment grids without storing data artifacts.

The runner is intentionally a thin orchestration layer. It writes per-experiment
resolved configs, calls the existing training/evaluation CLIs, validates manifests,
and aggregates metrics into compact report tables. It does not train new model
types, save heatmaps, or commit any run outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.manifest_audit import audit_manifests
from retinal_ood.evaluation.metrics import compute_ood_metrics
from retinal_ood.evaluation.report_index import generate_report_index
from retinal_ood.evaluation.report_tables import dataframe_to_markdown
from retinal_ood.utils.io import read_yaml


SUMMARY_COLUMNS = [
    "experiment",
    "model",
    "backbone",
    "layers",
    "auroc",
    "auprc",
    "fpr_at_95_tpr",
    "threshold_at_95_tpr",
    "threshold",
    "threshold_source",
    "id_false_rejection_rate",
    "total_count",
    "id_count",
    "ood_count",
    "warnings",
    "metrics_file",
]

PER_OOD_TYPE_COLUMNS = [
    "experiment",
    "model",
    "backbone",
    "layers",
    "ood_type",
    "auroc",
    "auprc",
    "fpr_at_95_tpr",
    "id_count",
    "ood_count",
    "metrics_file",
]

PER_OOD_SUBTYPE_COLUMNS = [
    "experiment",
    "model",
    "backbone",
    "layers",
    "ood_type",
    "ood_subtype",
    "auroc",
    "auprc",
    "fpr_at_95_tpr",
    "id_count",
    "ood_count",
    "metrics_file",
]


class ExperimentSpec(NamedTuple):
    name: str
    kind: str
    config_path: Path
    layers: list[str] | None


class PreparedExperiment(NamedTuple):
    spec: ExperimentSpec
    config: dict[str, Any]
    config_path: Path
    run_dir: Path
    eval_dir: Path
    checkpoint_path: Path
    commands: list[list[str]]
    manifest_paths: list[Path]


class GridRunResult(NamedTuple):
    commands: list[list[str]]
    summary_csv: Path
    summary_markdown: Path
    per_ood_type_csv: Path
    per_ood_subtype_csv: Path | None
    manifest_audit_json: Path


def run_grid(
    *,
    grid_config: str | Path,
    root_dir: str | Path,
    out_dir: str | Path,
    dry_run: bool = False,
    only: Sequence[str] | None = None,
    skip_existing: bool = False,
    max_train_images: int | None = None,
    max_test_images: int | None = None,
    device: str = "auto",
    seed: int | None = None,
) -> GridRunResult:
    """Run or dry-run a configured experiment grid and aggregate report tables."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    _validate_positive_optional("max_train_images", max_train_images)
    _validate_positive_optional("max_test_images", max_test_images)

    grid_path = Path(grid_config).resolve()
    data_root = Path(root_dir).resolve()
    output_root = Path(out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    specs = _load_experiment_specs(grid_path, only=only)
    prepared = [
        _prepare_experiment(
            spec,
            root_dir=data_root,
            out_dir=output_root,
            device=device,
            seed=seed,
            max_train_images=max_train_images,
            max_test_images=max_test_images,
        )
        for spec in specs
    ]

    manifest_audit_path = output_root / "manifest_audit.json"
    audit = _write_manifest_audit(prepared, root_dir=data_root, out_path=manifest_audit_path)
    warnings = "; ".join(audit.get("warnings", []))

    planned_commands: list[list[str]] = []
    for experiment in prepared:
        planned_commands.extend(experiment.commands)
        if dry_run:
            continue
        if skip_existing and (experiment.eval_dir / "metrics.json").exists():
            continue
        for command in experiment.commands:
            subprocess.run(command, check=True)

    summary_csv = output_root / "metrics_summary.csv"
    summary_markdown = output_root / "metrics_summary.md"
    per_ood_type_csv = output_root / "per_ood_type_metrics.csv"
    per_ood_subtype_csv = output_root / "per_ood_subtype_metrics.csv"

    if not dry_run:
        summary, per_type, per_subtype = _build_aggregate_tables(
            prepared,
            out_dir=output_root,
            warnings=warnings,
        )
        _write_csv(summary, summary_csv)
        _write_markdown(summary_markdown, "Experiment Grid Summary", summary)
        _write_csv(per_type, per_ood_type_csv)
        if per_subtype is not None:
            _write_csv(per_subtype, per_ood_subtype_csv)
        else:
            per_ood_subtype_csv = None
        generate_report_index(output_root)

    return GridRunResult(
        commands=planned_commands,
        summary_csv=summary_csv,
        summary_markdown=summary_markdown,
        per_ood_type_csv=per_ood_type_csv,
        per_ood_subtype_csv=per_ood_subtype_csv,
        manifest_audit_json=manifest_audit_path,
    )


def _validate_positive_optional(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")


def _load_experiment_specs(grid_path: Path, *, only: Sequence[str] | None) -> list[ExperimentSpec]:
    raw_grid = read_yaml(grid_path)
    raw_experiments = raw_grid.get("experiments")
    if not raw_experiments:
        raise ValueError(f"Grid config must define at least one experiment: {grid_path}")

    selected = _normalize_only(only)
    specs: list[ExperimentSpec] = []
    if isinstance(raw_experiments, dict):
        iterable = raw_experiments.items()
    elif isinstance(raw_experiments, list):
        iterable = []
        for item in raw_experiments:
            if not isinstance(item, dict) or not item.get("name"):
                raise ValueError("List-style experiments must be mappings with a name field")
            iterable.append((str(item["name"]), item))
    else:
        raise ValueError("experiments must be either a mapping or a list")

    for name, raw_spec in iterable:
        if selected and name not in selected:
            continue
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Experiment {name} must be a mapping")
        config_value = raw_spec.get("config")
        if not config_value:
            raise ValueError(f"Experiment {name} must define config")
        config_path = _resolve_config_path(Path(str(config_value)), grid_path)
        kind = str(raw_spec.get("type", "")).strip().lower()
        layers = raw_spec.get("layers")
        specs.append(
            ExperimentSpec(
                name=str(name),
                kind=kind,
                config_path=config_path,
                layers=[str(layer) for layer in layers] if layers else None,
            )
        )

    if selected:
        found = {spec.name for spec in specs}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"--only requested unknown experiment(s): {missing}")
    if not specs:
        raise ValueError("No experiments selected")
    return specs


def _normalize_only(only: Sequence[str] | None) -> set[str]:
    selected: set[str] = set()
    for value in only or []:
        selected.update(part.strip() for part in str(value).split(",") if part.strip())
    return selected


def _resolve_config_path(raw_path: Path, grid_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path.resolve()
    candidates = [
        (grid_path.parent / raw_path).resolve(),
        (Path.cwd() / raw_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _prepare_experiment(
    spec: ExperimentSpec,
    *,
    root_dir: Path,
    out_dir: Path,
    device: str,
    seed: int | None,
    max_train_images: int | None,
    max_test_images: int | None,
) -> PreparedExperiment:
    config = copy.deepcopy(read_yaml(spec.config_path))
    config.setdefault("project", {})
    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("output", {})
    config.setdefault("evaluation", {})

    config["project"]["run_name"] = spec.name
    if seed is not None:
        config["project"]["seed"] = int(seed)
    config["data"]["root_dir"] = root_dir.as_posix()
    config["model"]["device"] = device
    config["output"]["runs_dir"] = (out_dir / "runs").as_posix()
    config["output"]["save_heatmaps"] = False
    config["evaluation"]["save_heatmaps"] = False
    if spec.layers is not None:
        config["model"]["layers"] = spec.layers

    _resolve_manifest_paths_in_config(config, root_dir=root_dir, config_path=spec.config_path)
    _materialize_subset_manifests(
        config,
        out_dir=out_dir / "manifests" / spec.name,
        max_train_images=max_train_images,
        max_test_images=max_test_images,
    )

    kind = spec.kind or _infer_experiment_kind(config)
    if kind not in {"patchcore", "autoencoder"}:
        raise ValueError(f"Experiment {spec.name} has unsupported type/model: {kind}")
    spec = ExperimentSpec(spec.name, kind, spec.config_path, spec.layers)

    resolved_config_path = out_dir / "configs" / f"{spec.name}.yaml"
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_dir = Path(config["output"]["runs_dir"]) / spec.name
    eval_dir = run_dir / "evaluation"
    checkpoint_path = _checkpoint_path(spec.kind, run_dir)
    commands = _build_commands(spec.kind, resolved_config_path, checkpoint_path)
    return PreparedExperiment(
        spec=spec,
        config=config,
        config_path=resolved_config_path,
        run_dir=run_dir,
        eval_dir=eval_dir,
        checkpoint_path=checkpoint_path,
        commands=commands,
        manifest_paths=_manifest_paths(config),
    )


def _infer_experiment_kind(config: dict[str, Any]) -> str:
    model_name = str(config.get("model", {}).get("name", "")).lower()
    if model_name == "patchcore":
        return "patchcore"
    if "autoencoder" in model_name:
        return "autoencoder"
    return model_name


def _resolve_manifest_paths_in_config(
    config: dict[str, Any],
    *,
    root_dir: Path,
    config_path: Path,
) -> None:
    data_cfg = config.get("data", {})
    for key in ["train_manifest", "val_manifest", "test_id_manifest", "test_ood_manifest"]:
        if data_cfg.get(key):
            data_cfg[key] = _resolve_manifest_path(
                str(data_cfg[key]),
                root_dir=root_dir,
                config_path=config_path,
            ).as_posix()


def _resolve_manifest_path(raw_value: str, *, root_dir: Path, config_path: Path) -> Path:
    raw_path = Path(raw_value)
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates: list[Path] = []
    parts = raw_path.parts
    if parts and parts[0].lower() == root_dir.name.lower():
        candidates.append((root_dir / Path(*parts[1:])).resolve())
    candidates.extend(
        [
            (root_dir / raw_path).resolve(),
            (root_dir.parent / raw_path).resolve(),
            (config_path.parent / raw_path).resolve(),
            (Path.cwd() / raw_path).resolve(),
        ]
    )
    if raw_path.name == "test_id.csv":
        candidates.extend(_test_real_id_alias_candidates(raw_path, root_dir, config_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _test_real_id_alias_candidates(raw_path: Path, root_dir: Path, config_path: Path) -> list[Path]:
    alias = raw_path.with_name("test_real_id.csv")
    parts = alias.parts
    candidates: list[Path] = []
    if parts and parts[0].lower() == root_dir.name.lower():
        candidates.append((root_dir / Path(*parts[1:])).resolve())
    candidates.extend(
        [
            (root_dir / alias).resolve(),
            (root_dir.parent / alias).resolve(),
            (config_path.parent / alias).resolve(),
            (Path.cwd() / alias).resolve(),
        ]
    )
    return candidates


def _materialize_subset_manifests(
    config: dict[str, Any],
    *,
    out_dir: Path,
    max_train_images: int | None,
    max_test_images: int | None,
) -> None:
    if max_train_images is None and max_test_images is None:
        return
    data_cfg = config.get("data", {})
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = {
        "train_manifest": max_train_images,
        "test_id_manifest": max_test_images,
        "test_ood_manifest": max_test_images,
    }
    for key, limit in limits.items():
        if limit is None or not data_cfg.get(key):
            continue
        source = Path(str(data_cfg[key]))
        subset_path = out_dir / source.name
        pd.read_csv(source).head(limit).to_csv(subset_path, index=False)
        data_cfg[key] = subset_path.as_posix()


def _checkpoint_path(kind: str, run_dir: Path) -> Path:
    if kind == "patchcore":
        return run_dir / "patchcore_memory.npz"
    if kind == "autoencoder":
        return run_dir / "model.pt"
    raise ValueError(f"Unsupported experiment type: {kind}")


def _build_commands(kind: str, config_path: Path, checkpoint_path: Path) -> list[list[str]]:
    script_dir = Path(__file__).resolve().parent
    if kind == "patchcore":
        return [
            [sys.executable, str(script_dir / "train_patchcore.py"), "--config", str(config_path)],
            [
                sys.executable,
                str(script_dir / "evaluate.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--no-heatmaps",
            ],
        ]
    if kind == "autoencoder":
        return [
            [sys.executable, str(script_dir / "train_autoencoder.py"), "--config", str(config_path)],
            [
                sys.executable,
                str(script_dir / "evaluate_autoencoder.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
            ],
        ]
    raise ValueError(f"Unsupported experiment type: {kind}")


def _manifest_paths(config: dict[str, Any]) -> list[Path]:
    data_cfg = config.get("data", {})
    paths: list[Path] = []
    for key in ["train_manifest", "val_manifest", "test_id_manifest", "test_ood_manifest"]:
        value = data_cfg.get(key)
        if value:
            paths.append(Path(str(value)).resolve())
    return paths


def _write_manifest_audit(
    prepared: list[PreparedExperiment],
    *,
    root_dir: Path,
    out_path: Path,
) -> dict[str, Any]:
    manifest_paths = _unique_paths(path for item in prepared for path in item.manifest_paths)
    return audit_manifests(manifest_paths, root_dir=root_dir, write_json=out_path)


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _build_aggregate_tables(
    prepared: list[PreparedExperiment],
    *,
    out_dir: Path,
    warnings: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    summary_rows: list[dict[str, Any]] = []
    per_type_rows: list[dict[str, Any]] = []
    per_subtype_rows: list[dict[str, Any]] = []
    has_ood_subtype = False

    for experiment in prepared:
        metrics_path = experiment.eval_dir / "metrics.json"
        scores_path = experiment.eval_dir / "scores.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.json for {experiment.spec.name}: {metrics_path}")
        if not scores_path.exists():
            raise FileNotFoundError(f"Missing scores.csv for {experiment.spec.name}: {scores_path}")

        metrics = _read_json(metrics_path)
        scores = pd.read_csv(scores_path)
        model_cfg = experiment.config.get("model", {})
        shared = {
            "experiment": experiment.spec.name,
            "model": model_cfg.get("name", experiment.spec.kind),
            "backbone": model_cfg.get("backbone", ""),
            "layers": _stringify_layers(model_cfg.get("layers", "")),
            "metrics_file": _relative_path(metrics_path, out_dir),
        }
        summary_rows.append(
            {
                **shared,
                **_summary_metric_values(metrics, scores),
                "warnings": warnings,
            }
        )
        per_type_rows.extend(_per_ood_type_rows(metrics, shared))
        subtype_rows, subtype_present = _per_ood_subtype_rows(experiment, scores, shared)
        per_subtype_rows.extend(subtype_rows)
        has_ood_subtype = has_ood_subtype or subtype_present

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    per_type = pd.DataFrame(per_type_rows, columns=PER_OOD_TYPE_COLUMNS)
    per_subtype = (
        pd.DataFrame(per_subtype_rows, columns=PER_OOD_SUBTYPE_COLUMNS)
        if has_ood_subtype
        else None
    )
    return summary, per_type, per_subtype


def _summary_metric_values(metrics: dict[str, Any], scores: pd.DataFrame) -> dict[str, Any]:
    global_metrics = metrics.get("global", {})
    threshold = metrics.get("threshold", {})
    counts = metrics.get("counts", {})
    confusion = metrics.get("confusion_matrix", {})
    labels = scores["label"].astype(int).to_numpy() if "label" in scores.columns else np.array([])
    score_values = scores["score"].astype(float).to_numpy() if "score" in scores.columns else np.array([])
    id_count = int(counts.get("id", int((labels == 0).sum()) if labels.size else 0))
    ood_count = int(counts.get("ood", int((labels == 1).sum()) if labels.size else 0))
    total_count = int(counts.get("total", id_count + ood_count))
    fp = confusion.get("fp")
    if fp is None:
        fp = _false_positive_count_from_scores(scores, threshold.get("value"))
    id_false_rejection_rate = float(fp) / id_count if id_count else np.nan
    return {
        "auroc": global_metrics.get("auroc"),
        "auprc": global_metrics.get("auprc"),
        "fpr_at_95_tpr": global_metrics.get("fpr_at_95_tpr"),
        "threshold_at_95_tpr": _threshold_at_95_tpr(labels, score_values),
        "threshold": threshold.get("value"),
        "threshold_source": threshold.get("source", ""),
        "id_false_rejection_rate": id_false_rejection_rate,
        "total_count": total_count,
        "id_count": id_count,
        "ood_count": ood_count,
    }


def _false_positive_count_from_scores(scores: pd.DataFrame, threshold: Any) -> int:
    if "prediction" in scores.columns:
        id_rows = scores[scores["label"].astype(int) == 0]
        return int(id_rows["prediction"].astype(int).sum())
    if threshold is None or "score" not in scores.columns or "label" not in scores.columns:
        return 0
    id_scores = scores.loc[scores["label"].astype(int) == 0, "score"].astype(float)
    return int((id_scores >= float(threshold)).sum())


def _threshold_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or scores.size == 0 or len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    eligible = np.where(tpr >= 0.95)[0]
    if eligible.size == 0:
        return float("nan")
    best_fpr = np.min(fpr[eligible])
    best_indices = eligible[fpr[eligible] == best_fpr]
    for index in best_indices:
        if np.isfinite(thresholds[index]):
            return float(thresholds[index])
    return float(thresholds[best_indices[-1]])


def _per_ood_type_rows(metrics: dict[str, Any], shared: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ood_type, values in sorted(metrics.get("per_ood_type", {}).items()):
        rows.append(
            {
                **shared,
                "ood_type": ood_type,
                "auroc": values.get("auroc"),
                "auprc": values.get("auprc"),
                "fpr_at_95_tpr": values.get("fpr_at_95_tpr"),
                "id_count": values.get("id_count"),
                "ood_count": values.get("ood_count"),
            }
        )
    return rows


def _per_ood_subtype_rows(
    experiment: PreparedExperiment,
    scores: pd.DataFrame,
    shared: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    manifest_path = Path(str(experiment.config.get("data", {}).get("test_ood_manifest", "")))
    if not manifest_path.exists():
        return [], False
    manifest = pd.read_csv(manifest_path)
    if "ood_subtype" not in manifest.columns:
        return [], False

    manifest = manifest.copy()
    manifest["_join_path"] = manifest["image_path"].astype(str).map(_normalize_image_path)
    scores = scores.copy()
    scores["_join_path"] = scores["image_path"].astype(str).map(_normalize_image_path)
    subtype_lookup = manifest.set_index("_join_path")[["ood_subtype", "ood_type"]].to_dict("index")

    scores["ood_subtype"] = scores["_join_path"].map(
        lambda value: subtype_lookup.get(value, {}).get("ood_subtype", "")
    )
    scores["joined_ood_type"] = scores["_join_path"].map(
        lambda value: subtype_lookup.get(value, {}).get("ood_type", "")
    )
    id_scores = scores[scores["label"].astype(int) == 0]
    rows: list[dict[str, Any]] = []
    for subtype in sorted(str(value) for value in scores["ood_subtype"].dropna().unique() if value):
        subtype_scores = scores[
            (scores["label"].astype(int) == 1) & (scores["ood_subtype"].astype(str) == subtype)
        ]
        subset = pd.concat([id_scores, subtype_scores], ignore_index=True)
        metric_values = _safe_compute_metrics(subset["label"].to_numpy(), subset["score"].to_numpy())
        ood_type = _first_non_empty(subtype_scores["joined_ood_type"].tolist())
        if not ood_type:
            ood_type = _first_non_empty(subtype_scores.get("ood_type", pd.Series(dtype=str)).tolist())
        rows.append(
            {
                **shared,
                "ood_type": ood_type,
                "ood_subtype": subtype,
                **metric_values,
                "id_count": int(len(id_scores)),
                "ood_count": int(len(subtype_scores)),
            }
        )
    return rows, True


def _safe_compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    try:
        result = compute_ood_metrics(labels.astype(int), scores.astype(float))
    except ValueError:
        return {"auroc": np.nan, "auprc": np.nan, "fpr_at_95_tpr": np.nan}
    return {
        "auroc": result.auroc,
        "auprc": result.auprc,
        "fpr_at_95_tpr": result.fpr_at_95_tpr,
    }


def _normalize_image_path(value: str) -> str:
    return str(value).strip().replace("\\", "/")


def _first_non_empty(values: Sequence[Any]) -> str:
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _stringify_layers(layers: Any) -> str:
    if isinstance(layers, (list, tuple)):
        return "+".join(str(layer) for layer in layers)
    return str(layers) if layers is not None else ""


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _write_markdown(path: Path, title: str, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    note = (
        "Generated from metrics.json and scores.csv only. Raw images, checkpoints, "
        "heatmaps, and patient identifiers are not included."
    )
    path.write_text(f"# {title}\n\n{note}\n\n{dataframe_to_markdown(dataframe)}\n", encoding="utf-8")


def _parse_only(values: list[str] | None) -> list[str]:
    selected: list[str] = []
    for value in values or []:
        selected.extend(part.strip() for part in value.split(",") if part.strip())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a YAML-defined PatchCore/autoencoder experiment grid."
    )
    parser.add_argument("--grid-config", required=True, help="YAML experiment grid config")
    parser.add_argument("--root-dir", required=True, help="Dataset root used by manifest image_path")
    parser.add_argument("--out-dir", required=True, help="Output directory for reports and grid runs")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only")
    parser.add_argument(
        "--only",
        action="append",
        help="Experiment name to run; repeat or use comma-separated names",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with metrics.json")
    parser.add_argument("--max-train-images", type=int, help="Limit train manifest rows for smoke runs")
    parser.add_argument(
        "--max-test-images",
        type=int,
        help="Limit ID and OOD test manifest rows for smoke runs",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, help="Override project.seed in every resolved config")
    args = parser.parse_args()

    result = run_grid(
        grid_config=args.grid_config,
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        only=_parse_only(args.only),
        skip_existing=args.skip_existing,
        max_train_images=args.max_train_images,
        max_test_images=args.max_test_images,
        device=args.device,
        seed=args.seed,
    )
    if args.dry_run:
        for command in result.commands:
            print(" ".join(command))
    else:
        print(f"Saved metrics summary to {result.summary_csv}")
        print(f"Saved per-OOD type metrics to {result.per_ood_type_csv}")
        if result.per_ood_subtype_csv is not None:
            print(f"Saved per-OOD subtype metrics to {result.per_ood_subtype_csv}")
        print(f"Saved manifest audit to {result.manifest_audit_json}")


if __name__ == "__main__":
    main()
