"""Build dissertation-ready result tables from evaluation metrics files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

OVERALL_COLUMNS = [
    "run_name",
    "run_dir",
    "model",
    "backbone",
    "layers",
    "auroc",
    "auprc",
    "fpr_at_95_tpr",
    "threshold",
    "threshold_source",
    "total_count",
    "id_count",
    "ood_count",
    "tn",
    "fp",
    "fn",
    "tp",
    "metrics_file",
]

PER_OOD_COLUMNS = [
    "run_name",
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


@dataclass(frozen=True)
class ReportOutputPaths:
    overall_csv: Path
    overall_markdown: Path
    per_ood_csv: Path
    per_ood_markdown: Path


@dataclass(frozen=True)
class ReportTables:
    overall: pd.DataFrame
    per_ood: pd.DataFrame


def find_metrics_files(runs_dir: str | Path) -> list[Path]:
    """Find evaluation ``metrics.json`` files under a runs directory."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir does not exist: {runs_dir}")
    metrics_files = sorted(path for path in runs_dir.rglob("metrics.json") if path.is_file())
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.json files found under {runs_dir}")
    return metrics_files


def build_report_tables(runs_dir: str | Path) -> ReportTables:
    """Read all metrics files and return global and per-OOD result tables."""
    runs_dir = Path(runs_dir)
    overall_rows: list[dict[str, Any]] = []
    per_ood_rows: list[dict[str, Any]] = []
    for metrics_file in find_metrics_files(runs_dir):
        metrics = _read_json(metrics_file)
        config = _read_optional_config(metrics_file)
        overall_rows.append(_build_overall_row(metrics_file, runs_dir, metrics, config))
        per_ood_rows.extend(_build_per_ood_rows(metrics_file, runs_dir, metrics, config))

    overall = pd.DataFrame(overall_rows, columns=OVERALL_COLUMNS).sort_values(
        by=["run_name", "metrics_file"],
        kind="stable",
    )
    per_ood = pd.DataFrame(per_ood_rows, columns=PER_OOD_COLUMNS)
    if not per_ood.empty:
        per_ood = per_ood.sort_values(by=["run_name", "ood_type"], kind="stable")
    return ReportTables(overall=overall.reset_index(drop=True), per_ood=per_ood.reset_index(drop=True))


def write_report_tables(
    runs_dir: str | Path,
    *,
    overall_markdown: str | Path = "reports/generated/experiment_summary.md",
    overall_csv: str | Path | None = None,
    per_ood_markdown: str | Path | None = None,
    per_ood_csv: str | Path | None = None,
) -> ReportOutputPaths:
    """Write CSV and Markdown summaries for dissertation reporting."""
    tables = build_report_tables(runs_dir)
    output_paths = resolve_report_output_paths(
        overall_markdown,
        overall_csv=overall_csv,
        per_ood_markdown=per_ood_markdown,
        per_ood_csv=per_ood_csv,
    )
    _write_dataframe_csv(tables.overall, output_paths.overall_csv)
    _write_markdown_report(
        output_paths.overall_markdown,
        title="Experiment Summary",
        note=(
            "Generated from evaluation metrics.json files only. "
            "Per-image scores, patient identifiers, and raw image paths are not included."
        ),
        dataframe=tables.overall,
    )
    _write_dataframe_csv(tables.per_ood, output_paths.per_ood_csv)
    _write_markdown_report(
        output_paths.per_ood_markdown,
        title="Per-OOD Category Metrics",
        note="Each row compares one OOD category against the same ID test set.",
        dataframe=tables.per_ood,
    )
    return output_paths


def resolve_report_output_paths(
    overall_markdown: str | Path,
    *,
    overall_csv: str | Path | None = None,
    per_ood_markdown: str | Path | None = None,
    per_ood_csv: str | Path | None = None,
) -> ReportOutputPaths:
    """Resolve default companion CSV and per-category output paths."""
    overall_markdown = Path(overall_markdown)
    overall_csv_path = Path(overall_csv) if overall_csv is not None else overall_markdown.with_suffix(".csv")
    if per_ood_markdown is None:
        if overall_markdown.name == "experiment_summary.md":
            per_ood_markdown_path = overall_markdown.parent / "per_category_metrics.md"
        else:
            per_ood_markdown_path = overall_markdown.with_name(f"{overall_markdown.stem}_per_ood.md")
    else:
        per_ood_markdown_path = Path(per_ood_markdown)
    per_ood_csv_path = (
        Path(per_ood_csv)
        if per_ood_csv is not None
        else per_ood_markdown_path.with_suffix(".csv")
    )
    return ReportOutputPaths(
        overall_csv=overall_csv_path,
        overall_markdown=overall_markdown,
        per_ood_csv=per_ood_csv_path,
        per_ood_markdown=per_ood_markdown_path,
    )


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-compatible Markdown table."""
    columns = list(dataframe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(_format_markdown_cell(row[column]) for column in columns) + " |"
        for _, row in dataframe.iterrows()
    ]
    if not rows:
        rows = ["| " + " | ".join([""] * len(columns)) + " |"]
    return "\n".join([header, separator, *rows])


def _build_overall_row(
    metrics_file: Path,
    runs_dir: Path,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    global_metrics = metrics.get("global", {})
    confusion = metrics.get("confusion_matrix", {})
    threshold = metrics.get("threshold", {})
    counts = metrics.get("counts", {})
    model_cfg = config.get("model", {})
    return {
        "run_name": _run_name(metrics_file, config),
        "run_dir": _relative_run_dir(metrics_file, runs_dir),
        "model": model_cfg.get("name", ""),
        "backbone": model_cfg.get("backbone", ""),
        "layers": _stringify_layers(model_cfg.get("layers", "")),
        "auroc": global_metrics.get("auroc"),
        "auprc": global_metrics.get("auprc"),
        "fpr_at_95_tpr": global_metrics.get("fpr_at_95_tpr"),
        "threshold": threshold.get("value"),
        "threshold_source": threshold.get("source", ""),
        "total_count": counts.get("total"),
        "id_count": counts.get("id"),
        "ood_count": counts.get("ood"),
        "tn": confusion.get("tn"),
        "fp": confusion.get("fp"),
        "fn": confusion.get("fn"),
        "tp": confusion.get("tp"),
        "metrics_file": _relative_path(metrics_file, runs_dir),
    }


def _build_per_ood_rows(
    metrics_file: Path,
    runs_dir: Path,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    model_cfg = config.get("model", {})
    shared = {
        "run_name": _run_name(metrics_file, config),
        "model": model_cfg.get("name", ""),
        "backbone": model_cfg.get("backbone", ""),
        "layers": _stringify_layers(model_cfg.get("layers", "")),
        "metrics_file": _relative_path(metrics_file, runs_dir),
    }
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


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _read_optional_config(metrics_file: Path) -> dict[str, Any]:
    candidates = [
        metrics_file.parent / "resolved_evaluation_config.json",
        metrics_file.parent / "resolved_config.json",
        metrics_file.parent.parent / "resolved_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_json(candidate)
    return {}


def _run_name(metrics_file: Path, config: dict[str, Any]) -> str:
    run_name = config.get("project", {}).get("run_name")
    if run_name:
        return str(run_name)
    if metrics_file.parent.name == "evaluation":
        return metrics_file.parent.parent.name
    return metrics_file.parent.name


def _relative_run_dir(metrics_file: Path, runs_dir: Path) -> str:
    run_dir = metrics_file.parent.parent if metrics_file.parent.name == "evaluation" else metrics_file.parent
    return _relative_path(run_dir, runs_dir)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _stringify_layers(layers: Any) -> str:
    if isinstance(layers, (list, tuple)):
        return "+".join(str(layer) for layer in layers)
    return str(layers) if layers is not None else ""


def _write_dataframe_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _write_markdown_report(path: Path, *, title: str, note: str, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# {title}\n\n{note}\n\n{dataframe_to_markdown(dataframe)}\n"
    path.write_text(content, encoding="utf-8")


def _format_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")
