"""Generate a dissertation reporting index from aggregate OOD experiment outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retinal_ood.evaluation.report_tables import dataframe_to_markdown

RESEARCH_THRESHOLD_NOTE = "computed using test labels for paper evaluation only"
DEPLOYMENT_THRESHOLD_NOTE = "computed from val ID score quantile only, for UI/demo decision"

_LAYER_ORDER = {
    "layer1": 0,
    "layer2": 1,
    "layer3": 2,
    "layer4": 3,
    "layer2+layer3": 4,
}

_HEATMAP_COLUMNS = [
    "experiment",
    "outcome",
    "rank",
    "sample_index",
    "image_path",
    "label",
    "score",
    "prediction",
    "threshold",
    "original_file",
    "heatmap_file",
    "overlay_file",
    "manifest_file",
]

_CASE_COLUMNS = [
    "experiment",
    "case_type",
    "rank",
    "image_path",
    "label",
    "ood_type",
    "score",
    "threshold",
    "margin",
    "prediction",
    "scores_file",
]


@dataclass(frozen=True)
class ReportIndexOutputs:
    """Paths written by ``generate_report_index``."""

    files: dict[str, Path]


def generate_report_index(
    reports_dir: str | Path = "reports/generated",
    *,
    top_k: int = 5,
) -> ReportIndexOutputs:
    """Write report index tables from a grid/report output directory."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_required_csv(reports_dir / "metrics_summary.csv")
    per_category = _build_per_category_table(reports_dir)
    layer_ablation = _build_layer_ablation_table(summary)
    method_comparison = _build_ae_vs_patchcore_table(summary)
    threshold_policy = _build_threshold_policy_table(summary)
    heatmap_index, heatmap_note = _build_heatmap_index(reports_dir)
    case_selection = _build_case_selection(reports_dir, top_k=top_k)

    outputs = {
        "per_category_metrics": reports_dir / "per_category_metrics.md",
        "layer_ablation_table": reports_dir / "layer_ablation_table.md",
        "ae_vs_patchcore_table": reports_dir / "ae_vs_patchcore_table.md",
        "threshold_policy_table": reports_dir / "threshold_policy_table.md",
        "heatmap_index": reports_dir / "heatmap_index.md",
        "case_selection": reports_dir / "case_selection.md",
        "index": reports_dir / "index.md",
    }
    _write_table_pair(
        per_category,
        outputs["per_category_metrics"],
        title="Per-Category Metrics",
        note="Per-OOD category/subtype metrics for dissertation comparison.",
    )
    _write_table_pair(
        layer_ablation,
        outputs["layer_ablation_table"],
        title="PatchCore Layer Ablation",
        note="PatchCore-only comparison of feature layers.",
    )
    _write_table_pair(
        method_comparison,
        outputs["ae_vs_patchcore_table"],
        title="Autoencoder vs PatchCore",
        note="Baseline comparison table. PatchCore L2+L3 is preferred when available.",
    )
    _write_table_pair(
        threshold_policy,
        outputs["threshold_policy_table"],
        title="Threshold Policy",
        note=(
            f"research_threshold: {RESEARCH_THRESHOLD_NOTE}. "
            f"deployment_threshold: {DEPLOYMENT_THRESHOLD_NOTE}."
        ),
    )
    _write_table_pair(
        heatmap_index,
        outputs["heatmap_index"],
        title="Heatmap Index",
        note=heatmap_note,
    )
    _write_table_pair(
        case_selection,
        outputs["case_selection"],
        title="Selected TP / FP / FN / Borderline Cases",
        note="Selected from scores.csv only; no raw images or patient identifiers are included.",
    )
    _write_index_markdown(
        outputs["index"],
        reports_dir=reports_dir,
        summary=summary,
        files=outputs,
        warnings=_collect_warnings(reports_dir, summary),
        heatmap_note=heatmap_note,
    )
    return ReportIndexOutputs(files=outputs)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required report table does not exist: {path}")
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_per_category_table(reports_dir: Path) -> pd.DataFrame:
    subtype = _read_optional_csv(reports_dir / "per_ood_subtype_metrics.csv")
    if not subtype.empty:
        table = subtype.copy()
        table.insert(0, "category_level", "ood_subtype")
        table.insert(1, "category", table["ood_subtype"])
        return table
    per_type = _read_optional_csv(reports_dir / "per_ood_type_metrics.csv")
    if per_type.empty:
        return pd.DataFrame(
            columns=[
                "category_level",
                "category",
                "experiment",
                "ood_type",
                "auroc",
                "auprc",
                "fpr_at_95_tpr",
                "id_count",
                "ood_count",
            ]
        )
    table = per_type.copy()
    table.insert(0, "category_level", "ood_type")
    table.insert(1, "category", table["ood_type"])
    return table


def _build_layer_ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[_is_patchcore(summary) & summary.get("layers", pd.Series()).fillna("").astype(str).ne("")]
    table = table.copy()
    if table.empty:
        return pd.DataFrame(columns=_summary_selection_columns())
    table["layer_sort"] = table["layers"].astype(str).map(lambda value: _LAYER_ORDER.get(value, 99))
    table = table.sort_values(["layer_sort", "experiment"], kind="stable")
    return table[_summary_selection_columns()].reset_index(drop=True)


def _build_ae_vs_patchcore_table(summary: pd.DataFrame) -> pd.DataFrame:
    autoencoder = summary[_is_autoencoder(summary)].copy()
    patchcore = summary[_is_patchcore(summary)].copy()
    preferred_patchcore = patchcore[patchcore.get("layers", "").astype(str).eq("layer2+layer3")]
    if preferred_patchcore.empty:
        preferred_patchcore = _best_patchcore_rows(patchcore)
    else:
        preferred_patchcore = preferred_patchcore.copy()
    autoencoder["method"] = "Autoencoder"
    preferred_patchcore["method"] = "PatchCore"
    table = pd.concat([autoencoder, preferred_patchcore], ignore_index=True)
    if table.empty:
        return pd.DataFrame(columns=["method", *_summary_selection_columns()])
    columns = ["method", *_summary_selection_columns()]
    return table[columns].sort_values(["method", "experiment"], kind="stable").reset_index(drop=True)


def _build_threshold_policy_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        shared = {
            "experiment": row.get("experiment", ""),
            "model": row.get("model", ""),
            "layers": row.get("layers", ""),
            "id_count": row.get("id_count", ""),
            "ood_count": row.get("ood_count", ""),
        }
        rows.append(
            {
                **shared,
                "threshold_policy": "research_threshold",
                "threshold": row.get("threshold_at_95_tpr", np.nan),
                "source": "test labels",
                "intended_use": "paper evaluation",
                "warning": f"Research only: {RESEARCH_THRESHOLD_NOTE}; do not use for UI/demo deployment.",
            }
        )
        threshold_source = str(row.get("threshold_source", ""))
        deployment_warning = ""
        if threshold_source != "validation_id_quantile":
            deployment_warning = (
                "Deployment warning: UI/demo threshold should be computed from val ID score "
                f"quantile only; current source is {threshold_source or 'unknown'}."
            )
        rows.append(
            {
                **shared,
                "threshold_policy": "deployment_threshold",
                "threshold": row.get("threshold", np.nan),
                "source": threshold_source,
                "intended_use": "UI/demo decision",
                "warning": deployment_warning,
            }
        )
    return pd.DataFrame(rows)


def _build_heatmap_index(reports_dir: Path) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    for manifest in sorted(reports_dir.rglob("heatmap_manifest.csv")):
        if not _is_heatmap_manifest(manifest):
            continue
        run_name = manifest.parent.parent.parent.name
        heatmap_root = manifest.parent
        data = pd.read_csv(manifest)
        for _, row in data.iterrows():
            output_row = {"experiment": run_name, "manifest_file": _relative_path(manifest, reports_dir)}
            for column in _HEATMAP_COLUMNS:
                if column in {"experiment", "manifest_file"}:
                    continue
                value = row.get(column, "")
                if column.endswith("_file") and str(value).strip():
                    value = _relative_path(heatmap_root / str(value), reports_dir)
                output_row[column] = value
            rows.append(output_row)
    if not rows:
        note = "No heatmap_manifest.csv files were found. Run evaluation with --save-heatmaps to populate this table."
        return pd.DataFrame(columns=_HEATMAP_COLUMNS), note
    note = "Index of generated heatmap artifacts. Image files remain generated outputs, not Git-tracked data."
    return pd.DataFrame(rows, columns=_HEATMAP_COLUMNS), note


def _is_heatmap_manifest(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return bool({"heatmaps", "selected_heatmaps"} & set(parts)) and path.name == "heatmap_manifest.csv"


def _build_case_selection(reports_dir: Path, *, top_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scores_path in sorted(reports_dir.rglob("scores.csv")):
        run_name = scores_path.parent.parent.name if scores_path.parent.name == "evaluation" else scores_path.parent.name
        scores = pd.read_csv(scores_path)
        if scores.empty:
            continue
        threshold = _resolve_scores_threshold(scores, scores_path)
        scored = _normalized_scores(scores, threshold)
        rows.extend(_select_cases(run_name, scored, scores_path, reports_dir, top_k=top_k))
    return pd.DataFrame(rows, columns=_CASE_COLUMNS)


def _resolve_scores_threshold(scores: pd.DataFrame, scores_path: Path) -> float:
    if "threshold" in scores.columns:
        values = pd.to_numeric(scores["threshold"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    metrics_path = scores_path.with_name("metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = metrics.get("threshold", {}).get("value")
        if value is not None:
            return float(value)
    raise ValueError(f"Cannot resolve threshold for case selection: {scores_path}")


def _normalized_scores(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    required = ["image_path", "label", "score"]
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise ValueError(f"scores.csv missing required columns: {missing}")
    table = scores.copy()
    table["label"] = table["label"].astype(int)
    table["score"] = table["score"].astype(float)
    table["threshold"] = threshold
    if "prediction" not in table.columns:
        table["prediction"] = (table["score"] >= threshold).astype(int)
    else:
        table["prediction"] = table["prediction"].astype(int)
    if "ood_type" not in table.columns:
        table["ood_type"] = ""
    table["margin"] = table["score"] - threshold
    table["abs_margin"] = table["margin"].abs()
    return table


def _select_cases(
    run_name: str,
    scores: pd.DataFrame,
    scores_path: Path,
    reports_dir: Path,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    masks = {
        "tp": (scores["label"] == 1) & (scores["prediction"] == 1),
        "fp": (scores["label"] == 0) & (scores["prediction"] == 1),
        "fn": (scores["label"] == 1) & (scores["prediction"] == 0),
        "borderline": pd.Series([True] * len(scores), index=scores.index),
    }
    sort_columns = {
        "tp": (["score"], [False]),
        "fp": (["score"], [False]),
        "fn": (["score"], [True]),
        "borderline": (["abs_margin", "image_path"], [True, True]),
    }
    rows: list[dict[str, Any]] = []
    for case_type, mask in masks.items():
        sort_by, ascending = sort_columns[case_type]
        selected = scores[mask].sort_values(sort_by, ascending=ascending, kind="stable").head(top_k)
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            rows.append(
                {
                    "experiment": run_name,
                    "case_type": case_type,
                    "rank": rank,
                    "image_path": row.get("image_path", ""),
                    "label": int(row["label"]),
                    "ood_type": row.get("ood_type", ""),
                    "score": float(row["score"]),
                    "threshold": float(row["threshold"]),
                    "margin": float(row["margin"]),
                    "prediction": int(row["prediction"]),
                    "scores_file": _relative_path(scores_path, reports_dir),
                }
            )
    return rows


def _summary_selection_columns() -> list[str]:
    return [
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
        "id_count",
        "ood_count",
    ]


def _is_patchcore(summary: pd.DataFrame) -> pd.Series:
    model = summary.get("model", pd.Series("", index=summary.index)).fillna("").astype(str).str.lower()
    experiment = summary.get("experiment", pd.Series("", index=summary.index)).fillna("").astype(str).str.lower()
    return model.str.contains("patchcore") | experiment.str.contains("patchcore")


def _is_autoencoder(summary: pd.DataFrame) -> pd.Series:
    model = summary.get("model", pd.Series("", index=summary.index)).fillna("").astype(str).str.lower()
    experiment = summary.get("experiment", pd.Series("", index=summary.index)).fillna("").astype(str).str.lower()
    return model.str.contains("autoencoder") | experiment.str.contains("autoencoder")


def _best_patchcore_rows(patchcore: pd.DataFrame) -> pd.DataFrame:
    if patchcore.empty:
        return patchcore
    table = patchcore.copy()
    table["auroc"] = pd.to_numeric(table.get("auroc"), errors="coerce")
    return table.sort_values(["auroc", "experiment"], ascending=[False, True], kind="stable").head(1)


def _write_table_pair(dataframe: pd.DataFrame, markdown_path: Path, *, title: str, note: str) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = markdown_path.with_suffix(".csv")
    dataframe.to_csv(csv_path, index=False)
    markdown_path.write_text(
        f"# {title}\n\n{note}\n\n{dataframe_to_markdown(dataframe)}\n",
        encoding="utf-8",
    )


def _write_index_markdown(
    path: Path,
    *,
    reports_dir: Path,
    summary: pd.DataFrame,
    files: dict[str, Path],
    warnings: list[str],
    heatmap_note: str,
) -> None:
    table_links = [
        ("Per-category metric table", files["per_category_metrics"]),
        ("Layer ablation table", files["layer_ablation_table"]),
        ("AE vs PatchCore table", files["ae_vs_patchcore_table"]),
        ("Threshold policy table", files["threshold_policy_table"]),
        ("Heatmap index", files["heatmap_index"]),
        ("Top TP / FP / FN / borderline case selection", files["case_selection"]),
    ]
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No warnings recorded."
    links = "\n".join(f"- [{label}]({_relative_path(file, reports_dir)})" for label, file in table_links)
    content = f"""# Generated Experiment Report Index

This index is generated from aggregate metrics and score tables only. It does not include raw
medical images, model weights, generated heatmaps, or patient identifiers.

## Threshold Warning

- `research_threshold`: {RESEARCH_THRESHOLD_NOTE}.
- `deployment_threshold`: {DEPLOYMENT_THRESHOLD_NOTE}.

Research thresholds can use test labels to report paper metrics such as FPR@95%TPR. Deployment
thresholds must be calibrated from validation ID scores only before UI/demo decisions.

## Available Tables

{links}

## Run Summary

- experiments: {len(summary)}
- ID samples total across rows: {_safe_sum(summary, "id_count")}
- OOD samples total across rows: {_safe_sum(summary, "ood_count")}
- heatmap note: {heatmap_note}

## Warnings

{warning_lines}
"""
    path.write_text(content, encoding="utf-8")


def _collect_warnings(reports_dir: Path, summary: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "warnings" in summary.columns:
        for value in summary["warnings"].dropna().astype(str):
            if value.strip():
                warnings.append(value.strip())
    audit_path = reports_dir / "manifest_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        warnings.extend(str(warning) for warning in audit.get("warnings", []) if str(warning).strip())
    return _dedupe(warnings)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _safe_sum(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    return int(pd.to_numeric(dataframe[column], errors="coerce").fillna(0).sum())


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
