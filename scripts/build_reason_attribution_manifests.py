#!/usr/bin/env python
"""Build OOD-only manifests for the optional Stage 2 reason attribution module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.evaluation.report_tables import dataframe_to_markdown
from retinal_ood.reason_attribution.classifier import FAMILY_CLASSES, SUBTYPE_CLASSES
from retinal_ood.reason_attribution.grouped_split import parent_grouped_stratified_split

SAFE_COLUMNS = [
    "image_path",
    "label",
    "split",
    "source",
    "ood_type",
    "ood_subtype",
    "source_dataset",
    "source_url",
    "license_status",
    "source_split",
    "source_image_hash",
    "parent_image_hash",
    "synthetic_transform",
    "severity",
    "notes",
]

FORBIDDEN_COLUMNS = {
    "patient_id",
    "Eye_ID",
    "clinical_label",
    "disease_label",
    "biomarker_label",
}

PRIVATE_PATH_TOKENS = ["C:" + "\\Users\\", "/" + "Users/", "Documents" + "/Codex"]


class ReasonManifestResult(NamedTuple):
    manifest_paths: dict[str, Path]
    audit_markdown: Path
    audit_csv: Path


def build_reason_manifests(
    *,
    input_manifest: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    split_mode: str = "row",
    output_prefix: str = "reason",
    reports_dir: str | Path = "reports/dissertation_results/reason_attribution",
) -> ReasonManifestResult:
    """Build deterministic OOD-only reason attribution train/val/test manifests."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    _validate_split_mode(split_mode)
    source = pd.read_csv(input_manifest)
    _validate_input_frame(source)
    prepared = _prepare_safe_frame(source)
    split_frames = _split_reason_frame(
        prepared,
        split_mode=split_mode,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    for split_name, frame in split_frames.items():
        path = output_dir / f"{output_prefix}_{split_name}.csv"
        frame.to_csv(path, index=False)
        manifest_paths[split_name] = path

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    audit_table = _build_audit_table(split_frames, source)
    audit_csv = reports_path / f"{output_prefix}_manifest_audit.csv"
    audit_md = reports_path / f"{output_prefix}_manifest_audit.md"
    audit_table.to_csv(audit_csv, index=False)
    _write_audit_markdown(audit_md, audit_table)
    return ReasonManifestResult(
        manifest_paths=manifest_paths,
        audit_markdown=audit_md,
        audit_csv=audit_csv,
    )


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("train/val/test ratios must be positive")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")


def _validate_split_mode(split_mode: str) -> None:
    if split_mode not in {"row", "grouped"}:
        raise ValueError("split_mode must be 'row' or 'grouped'")


def _validate_input_frame(frame: pd.DataFrame) -> None:
    required = {"image_path", "label", "split", "source", "ood_type", "ood_subtype"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input manifest missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input manifest contains no rows")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not (labels == 1).all():
        raise ValueError("Reason attribution manifests must be OOD-only with label=1")
    ood_type = frame["ood_type"].astype(str).str.strip()
    unknown_types = sorted(set(ood_type) - set(FAMILY_CLASSES))
    if unknown_types:
        raise ValueError(f"Unsupported ood_type values for reason attribution: {unknown_types}")
    subtype = frame["ood_subtype"].astype(str).str.strip()
    missing_subtype = subtype.eq("") | subtype.str.lower().eq("nan")
    if missing_subtype.any():
        raise ValueError("All reason attribution rows must have explicit ood_subtype")
    unknown_subtypes = sorted(set(subtype) - set(SUBTYPE_CLASSES))
    if unknown_subtypes:
        raise ValueError(f"Unsupported ood_subtype values for reason attribution: {unknown_subtypes}")
    _fail_on_private_paths(frame)
    _fail_on_forbidden_identifiers(frame)


def _fail_on_private_paths(frame: pd.DataFrame) -> None:
    paths = frame["image_path"].astype(str).str.replace("\\", "/", regex=False)
    private = paths.map(lambda value: any(token.replace("\\", "/") in value for token in PRIVATE_PATH_TOKENS))
    if private.any():
        raise ValueError("Input manifest contains a private path in image_path")


def _fail_on_forbidden_identifiers(frame: pd.DataFrame) -> None:
    for column in FORBIDDEN_COLUMNS & set(frame.columns):
        values = frame[column].fillna("").astype(str).str.strip()
        non_empty = values.ne("") & values.str.lower().ne("nan")
        if non_empty.any():
            raise ValueError(f"Forbidden identifier column {column} contains non-empty values")


def _prepare_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in SAFE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = ""
    prepared = prepared[SAFE_COLUMNS].copy()
    prepared["label"] = 1
    prepared["ood_type"] = prepared["ood_type"].astype(str).str.strip()
    prepared["ood_subtype"] = prepared["ood_subtype"].astype(str).str.strip()
    return prepared


def _split_reason_frame(
    frame: pd.DataFrame,
    *,
    split_mode: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, pd.DataFrame]:
    if split_mode == "grouped":
        return parent_grouped_stratified_split(
            frame,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    return _stratified_split(
        frame,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def _stratified_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, pd.DataFrame]:
    split_rows = {"train": [], "val": [], "test": []}
    for subtype, group in frame.groupby("ood_subtype", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed + _stable_subtype_offset(str(subtype))).reset_index(drop=True)
        n_total = len(shuffled)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
        n_test = n_total - n_train - n_val
        if n_total >= 3 and min(n_train, n_val, n_test) <= 0:
            n_train = max(1, int(n_total * train_ratio))
            n_val = max(1, int(n_total * val_ratio))
            n_test = n_total - n_train - n_val
        if n_test <= 0:
            raise ValueError(f"Not enough rows to split subtype {subtype!r}")
        split_rows["train"].append(shuffled.iloc[:n_train])
        split_rows["val"].append(shuffled.iloc[n_train : n_train + n_val])
        split_rows["test"].append(shuffled.iloc[n_train + n_val :])
    result: dict[str, pd.DataFrame] = {}
    for split_name, parts in split_rows.items():
        split_frame = pd.concat(parts, ignore_index=True)
        split_frame["split"] = split_name
        result[split_name] = split_frame.sort_values(["ood_subtype", "image_path"], kind="stable").reset_index(drop=True)
    return result


def _stable_subtype_offset(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value))


def _build_audit_table(split_frames: dict[str, pd.DataFrame], source: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [frame.assign(reason_split=split_name) for split_name, frame in split_frames.items()],
        ignore_index=True,
    )
    rows: list[dict[str, Any]] = []
    for split_name, count in combined["reason_split"].value_counts().sort_index().items():
        rows.append({"section": "split", "label": split_name, "count": int(count), "notes": "reason attribution split"})
    for ood_type, count in combined["ood_type"].value_counts().sort_index().items():
        rows.append({"section": "ood_type", "label": ood_type, "count": int(count), "notes": "Stage 2 target family"})
    for subtype, count in combined["ood_subtype"].value_counts().sort_index().items():
        rows.append({"section": "ood_subtype", "label": subtype, "count": int(count), "notes": "Stage 2 optional subtype target"})
    rows.extend(
        [
            {
                "section": "safety",
                "label": "missing_subtype_count",
                "count": int(combined["ood_subtype"].astype(str).str.strip().eq("").sum()),
                "notes": "must be zero",
            },
            {
                "section": "safety",
                "label": "non_empty_patient_identifier_count",
                "count": int(
                    sum(
                        _non_empty_count(source[column])
                        for column in FORBIDDEN_COLUMNS & set(source.columns)
                    )
                ),
                "notes": "must be zero; forbidden columns are not written to reason manifests",
            },
            {
                "section": "safety",
                "label": "private_path_count",
                "count": int(
                    combined["image_path"]
                    .astype(str)
                    .str.replace("\\", "/", regex=False)
                    .map(lambda value: any(token.replace("\\", "/") in value for token in PRIVATE_PATH_TOKENS))
                    .sum()
                ),
                "notes": "must be zero",
            },
            {
                "section": "safety",
                "label": "stage_boundary",
                "count": int(len(combined)),
                "notes": "Stage 2 explanation only; Stage 1 remains ID-only and unchanged",
            },
        ]
    )
    return pd.DataFrame(rows, columns=["section", "label", "count", "notes"])


def _non_empty_count(series: pd.Series) -> int:
    values = series.fillna("").astype(str).str.strip()
    return int((values.ne("") & values.str.lower().ne("nan")).sum())


def _write_audit_markdown(path: Path, table: pd.DataFrame) -> None:
    note = (
        "These manifests are for Stage 2 explanation only. Stage 1 remains ID-only, "
        "unsupervised, binary ACCEPT/REJECT, and is not trained with OOD taxonomy labels."
    )
    path.write_text(
        f"# Reason Attribution Manifest Audit\n\n{note}\n\n{dataframe_to_markdown(table)}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OOD-only Stage 2 reason attribution manifests.")
    parser.add_argument("--input", required=True, help="Input OOD manifest, usually test_ood_full.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for reason_train/val/test CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=("row", "grouped"), default="row")
    parser.add_argument("--output-prefix", default="reason", help="Prefix for split CSV and audit filenames")
    parser.add_argument(
        "--reports-dir",
        default="reports/dissertation_results/reason_attribution",
        help="Directory for reason_manifest_audit.md/csv",
    )
    args = parser.parse_args()
    result = build_reason_manifests(
        input_manifest=args.input,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_mode=args.split_mode,
        output_prefix=args.output_prefix,
        reports_dir=args.reports_dir,
    )
    for split_name, path in result.manifest_paths.items():
        print(f"{split_name}: {path}")
    print(f"audit_markdown: {result.audit_markdown}")
    print(f"audit_csv: {result.audit_csv}")


if __name__ == "__main__":
    main()
