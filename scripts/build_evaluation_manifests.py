#!/usr/bin/env python
"""Build local-only evaluation manifest subsets for dissertation experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.data.image_audit import infer_ood_subtype_from_row
from retinal_ood.data.manifest_audit import REQUIRED_COLUMNS

OPTIONAL_PRESERVE_COLUMNS = [
    "source",
    "ood_type",
    "ood_subtype",
    "source_split",
    "source_image_hash",
    "patient_id",
    "scanner",
    "notes",
]


def build_evaluation_manifests(
    *,
    test_ood_manifest: str | Path,
    out_dir: str | Path,
    test_id_manifest: str | Path | None = None,
    seed: int = 42,
    smoke_per_type: int = 2,
) -> dict[str, str]:
    """Create full, balanced, and smoke OOD evaluation manifest subsets."""
    if smoke_per_type <= 0:
        raise ValueError("smoke_per_type must be positive")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ood = _read_manifest(Path(test_ood_manifest))
    warnings = _ensure_ood_subtypes(ood)
    _validate_ood_rows(ood)
    columns = _preserved_columns(ood)

    full = ood[columns].copy()
    balanced_by_type = _balanced_sample(full, group_column="ood_type", seed=seed)
    balanced_by_subtype = _balanced_sample(full, group_column="ood_subtype", seed=seed)
    smoke = _smoke_sample(full, seed=seed, smoke_per_type=smoke_per_type)

    outputs = {
        "test_ood_full": out_path / "test_ood_full.csv",
        "test_ood_balanced_by_type": out_path / "test_ood_balanced_by_type.csv",
        "test_ood_balanced_by_subtype": out_path / "test_ood_balanced_by_subtype.csv",
        "test_ood_smoke": out_path / "test_ood_smoke.csv",
        "audit_json": out_path / "evaluation_manifest_audit.json",
    }
    full.to_csv(outputs["test_ood_full"], index=False)
    balanced_by_type.to_csv(outputs["test_ood_balanced_by_type"], index=False)
    balanced_by_subtype.to_csv(outputs["test_ood_balanced_by_subtype"], index=False)
    smoke.to_csv(outputs["test_ood_smoke"], index=False)

    if test_id_manifest is not None:
        id_output = _write_optional_synthetic_fallback_id(
            Path(test_id_manifest),
            out_path,
            warnings,
        )
        if id_output is not None:
            outputs["test_id_synthetic_fallback"] = id_output

    audit = _build_subset_audit(
        seed=seed,
        test_ood_manifest=Path(test_ood_manifest),
        outputs=outputs,
        frames={
            "test_ood_full": full,
            "test_ood_balanced_by_type": balanced_by_type,
            "test_ood_balanced_by_subtype": balanced_by_subtype,
            "test_ood_smoke": smoke,
        },
        warnings=warnings,
    )
    outputs["audit_json"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {key: value.as_posix() for key, value in outputs.items()}


def _read_manifest(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise ValueError(f"Manifest file does not exist: {path}") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"Manifest {path} contains no rows")
    for column in [*REQUIRED_COLUMNS, *OPTIONAL_PRESERVE_COLUMNS]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    return df


def _validate_ood_rows(df: pd.DataFrame) -> None:
    if (df["split"] == "train").any():
        raise ValueError("Evaluation OOD manifest must not contain split=train rows")
    bad_label = df["label"] != 1
    if bad_label.any():
        raise ValueError("Evaluation OOD manifest must contain label=1 rows only")
    bad_type = df["ood_type"].eq("id")
    if bad_type.any():
        raise ValueError("Evaluation OOD manifest must not contain ood_type=id rows")
    if "ood_subtype" not in df.columns or df["ood_subtype"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Evaluation subset builder requires non-empty ood_subtype for every OOD row")
    if "source_split" in df.columns:
        bad_source = (df["ood_type"] == "sensory_artifact") & (df["source_split"].str.lower() == "train")
        if bad_source.any():
            raise ValueError("sensory_artifact rows must not be generated from source_split=train")


def _ensure_ood_subtypes(df: pd.DataFrame) -> list[str]:
    if "ood_subtype" not in df.columns:
        df["ood_subtype"] = ""
    explicit_missing = df["ood_subtype"].fillna("").astype(str).str.strip().eq("")
    if explicit_missing.any():
        df.loc[explicit_missing, "ood_subtype"] = [
            infer_ood_subtype_from_row(row)
            for _, row in df.loc[explicit_missing].iterrows()
        ]
        resolved = int(df.loc[explicit_missing, "ood_subtype"].astype(str).str.strip().ne("").sum())
        unresolved = int(df["ood_subtype"].fillna("").astype(str).str.strip().eq("").sum())
        warning = (
            f"{int(explicit_missing.sum())} OOD rows were missing explicit ood_subtype; "
            f"inferred {resolved} from image_path/notes, unresolved {unresolved}"
        )
        return [warning]
    return []


def _preserved_columns(df: pd.DataFrame) -> list[str]:
    columns = list(dict.fromkeys([*REQUIRED_COLUMNS, *OPTIONAL_PRESERVE_COLUMNS]))
    return [column for column in columns if column in df.columns]


def _balanced_sample(df: pd.DataFrame, *, group_column: str, seed: int) -> pd.DataFrame:
    if group_column not in df.columns:
        raise ValueError(f"Manifest must include {group_column} for balanced sampling")
    counts = df[group_column].value_counts()
    if counts.empty:
        raise ValueError(f"Manifest has no {group_column} groups")
    n = int(counts.min())
    frames = [
        group.sort_values("image_path").sample(n=n, random_state=seed)
        for _, group in sorted(df.groupby(group_column), key=lambda item: str(item[0]))
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values([group_column, "image_path"], kind="stable")
        .reset_index(drop=True)
    )


def _smoke_sample(df: pd.DataFrame, *, seed: int, smoke_per_type: int) -> pd.DataFrame:
    frames = []
    for _, group in sorted(df.groupby("ood_type"), key=lambda item: str(item[0])):
        n = min(smoke_per_type, len(group))
        frames.append(group.sort_values("image_path").sample(n=n, random_state=seed))
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ood_type", "image_path"], kind="stable")
        .reset_index(drop=True)
    )


def _write_optional_synthetic_fallback_id(
    test_id_manifest: Path,
    out_dir: Path,
    warnings: list[str],
) -> Path | None:
    df = _read_manifest(test_id_manifest)
    sources = [str(source).lower() for source in df["source"].dropna().unique().tolist()]
    synthetic_fallback = "real" in test_id_manifest.name.lower() and sources and all(
        "synthetic" in source for source in sources
    )
    if not synthetic_fallback:
        return None
    output = out_dir / "test_id_synthetic_fallback.csv"
    warning = (
        f"{test_id_manifest}: synthetic fallback only; not real clinical FAF validation"
    )
    warnings.append(warning)
    df = df.copy()
    if "notes" in df.columns:
        df["notes"] = df["notes"].fillna("").astype(str).map(
            lambda value: f"{value}; {warning}".strip("; ")
        )
    df.to_csv(output, index=False)
    return output


def _build_subset_audit(
    *,
    seed: int,
    test_ood_manifest: Path,
    outputs: dict[str, Path],
    frames: dict[str, pd.DataFrame],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "test_ood_manifest": test_ood_manifest.as_posix(),
        "outputs": {key: value.as_posix() for key, value in sorted(outputs.items())},
        "counts": {
            name: {
                "rows": int(len(frame)),
                "ood_type": _counts(frame, "ood_type"),
                "ood_subtype": _counts(frame, "ood_subtype"),
            }
            for name, frame in frames.items()
        },
        "warnings": warnings,
    }


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in df[column].fillna("").astype(str).value_counts().sort_index().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local-only OOD evaluation manifest subsets")
    parser.add_argument("--test-ood-manifest", required=True, help="Input test_ood.csv")
    parser.add_argument("--test-id-manifest", help="Optional test ID manifest for synthetic fallback copy")
    parser.add_argument("--out-dir", default="data/manifests/generated", help="Local output directory")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--smoke-per-type", type=int, default=2, help="Rows per OOD type for smoke manifest")
    args = parser.parse_args()

    outputs = build_evaluation_manifests(
        test_ood_manifest=args.test_ood_manifest,
        test_id_manifest=args.test_id_manifest,
        out_dir=args.out_dir,
        seed=args.seed,
        smoke_per_type=args.smoke_per_type,
    )
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
