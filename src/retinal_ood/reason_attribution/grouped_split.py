"""Deterministic parent-grouped splits for Stage 2 reason attribution."""

from __future__ import annotations

import random
from itertools import combinations
from typing import NamedTuple

import pandas as pd

_REQUIRED_COLUMNS = {"image_path", "ood_subtype", "parent_image_hash"}
_SPLIT_NAMES = ("train", "val", "test")


class _GroupRecord(NamedTuple):
    group_id: str
    profile: tuple[tuple[str, int], ...]


def normalize_group_id(parent_image_hash: object, image_path: object) -> str:
    """Normalize a parent group id, falling back to the exact trimmed image path."""
    normalized_parent = _normalize_parent_hash(parent_image_hash)
    if normalized_parent is not None:
        return normalized_parent
    return _normalize_image_path(image_path)


def parent_grouped_stratified_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, pd.DataFrame]:
    """Split rows by parent/image groups while preserving subtype-count strata."""
    _validate_grouped_frame(frame)
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    working = frame.copy()
    working["_normalized_group_id"] = [
        normalize_group_id(parent_image_hash, image_path)
        for parent_image_hash, image_path in zip(working["parent_image_hash"], working["image_path"])
    ]
    working["_ood_subtype_key"] = working["ood_subtype"].astype(str).str.strip()
    if working["_normalized_group_id"].eq("").any():
        raise ValueError("Grouped split requires non-empty normalized group ids for every row")
    if working["_ood_subtype_key"].eq("").any() or working["_ood_subtype_key"].str.lower().eq("nan").any():
        raise ValueError("Grouped split requires non-empty ood_subtype values for every row")

    groups_by_profile: dict[tuple[tuple[str, int], ...], list[str]] = {}
    for record in _build_group_records(working):
        groups_by_profile.setdefault(record.profile, []).append(record.group_id)

    split_group_ids = {split_name: [] for split_name in _SPLIT_NAMES}
    for profile in sorted(groups_by_profile):
        ordered_group_ids = sorted(groups_by_profile[profile])
        shuffled_group_ids = _deterministic_shuffle(
            ordered_group_ids,
            seed=seed + _stable_profile_offset(profile),
        )
        n_train, n_val, n_test = _compute_partition_sizes(
            len(shuffled_group_ids),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        split_group_ids["train"].extend(shuffled_group_ids[:n_train])
        split_group_ids["val"].extend(shuffled_group_ids[n_train : n_train + n_val])
        split_group_ids["test"].extend(shuffled_group_ids[n_train + n_val : n_train + n_val + n_test])

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in _SPLIT_NAMES:
        split_frame = (
            working.loc[working["_normalized_group_id"].isin(split_group_ids[split_name])]
            .drop(columns=["_normalized_group_id", "_ood_subtype_key"])
            .copy()
        )
        split_frame["split"] = split_name
        sort_columns = [column for column in split_frame.columns if column != "split"] + ["split"]
        split_frames[split_name] = split_frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    _validate_split_disjointness(split_frames)
    return split_frames


def _build_group_records(frame: pd.DataFrame) -> list[_GroupRecord]:
    records: list[_GroupRecord] = []
    for group_id, group in frame.groupby("_normalized_group_id", sort=True):
        subtype_counts = group["_ood_subtype_key"].value_counts().sort_index()
        profile = tuple((str(subtype), int(count)) for subtype, count in subtype_counts.items())
        records.append(_GroupRecord(group_id=str(group_id), profile=profile))
    return records


def _normalize_parent_hash(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _normalize_image_path(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _validate_grouped_frame(frame: pd.DataFrame) -> None:
    missing_columns = _REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Grouped split requires columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Grouped split requires at least one row")


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("train/val/test ratios must be positive")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")


def _deterministic_shuffle(group_ids: list[str], *, seed: int) -> list[str]:
    shuffled = list(group_ids)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _compute_partition_sizes(
    n_total: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    del test_ratio
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
        raise ValueError(f"Not enough groups to split profile with {n_total} entries")
    return n_train, n_val, n_test


def _stable_profile_offset(profile: tuple[tuple[str, int], ...]) -> int:
    token = "|".join(f"{subtype}:{count}" for subtype, count in profile)
    return sum((index + 1) * ord(char) for index, char in enumerate(token))


def _validate_split_disjointness(split_frames: dict[str, pd.DataFrame]) -> None:
    image_paths_by_split = {
        split_name: set(frame["image_path"].astype(str))
        for split_name, frame in split_frames.items()
    }
    normalized_groups_by_split = {
        split_name: {
            normalize_group_id(parent_image_hash, image_path)
            for parent_image_hash, image_path in zip(frame["parent_image_hash"], frame["image_path"])
        }
        for split_name, frame in split_frames.items()
    }
    for left_name, right_name in combinations(_SPLIT_NAMES, 2):
        overlapping_paths = image_paths_by_split[left_name] & image_paths_by_split[right_name]
        if overlapping_paths:
            sample = sorted(overlapping_paths)[0]
            raise ValueError(
                f"Grouped split invariant violated: image_path overlap between {left_name} and {right_name}: {sample}"
            )
        overlapping_groups = normalized_groups_by_split[left_name] & normalized_groups_by_split[right_name]
        if overlapping_groups:
            sample = sorted(overlapping_groups)[0]
            raise ValueError(
                "Grouped split invariant violated: normalized group overlap between "
                f"{left_name} and {right_name}: {sample}"
            )


__all__ = ["normalize_group_id", "parent_grouped_stratified_split"]
