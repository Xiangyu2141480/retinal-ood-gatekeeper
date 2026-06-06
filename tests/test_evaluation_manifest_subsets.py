from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_script_module():
    script_path = Path("scripts/build_evaluation_manifests.py")
    spec = importlib.util.spec_from_file_location("build_evaluation_manifests", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _ood_row(
    idx: int,
    *,
    ood_type: str,
    ood_subtype: str,
    source: str = "public_ood",
    split: str = "test",
) -> dict[str, object]:
    return {
        "image_path": f"images/ood/{ood_subtype}_{idx}.png",
        "label": 1,
        "split": split,
        "source": source,
        "ood_type": ood_type,
        "ood_subtype": ood_subtype,
        "source_split": "",
        "source_image_hash": f"hash-{idx}",
        "patient_id": "",
        "scanner": "",
        "notes": "",
    }


def _id_row(idx: int, *, source: str = "synthetic_faf") -> dict[str, object]:
    return {
        "image_path": f"images/id/{idx}.png",
        "label": 0,
        "split": "test",
        "source": source,
        "ood_type": "id",
        "ood_subtype": "",
        "patient_id": "",
        "scanner": "synthetic_heidelberg",
        "notes": "synthetic fallback",
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    test_ood = _write_manifest(
        tmp_path / "test_ood.csv",
        [
            *[_ood_row(i, ood_type="modality_shift", ood_subtype="colour_fundus") for i in range(3)],
            *[_ood_row(10 + i, ood_type="sensory_artifact", ood_subtype="text_watermark") for i in range(2)],
            *[_ood_row(20 + i, ood_type="semantic_outlier", ood_subtype="natural_cifar10") for i in range(4)],
        ],
    )
    test_id = _write_manifest(tmp_path / "test_real_id.csv", [_id_row(1), _id_row(2)])
    return test_ood, test_id


def test_subset_builder_writes_deterministic_balanced_and_smoke_outputs(tmp_path: Path):
    module = _load_script_module()
    test_ood, test_id = _write_inputs(tmp_path)

    result = module.build_evaluation_manifests(
        test_ood_manifest=test_ood,
        out_dir=tmp_path / "generated",
        test_id_manifest=test_id,
        seed=7,
        smoke_per_type=1,
    )
    result_again = module.build_evaluation_manifests(
        test_ood_manifest=test_ood,
        out_dir=tmp_path / "generated_again",
        test_id_manifest=test_id,
        seed=7,
        smoke_per_type=1,
    )

    full = pd.read_csv(result["test_ood_full"])
    balanced_type = pd.read_csv(result["test_ood_balanced_by_type"])
    balanced_subtype = pd.read_csv(result["test_ood_balanced_by_subtype"])
    smoke = pd.read_csv(result["test_ood_smoke"])
    smoke_again = pd.read_csv(result_again["test_ood_smoke"])

    assert len(full) == 9
    assert set(full.columns) >= {"source", "ood_type", "ood_subtype", "source_split", "source_image_hash"}
    assert balanced_type["ood_type"].value_counts().to_dict() == {
        "modality_shift": 2,
        "sensory_artifact": 2,
        "semantic_outlier": 2,
    }
    assert balanced_subtype["ood_subtype"].value_counts().to_dict() == {
        "colour_fundus": 2,
        "text_watermark": 2,
        "natural_cifar10": 2,
    }
    assert set(smoke["ood_type"]) == {"modality_shift", "sensory_artifact", "semantic_outlier"}
    assert smoke["image_path"].tolist() == smoke_again["image_path"].tolist()
    assert all(smoke["split"] == "test")
    assert all(smoke["label"] == 1)

    audit = json.loads(Path(result["audit_json"]).read_text(encoding="utf-8"))
    assert audit["seed"] == 7
    assert "synthetic fallback" in " ".join(audit["warnings"])
    assert Path(result["test_id_synthetic_fallback"]).exists()


def test_subset_builder_rejects_ood_train_rows(tmp_path: Path):
    module = _load_script_module()
    test_ood = _write_manifest(
        tmp_path / "test_ood.csv",
        [_ood_row(1, ood_type="modality_shift", ood_subtype="colour_fundus", split="train")],
    )

    with pytest.raises(ValueError, match="must not contain split=train"):
        module.build_evaluation_manifests(test_ood_manifest=test_ood, out_dir=tmp_path / "generated")


def test_subset_builder_fails_without_subtype_for_balancing(tmp_path: Path):
    module = _load_script_module()
    test_ood = _write_manifest(
        tmp_path / "test_ood.csv",
        [_ood_row(1, ood_type="modality_shift", ood_subtype="")],
    )

    with pytest.raises(ValueError, match="ood_subtype"):
        module.build_evaluation_manifests(test_ood_manifest=test_ood, out_dir=tmp_path / "generated")
