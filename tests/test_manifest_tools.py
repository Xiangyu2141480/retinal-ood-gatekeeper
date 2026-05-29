from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.data.manifest_tools import build_ood_manifest, merge_manifest_files


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=(80, 80, 80)).save(path)


def test_build_ood_manifest_uses_literature_review_categories(tmp_path: Path):
    root_dir = tmp_path / "data"
    _write_image(root_dir / "images" / "ood_modality" / "colour_fundus" / "a.png")
    _write_image(root_dir / "images" / "ood_semantic" / "natural" / "b.jpg")

    df = build_ood_manifest(
        root_dir=root_dir,
        mappings=[
            ("images/ood_modality/colour_fundus", "modality_shift"),
            ("images/ood_semantic/natural", "semantic_outlier"),
        ],
        out_manifest=root_dir / "manifests" / "test_ood_parts.csv",
    )

    assert len(df) == 2
    assert list(df.columns) == [
        "image_path",
        "label",
        "split",
        "source",
        "ood_type",
        "patient_id",
        "scanner",
        "notes",
    ]
    assert set(df["label"]) == {1}
    assert set(df["split"]) == {"test"}
    assert set(df["source"]) == {"public_ood"}
    assert set(df["ood_type"]) == {"modality_shift", "semantic_outlier"}
    assert all(path.startswith("images/") for path in df["image_path"])
    assert (root_dir / "manifests" / "test_ood_parts.csv").exists()


def test_build_ood_manifest_rejects_empty_folder(tmp_path: Path):
    root_dir = tmp_path / "data"
    (root_dir / "images" / "ood_modality" / "infrared").mkdir(parents=True)

    with pytest.raises(ValueError, match="No supported image files"):
        build_ood_manifest(
            root_dir=root_dir,
            mappings=[("images/ood_modality/infrared", "modality_shift")],
            out_manifest=root_dir / "manifests" / "test_modality.csv",
        )


def test_merge_manifest_files_preserves_required_columns(tmp_path: Path):
    manifest_a = tmp_path / "a.csv"
    manifest_b = tmp_path / "b.csv"
    rows = [
        {
            "image_path": "images/ood_modality/a.png",
            "label": 1,
            "split": "test",
            "source": "public_ood",
            "ood_type": "modality_shift",
            "patient_id": "",
            "scanner": "",
            "notes": "colour fundus",
        }
    ]
    pd.DataFrame(rows).to_csv(manifest_a, index=False)
    rows[0]["image_path"] = "images/ood_semantic/b.png"
    rows[0]["ood_type"] = "semantic_outlier"
    pd.DataFrame(rows).to_csv(manifest_b, index=False)

    out = tmp_path / "merged.csv"
    merged = merge_manifest_files([manifest_a, manifest_b], out)

    assert len(merged) == 2
    assert list(merged["ood_type"]) == ["modality_shift", "semantic_outlier"]
    assert list(pd.read_csv(out).columns) == [
        "image_path",
        "label",
        "split",
        "source",
        "ood_type",
        "patient_id",
        "scanner",
        "notes",
    ]


def test_manifest_tool_scripts_help_loads():
    import importlib.util

    for script_name in ["build_ood_manifest.py", "merge_manifests.py"]:
        script_path = Path("scripts") / script_name
        spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
