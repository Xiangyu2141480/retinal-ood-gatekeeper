import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from retinal_ood.visualization.heatmaps import (
    HeatmapNormalization,
    normalize_map,
    resize_anomaly_map,
    save_heatmap_artifacts,
    save_topk_heatmaps,
    select_top_k_outcomes,
)


def _write_toy_image(path: Path, color: tuple[int, int, int] = (16, 64, 128)) -> None:
    Image.new("RGB", (8, 6), color=color).save(path)


def test_resize_anomaly_map_matches_image_size():
    anomaly_map = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    resized = resize_anomaly_map(anomaly_map, (8, 6))

    assert resized.shape == (6, 8)


def test_normalize_map_uses_run_level_limits():
    anomaly_map = np.array([[0.0, 5.0]], dtype=np.float32)

    normalized = normalize_map(
        anomaly_map,
        normalization=HeatmapNormalization(vmin=0.0, vmax=10.0),
    )

    np.testing.assert_allclose(normalized, np.array([[0.0, 0.5]], dtype=np.float32))


def test_save_heatmap_artifacts_creates_aligned_pngs(tmp_path: Path):
    image_path = tmp_path / "input.png"
    _write_toy_image(image_path)

    paths = save_heatmap_artifacts(
        image_path,
        np.array([[0.0, 1.0], [0.5, 0.2]], dtype=np.float32),
        tmp_path / "heatmaps",
        stem="toy",
    )

    for path in paths.values():
        assert path.exists()
        assert Image.open(path).size == (8, 6)


def test_select_top_k_outcomes_ranks_by_confidence():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.05])

    selected = select_top_k_outcomes(labels, scores, threshold=0.5, top_k=1)

    assert selected == {"fp": [1], "fn": [5], "tp": [3], "tn": [0]}


def test_save_topk_heatmaps_writes_files_and_privacy_manifest(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    metadata_rows = []
    for idx in range(4):
        image_path = image_dir / f"sample_{idx}.png"
        _write_toy_image(image_path, color=(idx * 20, 64, 128))
        metadata_rows.append(
            {
                "image_path": f"images/sample_{idx}.png",
                "resolved_image_path": str(image_path),
                "patient_id": "not-for-output",
            }
        )
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.9])
    patch_maps = [
        np.full((2, 2), value, dtype=np.float32)
        for value in [0.1, 0.8, 0.2, 0.9]
    ]

    rows = save_topk_heatmaps(
        metadata_rows,
        patch_maps,
        labels,
        scores,
        threshold=0.5,
        out_dir=tmp_path / "run_heatmaps",
        top_k=1,
    )

    out_dir = tmp_path / "run_heatmaps"
    assert {row["outcome"] for row in rows} == {"fp", "fn", "tp", "tn"}
    assert (out_dir / "colorbar.png").exists()
    assert (out_dir / "heatmap_normalization.json").exists()
    assert json.loads((out_dir / "heatmap_normalization.json").read_text())["selected_count"] == 4

    manifest_path = out_dir / "heatmap_manifest.csv"
    assert manifest_path.exists()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == 4
    assert "patient_id" not in manifest_rows[0]
    for row in manifest_rows:
        assert not Path(row["overlay_file"]).is_absolute()
        assert (out_dir / row["original_file"]).exists()
        assert (out_dir / row["heatmap_file"]).exists()
        assert (out_dir / row["overlay_file"]).exists()


def test_save_topk_heatmaps_rejects_missing_patch_maps(tmp_path: Path):
    with pytest.raises(ValueError, match="same length"):
        save_topk_heatmaps(
            [{"image_path": "x.png"}],
            [],
            np.array([0]),
            np.array([0.1]),
            threshold=0.5,
            out_dir=tmp_path,
        )
