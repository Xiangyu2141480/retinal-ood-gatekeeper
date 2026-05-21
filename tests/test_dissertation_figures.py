import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.evaluation.report_tables import ReportTables
from retinal_ood.visualization.dissertation_figures import (
    generate_dissertation_figures,
    save_dissertation_figures,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run(
    runs_dir: Path,
    run_name: str,
    *,
    layers: list[str],
    auroc: float,
    fpr: float,
    ood_types: dict[str, dict[str, float | int]],
) -> None:
    eval_dir = runs_dir / run_name / "evaluation"
    _write_json(
        eval_dir / "metrics.json",
        {
            "global": {
                "auroc": auroc,
                "auprc": auroc - 0.05,
                "fpr_at_95_tpr": fpr,
            },
            "confusion_matrix": {"tn": 7, "fp": 3, "fn": 1, "tp": 9},
            "threshold": {"value": 0.55, "source": "validation_id_quantile"},
            "counts": {"total": 20, "id": 10, "ood": 10},
            "per_ood_type": ood_types,
        },
    )
    _write_json(
        eval_dir / "resolved_evaluation_config.json",
        {
            "project": {"run_name": run_name},
            "model": {"name": "patchcore", "backbone": "resnet50", "layers": layers},
            "data": {"notes": "patient_id should not appear in figure manifest"},
        },
    )


def test_generate_dissertation_figures_writes_pngs_and_manifest(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    _write_run(
        runs_dir,
        "patchcore_resnet50_layer2",
        layers=["layer2"],
        auroc=0.91,
        fpr=0.22,
        ood_types={
            "modality_shift": {"auroc": 0.94, "auprc": 0.92, "fpr_at_95_tpr": 0.2, "id_count": 10, "ood_count": 5},
            "semantic_outlier": {"auroc": 0.97, "auprc": 0.96, "fpr_at_95_tpr": 0.1, "id_count": 10, "ood_count": 5},
        },
    )
    _write_run(
        runs_dir,
        "patchcore_resnet50_layer3",
        layers=["layer3"],
        auroc=0.86,
        fpr=0.35,
        ood_types={
            "modality_shift": {"auroc": 0.88, "auprc": 0.85, "fpr_at_95_tpr": 0.3, "id_count": 10, "ood_count": 5},
            "semantic_outlier": {"auroc": 0.9, "auprc": 0.87, "fpr_at_95_tpr": 0.25, "id_count": 10, "ood_count": 5},
        },
    )

    outputs = generate_dissertation_figures(runs_dir, tmp_path / "figures", dpi=80)

    assert set(outputs.figures) == {
        "global_metrics",
        "fpr_at_95_tpr",
        "layer_ablation",
        "per_ood_auroc",
        "confusion_matrices",
    }
    for path in outputs.figures.values():
        assert path.exists()
        assert path.stat().st_size > 0
        assert Image.open(path).size[0] > 0

    manifest_text = outputs.manifest.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["run_count"] == 2
    assert manifest["ood_category_count"] == 2
    assert "patient_id" not in manifest_text
    assert "per-image scores" in manifest["note"].lower()


def test_save_dissertation_figures_skips_per_ood_when_no_category_metrics(tmp_path: Path):
    tables = ReportTables(
        overall=pd.DataFrame(
            [
                {
                    "run_name": "toy_layer1",
                    "layers": "layer1",
                    "auroc": 0.8,
                    "auprc": 0.7,
                    "fpr_at_95_tpr": 0.4,
                    "tn": 4,
                    "fp": 1,
                    "fn": 2,
                    "tp": 3,
                }
            ]
        ),
        per_ood=pd.DataFrame(),
    )

    outputs = save_dissertation_figures(tables, tmp_path / "figures", dpi=80)

    assert "per_ood_auroc" not in outputs.figures
    assert outputs.manifest.exists()
    assert json.loads(outputs.manifest.read_text())["ood_category_count"] == 0


def test_save_dissertation_figures_rejects_missing_required_columns(tmp_path: Path):
    tables = ReportTables(overall=pd.DataFrame([{"run_name": "broken"}]), per_ood=pd.DataFrame())

    with pytest.raises(ValueError, match="Missing required columns"):
        save_dissertation_figures(tables, tmp_path / "figures")
