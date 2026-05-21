import json
from pathlib import Path

import pandas as pd
import pytest

from retinal_ood.evaluation.report_tables import (
    build_report_tables,
    dataframe_to_markdown,
    find_metrics_files,
    write_report_tables,
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
    ood_types: dict[str, dict[str, float | int]],
) -> Path:
    eval_dir = runs_dir / run_name / "evaluation"
    _write_json(
        eval_dir / "metrics.json",
        {
            "global": {
                "auroc": auroc,
                "auprc": auroc - 0.1,
                "fpr_at_95_tpr": 1.0 - auroc,
            },
            "confusion_matrix": {"tn": 8, "fp": 2, "fn": 1, "tp": 9},
            "threshold": {"value": 0.42, "source": "validation_id_quantile"},
            "counts": {"total": 20, "id": 10, "ood": 10},
            "per_ood_type": ood_types,
        },
    )
    _write_json(
        eval_dir / "resolved_evaluation_config.json",
        {
            "project": {"run_name": run_name},
            "model": {
                "name": "patchcore",
                "backbone": "resnet50",
                "layers": layers,
            },
            "data": {"notes": "patient_id must never be propagated into report tables"},
        },
    )
    return eval_dir / "metrics.json"


def test_build_report_tables_reads_metrics_and_config(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    _write_run(
        runs_dir,
        "patchcore_resnet50_layer2",
        layers=["layer2"],
        auroc=0.91,
        ood_types={
            "modality_shift": {"auroc": 0.95, "auprc": 0.93, "fpr_at_95_tpr": 0.2, "id_count": 10, "ood_count": 4},
            "semantic_outlier": {"auroc": 0.99, "auprc": 0.98, "fpr_at_95_tpr": 0.0, "id_count": 10, "ood_count": 6},
        },
    )
    _write_run(
        runs_dir,
        "patchcore_resnet50_layer3",
        layers=["layer3"],
        auroc=0.87,
        ood_types={},
    )

    tables = build_report_tables(runs_dir)

    assert list(tables.overall["run_name"]) == [
        "patchcore_resnet50_layer2",
        "patchcore_resnet50_layer3",
    ]
    assert tables.overall.loc[0, "model"] == "patchcore"
    assert tables.overall.loc[0, "backbone"] == "resnet50"
    assert tables.overall.loc[0, "layers"] == "layer2"
    assert tables.overall.loc[0, "auroc"] == pytest.approx(0.91)
    assert tables.overall.loc[0, "threshold_source"] == "validation_id_quantile"
    assert tables.overall.loc[0, "metrics_file"] == "patchcore_resnet50_layer2/evaluation/metrics.json"
    assert "patient_id" not in tables.overall.columns

    assert list(tables.per_ood["ood_type"]) == ["modality_shift", "semantic_outlier"]
    assert tables.per_ood.loc[0, "run_name"] == "patchcore_resnet50_layer2"
    assert tables.per_ood.loc[0, "ood_count"] == 4
    assert "patient_id" not in tables.per_ood.columns


def test_write_report_tables_outputs_csv_and_markdown(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    _write_run(
        runs_dir,
        "patchcore_resnet50_layer2_layer3",
        layers=["layer2", "layer3"],
        auroc=1.0,
        ood_types={
            "sensory_artifact": {"auroc": 1.0, "auprc": 1.0, "fpr_at_95_tpr": 0.0, "id_count": 10, "ood_count": 10}
        },
    )
    out_md = tmp_path / "reports" / "generated" / "experiment_summary.md"

    paths = write_report_tables(runs_dir, overall_markdown=out_md)

    assert paths.overall_markdown == out_md
    assert paths.overall_csv == out_md.with_suffix(".csv")
    assert paths.per_ood_markdown == out_md.parent / "per_category_metrics.md"
    assert paths.per_ood_csv == out_md.parent / "per_category_metrics.csv"
    for path in [
        paths.overall_markdown,
        paths.overall_csv,
        paths.per_ood_markdown,
        paths.per_ood_csv,
    ]:
        assert path.exists()

    overall = pd.read_csv(paths.overall_csv)
    assert overall.loc[0, "layers"] == "layer2+layer3"
    assert overall.loc[0, "auroc"] == pytest.approx(1.0)
    assert "patient_id" not in paths.overall_markdown.read_text(encoding="utf-8")
    assert "Per-image scores" in paths.overall_markdown.read_text(encoding="utf-8")
    assert "sensory_artifact" in paths.per_ood_markdown.read_text(encoding="utf-8")


def test_find_metrics_files_rejects_empty_runs_dir(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No metrics.json files"):
        find_metrics_files(runs_dir)


def test_dataframe_to_markdown_escapes_pipe_characters():
    markdown = dataframe_to_markdown(pd.DataFrame([{"run_name": "a|b", "auroc": 0.91234}]))

    assert "a\\|b" in markdown
    assert "0.9123" in markdown
