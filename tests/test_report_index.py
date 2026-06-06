from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path("scripts/generate_report_index.py")
    spec = importlib.util.spec_from_file_location("generate_report_index", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_grid_outputs(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "experiment": "patchcore_layer1",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer1",
                "auroc": 0.82,
                "auprc": 0.78,
                "fpr_at_95_tpr": 0.36,
                "threshold_at_95_tpr": 0.71,
                "threshold": 0.55,
                "threshold_source": "validation_id_quantile",
                "id_false_rejection_rate": 0.1,
                "id_count": 10,
                "ood_count": 20,
                "warnings": "",
                "metrics_file": "runs/patchcore_layer1/evaluation/metrics.json",
            },
            {
                "experiment": "patchcore_layer2_layer3",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer2+layer3",
                "auroc": 0.94,
                "auprc": 0.91,
                "fpr_at_95_tpr": 0.12,
                "threshold_at_95_tpr": 0.67,
                "threshold": 0.5,
                "threshold_source": "validation_id_quantile",
                "id_false_rejection_rate": 0.05,
                "id_count": 10,
                "ood_count": 20,
                "warnings": "test_real_id.csv synthetic fallback",
                "metrics_file": "runs/patchcore_layer2_layer3/evaluation/metrics.json",
            },
            {
                "experiment": "autoencoder_baseline",
                "model": "conv_autoencoder",
                "backbone": "",
                "layers": "",
                "auroc": 0.73,
                "auprc": 0.7,
                "fpr_at_95_tpr": 0.48,
                "threshold_at_95_tpr": 0.62,
                "threshold": 0.44,
                "threshold_source": "test_id_quantile",
                "id_false_rejection_rate": 0.2,
                "id_count": 10,
                "ood_count": 20,
                "warnings": "",
                "metrics_file": "runs/autoencoder_baseline/evaluation/metrics.json",
            },
        ]
    ).to_csv(reports_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": "patchcore_layer2_layer3",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer2+layer3",
                "ood_type": "modality_shift",
                "auroc": 0.95,
                "auprc": 0.93,
                "fpr_at_95_tpr": 0.1,
                "id_count": 10,
                "ood_count": 11,
                "metrics_file": "runs/patchcore_layer2_layer3/evaluation/metrics.json",
            },
            {
                "experiment": "patchcore_layer2_layer3",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer2+layer3",
                "ood_type": "semantic_outlier",
                "auroc": 0.98,
                "auprc": 0.97,
                "fpr_at_95_tpr": 0.05,
                "id_count": 10,
                "ood_count": 9,
                "metrics_file": "runs/patchcore_layer2_layer3/evaluation/metrics.json",
            },
        ]
    ).to_csv(reports_dir / "per_ood_type_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": "patchcore_layer2_layer3",
                "model": "patchcore",
                "backbone": "resnet50",
                "layers": "layer2+layer3",
                "ood_type": "modality_shift",
                "ood_subtype": "colour_fundus",
                "auroc": 0.95,
                "auprc": 0.93,
                "fpr_at_95_tpr": 0.1,
                "id_count": 10,
                "ood_count": 11,
                "metrics_file": "runs/patchcore_layer2_layer3/evaluation/metrics.json",
            }
        ]
    ).to_csv(reports_dir / "per_ood_subtype_metrics.csv", index=False)
    _write_json(
        reports_dir / "manifest_audit.json",
        {
            "warnings": [
                "data/manifests/test_real_id.csv: filename suggests real ID validation, "
                "but all sources look synthetic; treat this as a synthetic fallback"
            ],
            "totals": {"rows": 30},
        },
    )
    _write_run_outputs(reports_dir, "patchcore_layer2_layer3")


def _write_run_outputs(reports_dir: Path, run_name: str) -> None:
    eval_dir = reports_dir / "runs" / run_name / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "image_path": "images/id/fp.png",
                "label": 0,
                "ood_type": "id",
                "score": 0.8,
                "prediction": 1,
                "threshold": 0.5,
            },
            {
                "image_path": "images/ood/tp.png",
                "label": 1,
                "ood_type": "modality_shift",
                "score": 0.9,
                "prediction": 1,
                "threshold": 0.5,
            },
            {
                "image_path": "images/ood/fn.png",
                "label": 1,
                "ood_type": "semantic_outlier",
                "score": 0.2,
                "prediction": 0,
                "threshold": 0.5,
            },
            {
                "image_path": "images/id/borderline.png",
                "label": 0,
                "ood_type": "id",
                "score": 0.49,
                "prediction": 0,
                "threshold": 0.5,
            },
        ]
    ).to_csv(eval_dir / "scores.csv", index=False)
    _write_json(
        eval_dir / "metrics.json",
        {
            "threshold": {"value": 0.5, "source": "validation_id_quantile"},
        },
    )
    heatmap_dir = eval_dir / "heatmaps"
    heatmap_dir.mkdir()
    pd.DataFrame(
        [
            {
                "outcome": "tp",
                "rank": 1,
                "sample_index": 1,
                "image_path": "images/ood/tp.png",
                "label": 1,
                "score": 0.9,
                "prediction": 1,
                "threshold": 0.5,
                "original_file": "tp/tp_original.png",
                "heatmap_file": "tp/tp_heatmap.png",
                "overlay_file": "tp/tp_overlay.png",
            }
        ]
    ).to_csv(heatmap_dir / "heatmap_manifest.csv", index=False)


def test_generate_report_index_writes_dissertation_tables(tmp_path: Path):
    from retinal_ood.evaluation.report_index import generate_report_index

    reports_dir = tmp_path / "reports" / "generated"
    _write_grid_outputs(reports_dir)

    outputs = generate_report_index(reports_dir, top_k=1)

    expected = {
        "index",
        "per_category_metrics",
        "layer_ablation_table",
        "ae_vs_patchcore_table",
        "threshold_policy_table",
        "heatmap_index",
        "case_selection",
    }
    assert expected.issubset(outputs.files)
    for path in outputs.files.values():
        assert path.exists()

    index_text = outputs.files["index"].read_text(encoding="utf-8")
    assert "research_threshold" in index_text
    assert "computed using test labels for paper evaluation only" in index_text
    assert "deployment_threshold" in index_text
    assert "computed from val ID score quantile only" in index_text
    assert "synthetic fallback" in index_text

    layer_table = pd.read_csv(outputs.files["layer_ablation_table"].with_suffix(".csv"))
    assert list(layer_table["experiment"]) == ["patchcore_layer1", "patchcore_layer2_layer3"]

    method_table = pd.read_csv(outputs.files["ae_vs_patchcore_table"].with_suffix(".csv"))
    assert set(method_table["method"]) == {"Autoencoder", "PatchCore"}

    threshold_table = pd.read_csv(outputs.files["threshold_policy_table"].with_suffix(".csv"))
    assert set(threshold_table["threshold_policy"]) == {"research_threshold", "deployment_threshold"}
    assert "research only" in threshold_table["warning"].str.cat(sep=" ").lower()

    case_table = pd.read_csv(outputs.files["case_selection"].with_suffix(".csv"))
    assert {"tp", "fp", "fn", "borderline"} <= set(case_table["case_type"])

    heatmap_table = pd.read_csv(outputs.files["heatmap_index"].with_suffix(".csv"))
    assert heatmap_table.loc[0, "overlay_file"].endswith("tp_overlay.png")


def test_script_entrypoint_calls_report_index_generator(tmp_path: Path):
    module = _load_script_module()
    reports_dir = tmp_path / "reports" / "generated"
    _write_grid_outputs(reports_dir)

    outputs = module.generate_report_index(reports_dir=reports_dir, top_k=2)

    assert outputs.files["index"] == reports_dir / "index.md"
    assert (reports_dir / "case_selection.csv").exists()


def test_generate_report_index_handles_missing_heatmaps(tmp_path: Path):
    from retinal_ood.evaluation.report_index import generate_report_index

    reports_dir = tmp_path / "reports" / "generated"
    _write_grid_outputs(reports_dir)
    for heatmap_manifest in reports_dir.rglob("heatmap_manifest.csv"):
        heatmap_manifest.unlink()

    outputs = generate_report_index(reports_dir)

    heatmap_md = outputs.files["heatmap_index"].read_text(encoding="utf-8")
    assert "No heatmap_manifest.csv files were found" in heatmap_md
    assert pd.read_csv(outputs.files["heatmap_index"].with_suffix(".csv")).empty
