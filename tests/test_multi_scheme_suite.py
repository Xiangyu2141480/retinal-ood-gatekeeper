from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retinal_ood.visualization.multi_scheme_suite import (
    RunRoot,
    generate_multi_scheme_comparison_package,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _score_rows(*, score_shift: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            "image_path": "images/dissertation_v1/id/test/id_001.png",
            "label": 0,
            "ood_type": "id",
            "ood_subtype": "id",
            "score": 0.10 + score_shift,
            "prediction": 0,
            "threshold": 0.50,
        },
        {
            "image_path": "images/dissertation_v1/id/test/id_002.png",
            "label": 0,
            "ood_type": "id",
            "ood_subtype": "id",
            "score": 0.20 + score_shift,
            "prediction": 0,
            "threshold": 0.50,
        },
        {
            "image_path": "images/dissertation_v1/ood/modality_shift/colour_fundus/a.png",
            "label": 1,
            "ood_type": "modality_shift",
            "ood_subtype": "colour_fundus",
            "score": 0.82 + score_shift,
            "prediction": 1,
            "threshold": 0.50,
        },
        {
            "image_path": "images/dissertation_v1/ood/sensory_artifact/text_watermark/b.png",
            "label": 1,
            "ood_type": "sensory_artifact",
            "ood_subtype": "text_watermark",
            "score": 0.88 + score_shift,
            "prediction": 1,
            "threshold": 0.50,
        },
        {
            "image_path": "images/dissertation_v1/ood/semantic_outlier/cifar10_natural/c.png",
            "label": 1,
            "ood_type": "semantic_outlier",
            "ood_subtype": "cifar10_natural",
            "score": 0.78 + score_shift,
            "prediction": 1,
            "threshold": 0.50,
        },
    ]


def _write_eval(
    root: Path,
    run_name: str,
    *,
    model_name: str,
    layers: list[str] | None,
    auroc: float,
    score_shift: float,
) -> None:
    eval_dir = root / "runs" / run_name / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = _score_rows(score_shift=score_shift)
    pd.DataFrame(rows).to_csv(eval_dir / "scores.csv", index=False)
    _write_json(
        eval_dir / "metrics.json",
        {
            "global": {
                "auroc": auroc,
                "auprc": auroc - 0.03,
                "fpr_at_95_tpr": 1.0 - auroc,
            },
            "confusion_matrix": {"tn": 2, "fp": 0, "fn": 0, "tp": 3},
            "threshold": {
                "value": 0.50,
                "source": "validation_id_quantile",
                "quantile": 0.95,
            },
            "counts": {"total": 5, "id": 2, "ood": 3},
            "per_ood_type": {
                "modality_shift": {
                    "auroc": auroc,
                    "auprc": auroc - 0.02,
                    "fpr_at_95_tpr": 1.0 - auroc,
                    "id_count": 2,
                    "ood_count": 1,
                },
                "sensory_artifact": {
                    "auroc": auroc - 0.05,
                    "auprc": auroc - 0.06,
                    "fpr_at_95_tpr": 1.05 - auroc,
                    "id_count": 2,
                    "ood_count": 1,
                },
                "semantic_outlier": {
                    "auroc": auroc + 0.02,
                    "auprc": auroc,
                    "fpr_at_95_tpr": 0.98 - auroc,
                    "id_count": 2,
                    "ood_count": 1,
                },
            },
        },
    )
    _write_json(
        eval_dir / "resolved_evaluation_config.json",
        {
            "project": {"run_name": run_name, "seed": 42},
            "model": {
                "name": model_name,
                "backbone": "resnet50" if "patchcore" in run_name or "feature" in run_name else "",
                "layers": layers or [],
            },
            "data": {
                "train_manifest": "datasets/dissertation_v1/manifests/train_id.csv",
                "val_manifest": "datasets/dissertation_v1/manifests/val_id.csv",
                "test_id_manifest": "datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv",
                "test_ood_manifest": "datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv",
            },
        },
    )


def _write_fixture(root: Path) -> list[RunRoot]:
    primary = root / "primary"
    secondary = root / "secondary"
    runs = [
        ("image_statistics", "image_statistics", None, 0.70, -0.02),
        ("autoencoder_baseline", "conv_autoencoder", None, 0.75, -0.01),
        ("global_feature_knn", "global_feature_knn", ["layer3"], 0.82, 0.00),
        ("mahalanobis_feature", "mahalanobis_feature", ["layer3"], 0.84, 0.01),
        ("patchcore_layer3", "patchcore", ["layer3"], 0.91, 0.02),
        ("patchcore_layer2_layer3", "patchcore", ["layer2", "layer3"], 0.89, 0.03),
    ]
    for target_root, offset in [(primary, 0.0), (secondary, -0.02)]:
        for run_name, model_name, layers, auroc, shift in runs:
            _write_eval(
                target_root,
                run_name,
                model_name=model_name,
                layers=layers,
                auroc=auroc + offset,
                score_shift=shift,
            )
    return [
        RunRoot("balanced_by_subtype", primary),
        RunRoot("balanced_by_type", secondary),
    ]


def test_generate_multi_scheme_comparison_package_writes_tables_figures_and_guides(tmp_path: Path):
    outputs = generate_multi_scheme_comparison_package(
        _write_fixture(tmp_path),
        results_dir=tmp_path / "reports" / "dissertation_results" / "multi_scheme_comparison",
        figures_dir=tmp_path / "reports" / "dissertation_figures",
        dpi=80,
    )

    required_tables = {
        "scheme_overview.csv",
        "scheme_overview.md",
        "metrics_by_scheme.csv",
        "metrics_by_scheme.md",
        "metrics_by_scheme_and_eval_set.csv",
        "metrics_by_scheme_and_eval_set.md",
        "per_ood_type_by_scheme.csv",
        "per_ood_type_by_scheme.md",
        "per_ood_subtype_by_scheme.csv",
        "per_ood_subtype_by_scheme.md",
        "threshold_policy_by_scheme.csv",
        "threshold_policy_by_scheme.md",
        "model_selection_summary.csv",
        "model_selection_summary.md",
        "workload_summary.md",
    }
    required_figures = {
        "figure_scheme_comparison_matrix.png",
        "figure_metrics_by_scheme.png",
        "figure_metrics_by_scheme_and_eval_set.png",
        "figure_fpr95_by_scheme.png",
        "figure_per_ood_type_by_scheme.png",
        "figure_per_ood_subtype_by_scheme.png",
        "figure_model_selection_radar_or_table.png",
        "figure_workload_summary.png",
        "figure_methods_family_diagram.png",
        "figure_patchcore_layer_ablation.png",
        "figure_threshold_policy_comparison.png",
        "figure_failure_modes_by_scheme.png",
        "figure_selection_guide.md",
        "figure_index.md",
        "caption_suggestions.md",
    }

    assert required_tables <= {path.name for path in outputs.tables.values()}
    assert required_figures <= {path.name for path in outputs.figures.values()}

    for name in required_figures:
        if not name.endswith(".png"):
            continue
        image_path = outputs.figures[name]
        assert image_path.exists()
        width, height = Image.open(image_path).size
        assert width > 0
        assert height > 0

    overview = pd.read_csv(outputs.tables["scheme_overview.csv"])
    metrics = pd.read_csv(outputs.tables["metrics_by_scheme_and_eval_set.csv"])
    threshold = pd.read_csv(outputs.tables["threshold_policy_by_scheme.csv"])
    assert "patchcore_l1" in set(overview["scheme"])
    assert "not_completed_runtime_limited" in set(overview["completion_status"])
    assert {"balanced_by_subtype", "balanced_by_type"} <= set(metrics["eval_set"])
    assert "ood_recall_at_threshold" in metrics.columns
    assert threshold["id_false_rejection_count"].ge(0).all()

    workload_text = outputs.tables["workload_summary.md"].read_text(encoding="utf-8")
    guide_text = outputs.figures["figure_selection_guide.md"].read_text(encoding="utf-8")
    captions_text = outputs.figures["caption_suggestions.md"].read_text(encoding="utf-8")
    assert "training remained ID-only" in workload_text
    assert "OOD was evaluation-only" in workload_text
    assert "must_include" in guide_text
    assert "synthetic fallback" in captions_text
    assert "patient_id" not in workload_text + guide_text + captions_text


def test_generate_multi_scheme_cli_parses_run_roots():
    script_path = Path("scripts/generate_multi_scheme_comparison_package.py")
    spec = importlib.util.spec_from_file_location("generate_multi_scheme_comparison_package", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parsed = module._parse_run_root("balanced_by_subtype=reports/generated/run")

    assert parsed.eval_set == "balanced_by_subtype"
    assert parsed.path == Path("reports/generated/run")
    with pytest.raises(ValueError, match="eval_set=path"):
        module._parse_run_root("missing_separator")
