from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "placeholder") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_minimal_delivery_fixture(root: Path) -> None:
    _write(root / ".gitattributes", "data/images/dissertation_v1/**/*.png filter=lfs diff=lfs merge=lfs -text\n")
    _write(root / "README.md", "not a clinical device\nnot a disease classifier\n")
    _write(root / "docs/datasets/dissertation_dataset_v1.md", "dataset v1\n")
    _write(root / "docs/datasets/dissertation_dataset_card_v1.md", "synthetic ID fallback\n")
    _write(root / "docs/experiments/dissertation_results_interpretation.md", "Mahalanobis\n")
    _write(
        root / "docs/experiments/dissertation_robustness_key_findings.md",
        "Training is ID-only. OOD images are used only for evaluation. synthetic FAF fallback.\n",
    )
    _write(root / "reports/dissertation_figures/figure_index.md", "overall figure index\n")
    _write(root / "reports/dissertation_figures/robustness/figure_index.md", "robustness figure index\n")
    _write(root / "reports/dissertation_figures/robustness/top_figures_for_thesis.md", "top figures\n")
    _write(root / "reports/dissertation_figures/figure_system_pipeline_overview.png", "not a real png")
    _write(root / "reports/dissertation_figures/figure_dataset_taxonomy.png", "not a real png")
    _write(root / "reports/dissertation_figures/figure_roc_overall_model_comparison.png", "not a real png")
    _write(root / "reports/dissertation_figures/figure_pr_overall_model_comparison.png", "not a real png")
    _write(root / "reports/dissertation_figures/figure_layer_ablation_patchcore.png", "not a real png")
    _write(root / "reports/dissertation_figures/figure_per_ood_type_comparison.png", "not a real png")
    _write(
        root / "reports/dissertation_figures/figure_score_distribution_with_threshold.png",
        "not a real png",
    )
    _write(
        root / "reports/dissertation_figures/figure_heatmaps_sensory_artifact_examples.png",
        "not a real png",
    )
    _write(root / "reports/dissertation_figures/robustness/figure_threshold_policy_tradeoff.png", "not a real png")
    _write(root / "reports/dissertation_results/robustness_analysis/result_provenance.md", "ID-only training\n")
    _write(
        root / "reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.md",
        "threshold policy\n",
    )
    _write(root / "scripts/generate_multi_scheme_comparison_package.py", "script\n")
    _write(root / "scripts/generate_dissertation_robustness_analysis.py", "script\n")
    _write(root / "scripts/run_experiment_grid.py", "script\n")
    _write(root / "scripts/validate_manifests.py", "script\n")
    _write(root / "scripts/audit_dataset_images.py", "script\n")
    _write(root / "datasets/dissertation_v1/README.md", "dataset package\n")
    _write(root / "data/images/dissertation_v1/id/train/id_train_000000.png", "image")

    manifest_cols = [
        "image_path",
        "label",
        "split",
        "source",
        "ood_type",
        "ood_subtype",
        "patient_id",
    ]
    manifest_rows = [
        ["images/dissertation_v1/id/train/id_train_000000.png", 0, "train", "synthetic", "id", "id", ""],
        [
            "images/dissertation_v1/id/test_synthetic_fallback/id_test_000000.png",
            0,
            "test",
            "synthetic",
            "id",
            "id",
            "",
        ],
        [
            "images/dissertation_v1/ood/modality_shift/colour_fundus/sample.png",
            1,
            "test",
            "public",
            "modality_shift",
            "colour_fundus",
            "",
        ],
    ]
    for name in [
        "train_id.csv",
        "val_id.csv",
        "test_id_synthetic_fallback.csv",
        "test_ood_full.csv",
        "test_ood_balanced_by_subtype.csv",
    ]:
        manifest_path = root / "datasets/dissertation_v1/manifests" / name
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest_rows, columns=manifest_cols).to_csv(manifest_path, index=False)

    _to_csv(
        pd.DataFrame(
        [
            {
                "scheme": "mahalanobis_feature",
                "scheme_label": "Mahalanobis feature",
                "auroc": 0.9724,
                "auprc": 0.9975,
                "fpr_at_95_tpr": 0.2,
                "ood_recall_at_threshold": 0.9079,
                "id_count": 150,
                "ood_count": 1650,
            },
            {
                "scheme": "patchcore_l3",
                "scheme_label": "PatchCore L3",
                "auroc": 0.8819,
                "auprc": 0.9880,
                "fpr_at_95_tpr": 0.6333,
                "ood_recall_at_threshold": 0.7097,
                "id_count": 150,
                "ood_count": 1650,
            },
        ]
        ),
        root / "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv",
    )
    _to_csv(
        pd.DataFrame(
            [
                {"scheme": "mahalanobis_feature", "completion_status": "completed"},
                {"scheme": "patchcore_l3", "completion_status": "completed"},
                {"scheme": "patchcore_l1", "completion_status": "not_completed_runtime_limited"},
            ]
        ),
        root / "reports/dissertation_results/multi_scheme_comparison/scheme_overview.csv",
    )
    _to_csv(
        pd.DataFrame(
            [
                {"experiment": "patchcore_layer3", "model": "patchcore", "layers": "layer3", "auroc": 0.8819}
            ]
        ),
        root / "reports/dissertation_results/primary_balanced_by_subtype_10k/metrics_summary.csv",
    )
    _to_csv(
        pd.DataFrame(
            [
                {
                    "scheme": "mahalanobis_feature",
                    "scheme_label": "Mahalanobis feature",
                    "metric": "auroc",
                    "estimate": 0.9724,
                    "ci_low": 0.9638,
                    "ci_high": 0.9800,
                }
            ]
        ),
        root / "reports/dissertation_results/robustness_analysis/bootstrap_ci.csv",
    )
    _to_csv(
        pd.DataFrame(
            [
                {
                    "scheme": "mahalanobis_feature",
                    "scheme_label": "Mahalanobis feature",
                    "policy": "val_id_quantile_95",
                    "policy_kind": "deployment",
                    "id_false_rejection_rate": 0.0467,
                    "ood_recall": 0.9079,
                }
            ]
        ),
        root / "reports/dissertation_results/robustness_analysis/threshold_policy_sweep.csv",
    )
    _to_csv(
        pd.DataFrame(
            [
                {
                    "scheme": "mahalanobis_feature",
                    "scheme_label": "Mahalanobis feature",
                    "ood_subtype": "text_watermark",
                    "analysis": "only_subtype",
                    "auroc": 0.7361,
                }
            ]
        ),
        root / "reports/dissertation_results/robustness_analysis/subtype_influence.csv",
    )
    _to_csv(
        pd.DataFrame(
        [
            {
                "analysis": "threshold policy",
                "main finding": "ID-calibrated Mahalanobis thresholds give the best trade-off.",
                "strongest method": "Mahalanobis feature",
                "key metric/result": "val_id_quantile_95: OOD recall 0.9079",
                "dissertation implication": "Use ID validation calibration.",
                "limitation": "Synthetic fallback ID.",
            }
        ]
        ),
        root / "reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.csv",
    )


def test_delivery_bundle_builder_writes_final_indexes_from_existing_outputs(tmp_path: Path):
    module = _load_script(
        "scripts/build_dissertation_delivery_bundle.py",
        "build_dissertation_delivery_bundle",
    )
    _write_minimal_delivery_fixture(tmp_path)

    outputs = module.build_delivery_bundle(
        repo_root=tmp_path,
        out_dir=tmp_path / "reports/dissertation_final",
        strict=True,
    )

    expected = {
        "final_result_summary_md",
        "final_result_summary_csv",
        "final_figure_index",
        "final_table_index",
        "final_reproducibility_checklist",
        "final_claim_evidence_index",
        "final_appendix_index",
    }
    assert expected == set(outputs)
    for path in outputs.values():
        assert path.exists()

    summary = outputs["final_result_summary_md"].read_text(encoding="utf-8")
    assert "Mahalanobis feature" in summary
    assert "PatchCore L3" in summary
    assert "synthetic ID fallback" in summary
    assert "not a disease classifier" in summary

    figure_index = outputs["final_figure_index"].read_text(encoding="utf-8")
    assert "figure_system_pipeline_overview.png" in figure_index
    assert "figure_threshold_policy_tradeoff.png" in figure_index


def test_delivery_bundle_strict_mode_fails_when_required_evidence_is_missing(tmp_path: Path):
    module = _load_script(
        "scripts/build_dissertation_delivery_bundle.py",
        "build_dissertation_delivery_bundle",
    )
    _write_minimal_delivery_fixture(tmp_path)
    (tmp_path / "reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv").unlink()

    with pytest.raises(FileNotFoundError, match="metrics_by_scheme.csv"):
        module.build_delivery_bundle(
            repo_root=tmp_path,
            out_dir=tmp_path / "reports/dissertation_final",
            strict=True,
        )


def test_final_repository_audit_passes_clean_fixture_and_fails_private_path(tmp_path: Path):
    module = _load_script("scripts/final_repository_audit.py", "final_repository_audit")
    _write_minimal_delivery_fixture(tmp_path)
    (tmp_path / "reports/dissertation_final").mkdir(parents=True)
    for name in [
        "final_result_summary.md",
        "final_result_summary.csv",
        "final_figure_index.md",
        "final_table_index.md",
        "final_reproducibility_checklist.md",
        "final_claim_evidence_index.md",
        "final_appendix_index.md",
    ]:
        _write(tmp_path / "reports/dissertation_final" / name, "final\n")
    for name in [
        "final_evidence_index.md",
        "final_figure_shortlist.md",
        "final_table_shortlist.md",
        "reproducibility_runbook.md",
        "school_server_runbook.md",
        "claims_and_limitations_matrix.md",
        "README.md",
    ]:
        _write(tmp_path / "docs/dissertation" / name, "not a clinical device\nnot a disease classifier\n")

    clean = module.run_audit(repo_root=tmp_path)

    assert clean.passed, clean.failures

    private_path = "C:" + "\\Users\\someone\\private\\data"
    _write(tmp_path / "docs/dissertation/private_note.md", private_path)
    dirty = module.run_audit(repo_root=tmp_path)

    assert not dirty.passed
    assert any("private path" in failure.lower() for failure in dirty.failures)


def test_final_repository_audit_cli_writes_json(tmp_path: Path):
    module = _load_script("scripts/final_repository_audit.py", "final_repository_audit")
    _write_minimal_delivery_fixture(tmp_path)
    (tmp_path / "reports/dissertation_final").mkdir(parents=True)
    for name in [
        "final_result_summary.md",
        "final_result_summary.csv",
        "final_figure_index.md",
        "final_table_index.md",
        "final_reproducibility_checklist.md",
        "final_claim_evidence_index.md",
        "final_appendix_index.md",
    ]:
        _write(tmp_path / "reports/dissertation_final" / name, "final\n")
    for name in [
        "final_evidence_index.md",
        "final_figure_shortlist.md",
        "final_table_shortlist.md",
        "reproducibility_runbook.md",
        "school_server_runbook.md",
        "claims_and_limitations_matrix.md",
        "README.md",
    ]:
        _write(tmp_path / "docs/dissertation" / name, "not a clinical device\nnot a disease classifier\n")
    out_json = tmp_path / "audit.json"

    result = module.run_audit(repo_root=tmp_path, write_json=out_json)

    assert result.passed
    assert json.loads(out_json.read_text(encoding="utf-8"))["passed"] is True


def test_final_repository_audit_fails_missing_referenced_final_index_path(tmp_path: Path):
    module = _load_script("scripts/final_repository_audit.py", "final_repository_audit")
    _write_minimal_delivery_fixture(tmp_path)
    (tmp_path / "reports/dissertation_final").mkdir(parents=True)
    for name in [
        "final_result_summary.md",
        "final_result_summary.csv",
        "final_table_index.md",
        "final_reproducibility_checklist.md",
        "final_claim_evidence_index.md",
        "final_appendix_index.md",
    ]:
        _write(tmp_path / "reports/dissertation_final" / name, "final\n")
    _write(
        tmp_path / "reports/dissertation_final/final_figure_index.md",
        "| path | title |\n| --- | --- |\n| reports/dissertation_figures/missing.png | Missing |\n",
    )
    for name in [
        "final_evidence_index.md",
        "final_figure_shortlist.md",
        "final_table_shortlist.md",
        "reproducibility_runbook.md",
        "school_server_runbook.md",
        "claims_and_limitations_matrix.md",
        "README.md",
    ]:
        _write(tmp_path / "docs/dissertation" / name, "not a clinical device\nnot a disease classifier\n")

    result = module.run_audit(repo_root=tmp_path)

    assert not result.passed
    assert any("referenced path missing" in failure for failure in result.failures)
