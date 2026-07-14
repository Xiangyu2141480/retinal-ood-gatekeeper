# Dissertation Handoff README

This directory is the examiner-friendly entry point for the final dissertation delivery bundle.

## Start Here

The project overview provides the research question, strategy, dataset, experiments, main
results, selected figures, conclusions, and limitations in one location.

- **Supervisor / project overview:** [`project_overview.md`](project_overview.md)
- **Progress log:** [`progress_log.md`](progress_log.md)
- **Final evidence index:** [`final_evidence_index.md`](final_evidence_index.md)
- **Final figure shortlist:** [`final_figure_shortlist.md`](final_figure_shortlist.md)
- **Reproducibility guide:** [`reproducibility_runbook.md`](reproducibility_runbook.md)

## Project Summary

The repository implements an unsupervised binary OOD gatekeeper for retinal Fundus Autofluorescence (FAF) image quality control:

`input image -> OOD gatekeeper -> ACCEPT valid FAF / REJECT invalid or OOD`

It is not a disease classifier, not a clinical device, and not clinical deployment validation. The optional Phase 2 module runs only after a rejection and assigns likely rejection explanations, not clinical diagnoses.

The completed system contains:

1. Stage 1 ID-only OOD gatekeeper.
2. Multi-scheme OOD model comparison.
3. Robustness and failure analysis.
4. Optional supervised Stage 2 rejected-input reason attribution after Stage 1 rejection.
5. Reproducible dataset and Git LFS package.
6. Dissertation-ready figures and evidence index.

## Main Findings

- Stage 1 best quantitative method: Mahalanobis feature distance.
- Stage 1 best localization-oriented method: PatchCore L3.
- Stage 2 best family method on the final grouped protocol: `feature_statistics_fusion`.
- Stage 2 best subtype method: `hierarchical_classifier`.
- Stage 2 is optional post-rejection explanation only; reason labels are likely explanations, not clinical diagnoses.

## Repository Map

| area | location |
| --- | --- |
| Dataset package | `datasets/dissertation_v1/` |
| Stage 1 results | `reports/dissertation_results/multi_scheme_comparison/` |
| Robustness and failure analysis | `reports/dissertation_results/robustness_analysis/` |
| Final Stage 2 grouped results | `reports/stage2_grouped/` |
| Legacy Stage 2 row-level results | `reports/dissertation_results/reason_attribution_method_comparison/` |
| Final figures | `reports/dissertation_figures/` |
| Final polish tables and captions | `reports/dissertation_final/` |
| Manuscript draft blocks | `docs/dissertation/manuscript_draft/` |

## Recommended Figures And Tables

- Main figure shortlist: `docs/dissertation/final_figure_shortlist.md`
- Main table shortlist: `docs/dissertation/final_table_shortlist.md`
- Polished caption suggestions: `reports/dissertation_final/caption_suggestions.md`
- Figure interpretation notes: `reports/dissertation_final/interpretation_notes.md`
- Recommended methods: `reports/dissertation_final/recommended_model_summary.md`
- Claims and evidence: `reports/dissertation_final/claim_evidence_matrix.md`

## Dataset Summary

Dataset v1 is packaged under `datasets/dissertation_v1/` with Git LFS-tracked images under `data/images/dissertation_v1/`.

Core manifests:

- `train_id.csv`: 700 ID-only training examples.
- `val_id.csv`: 150 ID validation examples for threshold calibration.
- `test_id_synthetic_fallback.csv`: 150 synthetic fallback ID examples.
- `test_ood_full.csv`: 2100 OOD examples.
- `test_ood_balanced_by_subtype.csv`: 1650 OOD examples for balanced evaluation.

The synthetic fallback ID split is not real clinical FAF validation.

## Final Experiment Summary

The final evidence package includes:

- Dataset v1 package from PR #20.
- Multi-scheme comparison and dissertation figure package from PR #21.
- Robustness, threshold-safety, and failure-analysis package from PR #22.
- Optional rejected-input reason attribution and method comparison from PR #24/#25.
- Final delivery bundle in this PR.

Main result: Mahalanobis feature distance is the strongest quantitative gatekeeper on dataset v1. PatchCore L3 remains useful as a heatmap/localization companion.

Optional Phase 2 result: on the final parent-grouped split,
`feature_statistics_fusion` is selected by validation family macro-F1
(0.9924836011792534) and reaches grouped test accuracy/macro-F1 1.0000/1.0000.
The non-oracle `hierarchical_classifier` is selected by validation subtype macro-F1
(0.9058507271445498) and reaches grouped test accuracy 0.9357 and macro-F1 0.9176.

## Where To Find Final Figures

- Main figure shortlist: `docs/dissertation/final_figure_shortlist.md`
- Generated final figure index: `reports/dissertation_final/final_figure_index.md`
- Polished figure inventory: `reports/dissertation_final/polished_figure_inventory.md`
- Existing figure directory: `reports/dissertation_figures/`
- Final grouped Phase 2 figures: `reports/dissertation_figures/stage2_grouped/`
- Legacy row-level Phase 2 figures: `reports/dissertation_figures/reason_attribution_method_comparison/`

## Where To Find Final Tables

- Main table shortlist: `docs/dissertation/final_table_shortlist.md`
- Generated final table index: `reports/dissertation_final/final_table_index.md`
- Existing result directory: `reports/dissertation_results/`
- Final grouped Phase 2 tables: `reports/stage2_grouped/`
- Legacy row-level Phase 2 tables: `reports/dissertation_results/reason_attribution_method_comparison/`

## How To Reproduce

Start with:

```bash
git lfs install
git lfs pull
pip install -e ".[dev]"
python scripts/build_dissertation_delivery_bundle.py --strict --out-dir reports/dissertation_final
python scripts/final_repository_audit.py
```

Full instructions:

- `docs/dissertation/reproducibility_runbook.md`
- `docs/dissertation/school_server_runbook.md`

## UI/Demo

The local drag-drop gatekeeper can be started after a compatible trained artifact exists:

```bash
python scripts/serve_gatekeeper_app.py --config configs/patchcore_l23.yaml --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz
```

This UI is a local software prototype. It is not a clinical device and does not diagnose disease.

## Safety Warnings

- Not a disease classifier.
- Not a clinical device.
- Not deployment-ready.
- OOD data is evaluation/stress-test only.
- Training remains ID-only.
- Synthetic ID fallback is not real clinical FAF validation.
- Stage 2 reason labels are likely explanations, not clinical diagnoses.
- OOD labels are used for Stage 2 explanation targets only, not Stage 1 fitting.
- Final Stage 2 grouped splits are disjoint by image path and group ID, including
  `parent_image_hash` for generated sensory artifacts. Variants from a common parent remain
  dependent within their assigned split. This remains a controlled, synthetic-backed benchmark
  and is not patient-independent clinical validation.
- Real clinical FAF validation is required before any clinical use.
