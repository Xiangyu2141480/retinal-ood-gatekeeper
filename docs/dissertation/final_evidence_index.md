# Final Dissertation Evidence Index

This is the central map for the dissertation handoff bundle. It links each thesis claim to the committed dataset package, result tables, figures, scripts, and limitations wording.

## 1. Project Objective

The project builds an unsupervised binary OOD gatekeeper for retinal Fundus Autofluorescence (FAF) image quality control:

`input image -> OOD gatekeeper -> ACCEPT valid FAF / REJECT invalid or OOD`

The system is not a disease classifier. It does not assign disease classes, gene labels, patient labels, clinical labels, or biomarkers.

The completed system contains:

1. Stage 1 ID-only OOD gatekeeper.
2. Multi-scheme OOD model comparison.
3. Robustness and failure analysis.
4. Optional Stage 2 rejected-input reason attribution.
5. Reproducible dataset and Git LFS package.
6. Dissertation-ready figures and evidence index.

## 2. Dataset Package

Dataset v1 is packaged under `datasets/dissertation_v1/` with metadata, checksums, manifests, and Git LFS-tracked image files under `data/images/dissertation_v1/`.

Key manifests:

- `datasets/dissertation_v1/manifests/train_id.csv`: 700 ID-only training rows, all `label=0`, `ood_type=id`.
- `datasets/dissertation_v1/manifests/val_id.csv`: 150 validation ID rows for ID-calibrated thresholds.
- `datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv`: 150 synthetic fallback ID rows, not real clinical FAF validation.
- `datasets/dissertation_v1/manifests/test_ood_full.csv`: 2100 OOD rows across modality shift, sensory artifact, and semantic outlier groups.
- `datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv`: 1650 balanced OOD rows used in the main comparison and robustness package.
- `datasets/dissertation_v1/manifests/reason_grouped_train.csv`: 1260 OOD-only Stage 2 training rows in 630 groups.
- `datasets/dissertation_v1/manifests/reason_grouped_val.csv`: 420 OOD-only Stage 2 validation rows in 210 groups.
- `datasets/dissertation_v1/manifests/reason_grouped_test.csv`: 420 OOD-only Stage 2 test rows in 210 groups.
- `reason_train.csv`, `reason_val.csv`, and `reason_test.csv` are retained only as the legacy row-stratified sensitivity comparison.

## 3. Model Families Evaluated

The completed comparison includes image statistics, autoencoder, global feature kNN, Mahalanobis feature distance, and PatchCore layer variants. All methods are unsupervised OOD gatekeepers trained on ID data only.

The Phase 2 reason-attribution comparison is separate and optional. It is invoked only after the Stage 1 gatekeeper rejects an input. OOD labels are used only as Stage 2 explanation targets, not for Stage 1 fitting/training.

## 4. Main Result

The main multi-scheme comparison identifies Mahalanobis feature distance as the strongest quantitative gatekeeper on dataset v1:

- AUROC: 0.9724
- AUPRC: 0.9975
- FPR@95%TPR: 0.2000
- OOD recall at ID-calibrated threshold: 0.9079

Main evidence: `reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv`.

## 5. Robustness Result

Bootstrap and sensitivity analyses support the Mahalanobis ranking for AUROC and AUPRC, while FPR@95%TPR confidence intervals overlap with Global feature kNN. Safety claims should therefore be cautious and threshold-specific.

Main evidence: `docs/experiments/dissertation_robustness_key_findings.md` and `reports/dissertation_results/robustness_analysis/bootstrap_ci.csv`.

## 6. Threshold-Safety Result

The best balanced prototype threshold is Mahalanobis `val_id_quantile_95`, with ID false rejection 0.0467 and OOD recall 0.9079. Research thresholds such as 95% OOD TPR use OOD labels for evaluation only and are not deployment calibration procedures.

Main evidence: `reports/dissertation_results/robustness_analysis/threshold_policy_sweep.csv`.

## 7. Failure-Analysis Result

Text watermark is the hardest Mahalanobis subtype. It has only-subtype AUROC 0.7361 and dominates Mahalanobis false negatives. PatchCore L3 remains useful because it catches many Mahalanobis misses and provides localizable heatmap evidence.

Main evidence: `reports/dissertation_results/robustness_analysis/method_disagreement_cases.md` and `reports/dissertation_figures/robustness/figure_false_negative_ood_examples.png`.

## 8. Optional Stage 2 Reason Attribution Result

Phase 2 adds optional post-rejection reason attribution. It explains likely rejection causes after Stage 1 has already decided `REJECT`; it does not change the Stage 1 ID-only unsupervised OOD gatekeeper.

The final parent-grouped validation split selects `feature_statistics_fusion` for family attribution
(validation macro-F1 0.9924836011792534). On the grouped test split it reaches accuracy and
macro-F1 1.0000. All three family F1 values are 1.0000, so there is no unique hardest family.

The final parent-grouped validation split selects the non-oracle `hierarchical_classifier` for
subtype attribution (validation macro-F1 0.9058507271445498). Grouped test accuracy is 0.9357 and
macro-F1 is 0.9176. The hardest grouped-test subtype is `text_watermark` (F1 0.7742).

Main evidence: `reports/stage2_grouped/selected_models.json`,
`reports/stage2_grouped/family_metrics.csv`, `reports/stage2_grouped/subtype_metrics.csv`,
`reports/stage2_grouped/group_overlap_summary.csv`, and
`docs/experiments/reason_attribution_parent_grouped.md`.

## 9. Feature-Space Interpretation

Feature-space PCA shows semantic outliers and several modality shifts moving far from ID, explaining Mahalanobis strength on global feature shifts. Text watermark remains close to ID, explaining its failure-mode behavior.

Main evidence: `reports/dissertation_figures/robustness/figure_feature_space_pca_by_ood_type.png`.

## 10. Runtime/Resource Result

Runtime values are local smoke scoring estimates, not hardware-independent deployment benchmarks. Mahalanobis gives the best performance/resource trade-off; PatchCore L3 costs more but adds heatmap/localization support.

Main evidence: `reports/dissertation_results/robustness_analysis/runtime_resource_summary.md`.

## 11. Limitations

- `test_id_synthetic_fallback.csv` is synthetic FAF fallback, not real clinical FAF validation.
- OOD sets are curated stress-test/evaluation sets, not clinical prevalence estimates.
- The results are proof-of-concept and not clinical deployment validation.
- Mahalanobis may perform strongly because many OOD groups introduce global feature shifts.
- Subtle local artifacts, especially text watermark, remain challenging.
- Stage 2 reason labels are likely explanations for rejected inputs, not clinical diagnoses.
- Final Stage 2 grouped splits are disjoint across partitions by image path and group ID, including
  parent hashes for generated variants. Related variants remain dependent within one split. The
  remaining limitations are controlled, closed-set, synthetic-backed evaluation and lack of
  patient/device/site-independent validation, not residual cross-partition parent overlap.
- Real clinical FAF validation and prospective threshold calibration are future work.

## 12. Future Work

Recommended future work:

- Validate on real clinical FAF data from acquisition devices used in practice.
- Calibrate thresholds prospectively using validation ID data and operational cost targets.
- Expand subtle/local artifact coverage.
- Study hybrid Mahalanobis decision plus PatchCore explanation workflows.
- Replicate grouped Stage 2 attribution across additional seeds and real independent clinical
  cohorts before making patient- or device-generalisation claims.
- Measure runtime/resource behavior on target school-server and deployment hardware.

## 13. Where To Find Every Figure/Table/Script

- Final result summary: `reports/dissertation_final/final_result_summary.md`
- Final figure index: `reports/dissertation_final/final_figure_index.md`
- Final table index: `reports/dissertation_final/final_table_index.md`
- Final claim/evidence index: `reports/dissertation_final/final_claim_evidence_index.md`
- Full figure shortlist: `docs/dissertation/final_figure_shortlist.md`
- Full table shortlist: `docs/dissertation/final_table_shortlist.md`
- Reproducibility runbook: `docs/dissertation/reproducibility_runbook.md`
- School-server runbook: `docs/dissertation/school_server_runbook.md`
- Final Phase 2 grouped findings: `reports/stage2_grouped/summary.md`
- Phase 2 grouped split audit: `reports/stage2_grouped/split_audit.md`
- Pre-remediation code/data audit: `docs/dissertation/vincent_feedback_code_audit.md`
- Pre-remediation overlap evidence: `reports/audit/parent_overlap_summary.csv` and
  `reports/audit/parent_overlap_examples.csv`
- Legacy Phase 2 sensitivity evidence: `reports/dissertation_results/reason_attribution_method_comparison/`
- Final audit: `scripts/final_repository_audit.py`
- Final bundle builder: `scripts/build_dissertation_delivery_bundle.py`

## Claim-Evidence Map

| dissertation claim | evidence file | script/source | figure/table | notes |
|---|---|---|---|---|
| The system is an upstream OOD gatekeeper, not a disease classifier. | `README.md`; `docs/dissertation/README.md` | Project documentation | `figure_two_stage_updated_pipeline.png` | Use binary ACCEPT/REJECT wording. |
| Training uses ID-only FAF images. | `datasets/dissertation_v1/manifests/train_id.csv` | `scripts/validate_manifests.py` | `final_result_summary.csv` | Training rows are `label=0`, `ood_type=id`. |
| OOD data remains evaluation/stress-test only. | `reports/dissertation_results/robustness_analysis/result_provenance.md` | Robustness analysis provenance | `threshold_policy_sweep.md` | OOD labels are used for grouping and evaluation only. |
| The dataset covers four input categories. | `datasets/dissertation_v1/manifests/test_ood_full.csv` | Dataset builder and manifest validation | `figure_dataset_taxonomy.png` | ID, modality shift, sensory artefact, semantic outlier. |
| Mahalanobis is the best quantitative model on dataset v1. | `reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv` | `scripts/generate_multi_scheme_comparison_package.py` | `figure_stage1_overall_comparison_combined.png` | AUROC 0.9724 and lowest FPR@95%TPR estimate. |
| PatchCore L3 is the best PatchCore/localization model. | `reports/dissertation_results/primary_balanced_by_subtype_10k/metrics_summary.csv` | `scripts/generate_selected_patchcore_heatmaps.py` | `figure_patchcore_layer_ablation_combined.png` | Useful for heatmaps, not the strongest quantitative gatekeeper. |
| Autoencoder is a weaker reconstruction baseline. | `reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv` | `scripts/train_autoencoder.py`; `scripts/evaluate_autoencoder.py` | `figure_stage1_overall_comparison_combined.png` | Demonstrates reconstruction-baseline limitations. |
| FPR@95%TPR is critical for safety interpretation. | `reports/dissertation_results/robustness_analysis/bootstrap_ci.csv` | `scripts/bootstrap_dissertation_metrics.py` | `figure_bootstrap_ci_fpr95.png` | FPR intervals are wider and overlap. |
| Results are proof-of-concept due to synthetic ID fallback. | `docs/experiments/dissertation_robustness_key_findings.md` | Final interpretation docs | `claims_and_limitations_matrix.md` | Not real clinical FAF validation or deployment validation. |
| Phase 2 reason attribution is optional post-rejection explanation. | `docs/experiments/reason_attribution_parent_grouped.md`; `reports/stage2_grouped/selected_models.json` | `scripts/generate_stage2_grouped_report.py` | `figure_grouped_stage2_method_comparison.png` | Stage 1 remains ID-only and unsupervised; OOD labels are Stage 2 targets only. |
| The grouped Stage 2 family method is `feature_statistics_fusion`. | `reports/stage2_grouped/family_metrics.csv`; `selected_models.json` | `scripts/generate_stage2_grouped_report.py` | `figure_grouped_family_confusion_matrix.png` | Validation macro-F1 0.9925; grouped test accuracy/macro-F1 1.0000/1.0000; no unique hardest family. |
| The grouped Stage 2 subtype method is the non-oracle `hierarchical_classifier`. | `reports/stage2_grouped/subtype_metrics.csv`; `predictions_test.csv` | `scripts/generate_stage2_grouped_report.py` | `figure_grouped_subtype_confusion_matrix.png` | Validation macro-F1 0.9059; grouped test accuracy/macro-F1 0.9357/0.9176; hardest subtype `text_watermark`. |
