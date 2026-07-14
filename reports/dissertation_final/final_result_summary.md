# Final Dissertation Result Summary

This is a thesis-ready summary of existing merged outputs. It does not rerun training.

The project remains an unsupervised binary FAF OOD gatekeeper, not a disease classifier. Training is ID-only and OOD data is evaluation/stress-test only. `test_id_synthetic_fallback.csv` is synthetic ID fallback, not real clinical FAF validation.

The completed system contains:

1. Stage 1 ID-only OOD gatekeeper.
2. Multi-scheme OOD model comparison.
3. Robustness and failure analysis.
4. Optional Stage 2 rejected-input reason attribution.
5. Reproducible dataset and Git LFS package.
6. Dissertation-ready figures and evidence index.

## Dataset Manifests

| split_or_manifest | rows | label_counts | ood_type_counts |
| --- | --- | --- | --- |
| train_id | 700 | 0=700 | id=700 |
| val_id | 150 | 0=150 | id=150 |
| test_id_synthetic_fallback | 150 | 0=150 | id=150 |
| test_ood_full | 2100 | 1=2100 | modality_shift=400; semantic_outlier=500; sensory_artifact=1200 |
| test_ood_balanced_by_subtype | 1650 | 1=1650 | modality_shift=300; semantic_outlier=150; sensory_artifact=1200 |

## Summary Rows

| item | value | evidence |
| --- | --- | --- |
| packaged_dataset_images | 3100 | data/images/dissertation_v1/ |
| dataset_manifest_rows | 4750 | datasets/dissertation_v1/manifests/*.csv |
| dataset_categories | id, modality_shift, semantic_outlier, sensory_artifact | datasets/dissertation_v1/manifests/test_ood_full.csv |
| number_of_schemes | 8 | reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv |
| completed_model_configurations | 8 | reports/dissertation_results/multi_scheme_comparison/scheme_overview.csv |
| best_quantitative_model | Mahalanobis feature (AUROC 0.9724, AUPRC 0.9975, FPR@95%TPR 0.2000) | reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv |
| best_localization_model | PatchCore L3 | reports/dissertation_results/primary_balanced_by_subtype_10k/metrics_summary.csv |
| safety_threshold_result | Mahalanobis val_id_quantile_95: ID false rejection 0.0467, OOD recall 0.9079 | reports/dissertation_results/robustness_analysis/threshold_policy_sweep.csv |
| hardest_subtype | text_watermark (AUROC 0.7361) | reports/dissertation_results/robustness_analysis/subtype_influence.csv |
| key_robustness_result | Mahalanobis remains strongest under bootstrap AUROC/AUPRC; FPR intervals overlap. | reports/dissertation_results/robustness_analysis/bootstrap_ci.csv |
| key_failure_analysis_result | Text watermark dominates Mahalanobis false negatives; PatchCore L3 catches many misses. | reports/dissertation_results/robustness_analysis/method_disagreement_cases.md |
| optional_stage2_scope | Optional supervised post-rejection reason attribution; Stage 1 remains ID-only and unsupervised. | reports/stage2_grouped/summary.md |
| best_reason_family_method | feature_statistics_fusion: validation macro-F1 0.9925; grouped test accuracy and macro-F1 1.0000. | reports/stage2_grouped/selected_models.json |
| best_reason_subtype_method | non-oracle hierarchical_classifier: validation macro-F1 0.9059; grouped test accuracy 0.9357 and macro-F1 0.9176. | reports/stage2_grouped/selected_models.json |
| hardest_reason_family | no unique hardest family; all three grouped test F1 values are 1.0000 | reports/stage2_grouped/per_family_f1_by_method.csv |
| hardest_reason_subtype | text_watermark (F1 0.7742) | reports/stage2_grouped/per_subtype_f1_by_method.csv |
| legacy_stage2_sensitivity | grouped minus legacy: family test macro-F1 +0.0099; subtype test macro-F1 +0.0116 | reports/stage2_grouped/legacy_vs_grouped_comparison.csv |
| reason_attribution_limitation | Reason labels are likely explanations, not clinical diagnoses; cross-partition parent/group overlap is zero, but variants from a common parent remain dependent within one split. | reports/stage2_grouped/split_audit.md |

## Polished Results Narrative

The final dissertation result should be presented as a staged quality-control system. Stage 1 is the
primary ID-only unsupervised OOD gatekeeper. Within Stage 1, Mahalanobis feature distance is the
recommended quantitative method because it gives the strongest overall AUROC/AUPRC trade-off and the
lowest FPR@95%TPR among the evaluated schemes. PatchCore L3 should be discussed as the
localization-oriented companion because it provides heatmap evidence and helps interpret selected
failure cases, even though it is not the strongest quantitative model.

Stage 2 is optional, supervised, and runs only after Stage 1 has rejected an input. It should be
reported as post-hoc reason attribution rather than OOD detection. Grouped validation selects
`feature_statistics_fusion` for family attribution and the non-oracle `hierarchical_classifier`
for subtype attribution. Their grouped test results are retrospective only. These labels are
likely explanations, not clinical diagnoses.

The main limitations to retain are synthetic ID fallback, proof-of-concept stress-test evaluation,
no clinical deployment validation, and supervised closed-set Stage 2 explanation. Parent grouping
eliminates cross-partition parent/group overlap, but transformed variants from a common parent
remain dependent within one split; it does not establish patient, device, or site independence.
