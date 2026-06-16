# Final Dissertation Result Summary

This is a thesis-ready summary of existing merged outputs. It does not rerun training.

The project remains an unsupervised binary FAF OOD gatekeeper, not a disease classifier. Training is ID-only and OOD data is evaluation/stress-test only. `test_id_synthetic_fallback.csv` is synthetic ID fallback, not real clinical FAF validation.

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
