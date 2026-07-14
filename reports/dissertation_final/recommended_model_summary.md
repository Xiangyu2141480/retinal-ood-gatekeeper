# Recommended Model Summary

| stage   | recommendation            | role                                   | key_metric                                                                | evidence                                                                              |
|:--------|:--------------------------|:---------------------------------------|:--------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
| Stage 1 | Mahalanobis feature       | Best quantitative OOD gatekeeper       | AUROC 0.9724; AUPRC 0.9975; FPR@95%TPR 0.2000                             | reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv            |
| Stage 1 | PatchCore L3              | Best localization-oriented companion   | Layer-ablation winner among PatchCore variants for AUROC                  | reports/dissertation_results/primary_balanced_by_subtype_10k/layer_ablation_table.csv |
| Stage 2 | feature_statistics_fusion | Best reason-family attribution method  | Validation macro-F1 0.9925; grouped test accuracy/macro-F1 1.0000/1.0000  | reports/stage2_grouped/selected_models.json                                           |
| Stage 2 | hierarchical_classifier   | Best reason-subtype attribution method | Validation macro-F1 0.9059; grouped test accuracy 0.9357; macro-F1 0.9176 | reports/stage2_grouped/selected_models.json                                           |
