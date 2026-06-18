# Best Method Summary

| item | value |
| --- | --- |
| best_reason_family_method | linear_svm |
| best_reason_family_validation_macro_f1 | 0.9908 |
| best_reason_family_test_accuracy | 0.9881 |
| best_reason_family_test_macro_f1 | 0.9901 |
| best_reason_family_test_balanced_accuracy | 0.9872 |
| family_macro_f1_delta_vs_pr24_baseline | +0.0954 |
| family_improves_over_pr24_baseline | True |
| unknown_threshold_gamma | 0.50 |
| known_coverage_at_gamma | 0.9952 |
| unknown_rate_at_gamma | 0.0048 |
| accuracy_excluding_unknown | 0.9928 |
| best_subtype_method | hierarchical_classifier |
| best_subtype_validation_macro_f1 | 0.8671 |
| best_subtype_test_macro_f1 | 0.9059 |
| subtype_macro_f1_delta_vs_pr24_baseline | 0.2335 |
| hardest_family | semantic_outlier (F1=0.9848) |
| hardest_subtype | rectangle_annotation (F1=0.7241) |
| selected_final_method | linear_svm |
| selection_rationale | Selected by validation reason-family macro-F1, with simpler methods preferred on ties. |
| conceptual_boundary | Stage 1 remains ID-only unsupervised OOD detection; Stage 2 is post-hoc explanation. |
| clinical_scope | Reason labels are likely rejection explanations, not disease predictions. |
