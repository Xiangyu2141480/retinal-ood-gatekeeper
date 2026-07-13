# Best Method Summary

| item | value |
| --- | --- |
| best_reason_family_method | feature_statistics_fusion |
| best_reason_family_validation_macro_f1 | 0.9925 |
| best_reason_family_test_accuracy | 1.0000 |
| best_reason_family_test_macro_f1 | 1.0000 |
| best_reason_family_test_balanced_accuracy | 1.0000 |
| family_macro_f1_delta_vs_pr24_baseline | +0.1053 |
| family_improves_over_pr24_baseline | True |
| unknown_threshold_gamma | 0.50 |
| known_coverage_at_gamma | 1.0000 |
| unknown_rate_at_gamma | 0.0000 |
| accuracy_excluding_unknown | 1.0000 |
| best_subtype_method | hierarchical_classifier |
| best_subtype_validation_macro_f1 | 0.9059 |
| best_subtype_test_macro_f1 | 0.9176 |
| subtype_macro_f1_delta_vs_pr24_baseline | 0.2452 |
| hardest_family | no unique hardest; modality_shift, semantic_outlier, sensory_artifact tied (F1=1.0000) |
| hardest_subtype | text_watermark (F1=0.7742) |
| selected_final_method | feature_statistics_fusion |
| selection_rationale | Selected by validation reason-family macro-F1, with simpler methods preferred on ties. |
| conceptual_boundary | Stage 1 remains ID-only unsupervised OOD detection; Stage 2 is post-hoc explanation. |
| clinical_scope | Reason labels are likely rejection explanations, not disease predictions. |
