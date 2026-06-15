# Threshold Policy By Scheme

Generated from compact evaluation metrics and score CSVs. Training remained ID-only; OOD labels were used only for evaluation grouping.

| eval_set | scheme | scheme_label | threshold | threshold_source | threshold_at_95_tpr | id_false_rejection_count | id_false_rejection_rate | ood_recall_at_threshold | id_count | ood_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_by_subtype | image_statistics | Image statistics | 1.7220 | validation_id_quantile | 0.6057 | 4 | 0.0267 | 0.2994 | 150 | 1650 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_subtype | autoencoder | Autoencoder | 0.0017 | validation_id_quantile | 0.0002 | 4 | 0.0267 | 0.5588 | 150 | 1650 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_subtype | global_feature_knn | Global feature kNN | 3.9069 | validation_id_quantile | 2.6765 | 5 | 0.0333 | 0.8024 | 150 | 1650 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | 38.5824 | validation_id_quantile | 32.0572 | 7 | 0.0467 | 0.9079 | 150 | 1650 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | 32.9785 | validation_id_quantile | 24.6483 | 9 | 0.0600 | 0.6200 | 150 | 1650 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | 34.7633 | validation_id_quantile | 26.8046 | 10 | 0.0667 | 0.7097 | 150 | 1650 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | 60.8534 | validation_id_quantile | 33.7603 | 5 | 0.0333 | 0.3527 | 150 | 1650 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | 44.2789 | validation_id_quantile | 35.7651 | 11 | 0.0733 | 0.6933 | 150 | 1650 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_type | image_statistics | Image statistics | 1.7220 | validation_id_quantile | 0.7235 | 4 | 0.0267 | 0.5492 | 150 | 1200 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_type | autoencoder | Autoencoder | 0.0017 | validation_id_quantile | 0.0004 | 4 | 0.0267 | 0.7400 | 150 | 1200 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_type | global_feature_knn | Global feature kNN | 3.9069 | validation_id_quantile | 3.3913 | 5 | 0.0333 | 0.9075 | 150 | 1200 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | 38.5824 | validation_id_quantile | 41.7868 | 7 | 0.0467 | 0.9625 | 150 | 1200 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_type | patchcore_l3 | PatchCore L3 | 34.7633 | validation_id_quantile | 28.8239 | 10 | 0.0667 | 0.8633 | 150 | 1200 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | 44.2789 | validation_id_quantile | 37.6004 | 11 | 0.0733 | 0.8517 | 150 | 1200 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| full_ood_stress | image_statistics | Image statistics | 1.7220 | validation_id_quantile | 0.6376 | 4 | 0.0267 | 0.4229 | 150 | 2100 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| full_ood_stress | autoencoder | Autoencoder | 0.0017 | validation_id_quantile | 0.0002 | 4 | 0.0267 | 0.6314 | 150 | 2100 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| full_ood_stress | global_feature_knn | Global feature kNN | 3.9069 | validation_id_quantile | 2.8789 | 5 | 0.0333 | 0.8448 | 150 | 2100 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | 38.5824 | validation_id_quantile | 34.4895 | 7 | 0.0467 | 0.9276 | 150 | 2100 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| full_ood_stress | patchcore_l3 | PatchCore L3 | 34.7633 | validation_id_quantile | 27.2317 | 10 | 0.0667 | 0.7700 | 150 | 2100 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | 44.2789 | validation_id_quantile | 36.1598 | 11 | 0.0733 | 0.7562 | 150 | 2100 | synthetic ID fallback; not real clinical FAF validation; training ID-only; OOD evaluation-only |
