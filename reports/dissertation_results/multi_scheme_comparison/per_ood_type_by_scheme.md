# Per OOD Type By Scheme

Generated from compact evaluation metrics and score CSVs. Training remained ID-only; OOD labels were used only for evaluation grouping.

| eval_set | scheme | scheme_label | run_name | ood_type | auroc | auprc | fpr_at_95_tpr | id_count | ood_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | modality_shift | 0.9037 | 0.9189 | 0.3667 | 150 | 300 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | 0.7056 | 0.9440 | 0.8200 | 150 | 1200 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | semantic_outlier | 0.9972 | 0.9971 | 0.0133 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | 0.9927 | 0.9957 | 0.0267 | 150 | 300 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | 0.6809 | 0.9537 | 0.9600 | 150 | 1200 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | semantic_outlier | 0.9809 | 0.9854 | 0.0867 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | 1.0000 | 1.0000 | 0.0000 | 150 | 300 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | 0.9255 | 0.9903 | 0.5067 | 150 | 1200 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | semantic_outlier | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | 1.0000 | 1.0000 | 0.0000 | 150 | 300 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | 0.9621 | 0.9952 | 0.3000 | 150 | 1200 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | semantic_outlier | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | modality_shift | 0.9968 | 0.9984 | 0.0133 | 150 | 300 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | 0.7719 | 0.9677 | 0.8200 | 150 | 1200 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | semantic_outlier | 0.9392 | 0.9201 | 0.2000 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | 0.9822 | 0.9877 | 0.0600 | 150 | 300 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | 0.8431 | 0.9777 | 0.7000 | 150 | 1200 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | semantic_outlier | 0.9918 | 0.9903 | 0.0400 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | modality_shift | 0.8941 | 0.9169 | 0.3667 | 150 | 300 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | 0.6878 | 0.9468 | 0.8600 | 150 | 1200 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | semantic_outlier | 0.9614 | 0.9361 | 0.1200 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | 0.9919 | 0.9954 | 0.0533 | 150 | 300 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | 0.8274 | 0.9760 | 0.7400 | 150 | 1200 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | semantic_outlier | 0.9912 | 0.9899 | 0.0267 | 150 | 150 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | modality_shift | 0.9050 | 0.9365 | 0.3600 | 150 | 400 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | 0.7020 | 0.8524 | 0.7933 | 150 | 400 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | semantic_outlier | 0.9970 | 0.9988 | 0.0133 | 150 | 400 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | 0.9927 | 0.9968 | 0.0267 | 150 | 400 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | 0.6803 | 0.8816 | 0.9533 | 150 | 400 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | semantic_outlier | 0.9755 | 0.9919 | 0.1133 | 150 | 400 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | 1.0000 | 1.0000 | 0.0000 | 150 | 400 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | 0.9312 | 0.9754 | 0.4467 | 150 | 400 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | semantic_outlier | 1.0000 | 1.0000 | 0.0000 | 150 | 400 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | 1.0000 | 1.0000 | 0.0000 | 150 | 400 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | 0.9653 | 0.9878 | 0.2333 | 150 | 400 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | semantic_outlier | 1.0000 | 1.0000 | 0.0000 | 150 | 400 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | 0.9822 | 0.9908 | 0.0600 | 150 | 400 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | 0.8548 | 0.9438 | 0.6267 | 150 | 400 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | semantic_outlier | 0.9919 | 0.9963 | 0.0267 | 150 | 400 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | 0.9919 | 0.9965 | 0.0533 | 150 | 400 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | 0.8420 | 0.9408 | 0.6800 | 150 | 400 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | semantic_outlier | 0.9906 | 0.9958 | 0.0400 | 150 | 400 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | modality_shift | 0.9050 | 0.9365 | 0.3600 | 150 | 400 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | 0.7056 | 0.9440 | 0.8200 | 150 | 1200 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | semantic_outlier | 0.9971 | 0.9991 | 0.0133 | 150 | 500 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | 0.9927 | 0.9968 | 0.0267 | 150 | 400 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | 0.6809 | 0.9537 | 0.9600 | 150 | 1200 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | semantic_outlier | 0.9734 | 0.9928 | 0.1400 | 150 | 500 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | 1.0000 | 1.0000 | 0.0000 | 150 | 400 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | 0.9255 | 0.9903 | 0.5067 | 150 | 1200 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | semantic_outlier | 1.0000 | 1.0000 | 0.0000 | 150 | 500 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | 1.0000 | 1.0000 | 0.0000 | 150 | 400 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | 0.9621 | 0.9952 | 0.3000 | 150 | 1200 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | semantic_outlier | 1.0000 | 1.0000 | 0.0000 | 150 | 500 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | 0.9822 | 0.9908 | 0.0600 | 150 | 400 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | 0.8431 | 0.9777 | 0.7000 | 150 | 1200 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | semantic_outlier | 0.9924 | 0.9972 | 0.0267 | 150 | 500 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | 0.9919 | 0.9965 | 0.0533 | 150 | 400 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | 0.8274 | 0.9760 | 0.7400 | 150 | 1200 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | semantic_outlier | 0.9907 | 0.9967 | 0.0400 | 150 | 500 |
