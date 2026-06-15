# Per OOD Subtype By Scheme

Generated from compact evaluation metrics and score CSVs. Training remained ID-only; OOD labels were used only for evaluation grouping.

| eval_set | scheme | scheme_label | run_name | ood_type | ood_subtype | auroc | auprc | fpr_at_95_tpr | id_count | ood_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | modality_shift | colour_fundus | 0.8364 | 0.7733 | 0.4467 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | modality_shift | oct_screenshot | 0.9710 | 0.9201 | 0.0733 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | text_watermark | 0.5129 | 0.5271 | 0.9533 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | rectangle_annotation | 0.7349 | 0.6786 | 0.6133 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | arrow_annotation | 0.7120 | 0.6707 | 0.6600 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | composite_layout | 0.9538 | 0.9119 | 0.0733 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | blur_artifact | 0.6533 | 0.6385 | 0.7800 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | border_crop | 0.6991 | 0.6349 | 0.6867 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | gaussian_noise | 0.8683 | 0.8211 | 0.5133 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | sensory_artifact | jpeg_compression | 0.5104 | 0.5218 | 0.9467 | 150 | 150 |
| balanced_by_subtype | image_statistics | Image statistics | image_statistics | semantic_outlier | cifar10_natural | 0.9972 | 0.9971 | 0.0133 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | oct_screenshot | 0.9854 | 0.9778 | 0.0267 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | text_watermark | 0.5295 | 0.5292 | 0.9067 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | arrow_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | composite_layout | 0.9932 | 0.9827 | 0.0133 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | blur_artifact | 0.1116 | 0.3253 | 1.0000 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | border_crop | 0.4336 | 0.4550 | 0.9600 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | gaussian_noise | 0.8740 | 0.8622 | 0.5067 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | jpeg_compression | 0.5057 | 0.5137 | 0.9600 | 150 | 150 |
| balanced_by_subtype | autoencoder | Autoencoder | autoencoder_baseline | semantic_outlier | cifar10_natural | 0.9809 | 0.9854 | 0.0867 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | text_watermark | 0.5700 | 0.5509 | 0.8200 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | rectangle_annotation | 0.9988 | 0.9988 | 0.0133 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | arrow_annotation | 0.9584 | 0.9517 | 0.1800 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | blur_artifact | 0.9854 | 0.9847 | 0.0733 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | border_crop | 0.9619 | 0.9589 | 0.1800 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | gaussian_noise | 0.9549 | 0.9493 | 0.1800 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | jpeg_compression | 0.9745 | 0.9761 | 0.1133 | 150 | 150 |
| balanced_by_subtype | global_feature_knn | Global feature kNN | global_feature_knn | semantic_outlier | cifar10_natural | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | text_watermark | 0.7361 | 0.6876 | 0.6467 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | arrow_annotation | 0.9968 | 0.9965 | 0.0133 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | blur_artifact | 0.9964 | 0.9962 | 0.0133 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | border_crop | 0.9946 | 0.9935 | 0.0133 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | gaussian_noise | 0.9818 | 0.9685 | 0.0667 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | jpeg_compression | 0.9910 | 0.9912 | 0.0467 | 150 | 150 |
| balanced_by_subtype | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | semantic_outlier | cifar10_natural | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | modality_shift | colour_fundus | 0.9936 | 0.9933 | 0.0400 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | text_watermark | 0.9492 | 0.9485 | 0.2267 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | arrow_annotation | 0.9999 | 0.9999 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | blur_artifact | 0.4927 | 0.4813 | 0.9067 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | border_crop | 0.7492 | 0.7766 | 0.7800 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | gaussian_noise | 0.4256 | 0.4481 | 0.9467 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | sensory_artifact | jpeg_compression | 0.5585 | 0.5207 | 0.8533 | 150 | 150 |
| balanced_by_subtype | patchcore_l2 | PatchCore L2 | patchcore_layer2 | semantic_outlier | cifar10_natural | 0.9392 | 0.9201 | 0.2000 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | colour_fundus | 0.9696 | 0.9336 | 0.0667 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | oct_screenshot | 0.9948 | 0.9935 | 0.0133 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | text_watermark | 0.9572 | 0.9467 | 0.2000 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | rectangle_annotation | 0.9929 | 0.9910 | 0.0200 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | arrow_annotation | 0.9984 | 0.9983 | 0.0133 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | composite_layout | 0.9931 | 0.9904 | 0.0133 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | blur_artifact | 0.5776 | 0.5519 | 0.9000 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | border_crop | 0.8779 | 0.8682 | 0.4467 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | gaussian_noise | 0.7501 | 0.7490 | 0.7400 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | jpeg_compression | 0.5979 | 0.5539 | 0.8200 | 150 | 150 |
| balanced_by_subtype | patchcore_l3 | PatchCore L3 | patchcore_layer3 | semantic_outlier | cifar10_natural | 0.9918 | 0.9903 | 0.0400 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | modality_shift | colour_fundus | 0.8424 | 0.8016 | 0.5200 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | modality_shift | oct_screenshot | 0.9457 | 0.8987 | 0.1533 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | text_watermark | 0.6406 | 0.6239 | 0.8933 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | rectangle_annotation | 0.9765 | 0.9416 | 0.0533 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | arrow_annotation | 0.9853 | 0.9741 | 0.0333 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | composite_layout | 0.4402 | 0.4410 | 0.9400 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | blur_artifact | 0.5909 | 0.5802 | 0.8933 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | border_crop | 0.7416 | 0.7076 | 0.6267 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | gaussian_noise | 0.5818 | 0.5813 | 0.9133 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | sensory_artifact | jpeg_compression | 0.5459 | 0.5422 | 0.8933 | 150 | 150 |
| balanced_by_subtype | patchcore_l4 | PatchCore L4 | patchcore_layer4 | semantic_outlier | cifar10_natural | 0.9614 | 0.9361 | 0.1200 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | colour_fundus | 0.9837 | 0.9725 | 0.0733 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | text_watermark | 0.9594 | 0.9517 | 0.1400 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | rectangle_annotation | 0.9999 | 0.9999 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | arrow_annotation | 0.9994 | 0.9994 | 0.0067 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | blur_artifact | 0.5531 | 0.5294 | 0.8400 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | border_crop | 0.8884 | 0.8946 | 0.5267 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | gaussian_noise | 0.6432 | 0.6496 | 0.8000 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | jpeg_compression | 0.5756 | 0.5411 | 0.8667 | 150 | 150 |
| balanced_by_subtype | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | semantic_outlier | cifar10_natural | 0.9912 | 0.9899 | 0.0267 | 150 | 150 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | modality_shift | colour_fundus | 0.8406 | 0.8198 | 0.4333 | 150 | 200 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | modality_shift | oct_screenshot | 0.9693 | 0.9350 | 0.0733 | 150 | 200 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | text_watermark | 0.4800 | 0.2337 | 0.9800 | 150 | 43 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | rectangle_annotation | 0.7512 | 0.4605 | 0.6133 | 150 | 52 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | arrow_annotation | 0.7275 | 0.4537 | 0.6600 | 150 | 51 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | composite_layout | 0.9551 | 0.7887 | 0.0733 | 150 | 46 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | blur_artifact | 0.6444 | 0.3790 | 0.7200 | 150 | 49 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | border_crop | 0.6540 | 0.3735 | 0.7533 | 150 | 58 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | gaussian_noise | 0.8591 | 0.6292 | 0.5667 | 150 | 51 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | sensory_artifact | jpeg_compression | 0.5351 | 0.2937 | 0.9467 | 150 | 50 |
| balanced_by_type | image_statistics | Image statistics | image_statistics | semantic_outlier | cifar10_natural | 0.9970 | 0.9988 | 0.0133 | 150 | 400 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | oct_screenshot | 0.9855 | 0.9833 | 0.0267 | 150 | 200 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | text_watermark | 0.5453 | 0.2490 | 0.9067 | 150 | 43 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 52 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | arrow_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 51 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | composite_layout | 0.9930 | 0.9465 | 0.0133 | 150 | 46 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | blur_artifact | 0.1114 | 0.1464 | 1.0000 | 150 | 49 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | border_crop | 0.4638 | 0.2661 | 0.9533 | 150 | 58 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | gaussian_noise | 0.8732 | 0.6980 | 0.4867 | 150 | 51 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | jpeg_compression | 0.4620 | 0.2346 | 1.0000 | 150 | 50 |
| balanced_by_type | autoencoder | Autoencoder | autoencoder_baseline | semantic_outlier | cifar10_natural | 0.9755 | 0.9919 | 0.1133 | 150 | 400 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | text_watermark | 0.5667 | 0.2693 | 0.8600 | 150 | 43 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | rectangle_annotation | 0.9992 | 0.9979 | 0.0000 | 150 | 52 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | arrow_annotation | 0.9590 | 0.8813 | 0.1800 | 150 | 51 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 46 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | blur_artifact | 0.9886 | 0.9650 | 0.0467 | 150 | 49 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | border_crop | 0.9553 | 0.8908 | 0.2533 | 150 | 58 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | gaussian_noise | 0.9472 | 0.8591 | 0.1867 | 150 | 51 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | jpeg_compression | 0.9819 | 0.9598 | 0.1000 | 150 | 50 |
| balanced_by_type | global_feature_knn | Global feature kNN | global_feature_knn | semantic_outlier | cifar10_natural | 1.0000 | 1.0000 | 0.0000 | 150 | 400 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | text_watermark | 0.7231 | 0.3834 | 0.6467 | 150 | 43 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 52 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | arrow_annotation | 0.9969 | 0.9900 | 0.0133 | 150 | 51 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 46 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | blur_artifact | 0.9970 | 0.9901 | 0.0133 | 150 | 49 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | border_crop | 0.9940 | 0.9805 | 0.0133 | 150 | 58 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | gaussian_noise | 0.9825 | 0.9246 | 0.0533 | 150 | 51 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | jpeg_compression | 0.9915 | 0.9793 | 0.0333 | 150 | 50 |
| balanced_by_type | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | semantic_outlier | cifar10_natural | 1.0000 | 1.0000 | 0.0000 | 150 | 400 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | colour_fundus | 0.9695 | 0.9473 | 0.0667 | 150 | 200 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | oct_screenshot | 0.9949 | 0.9953 | 0.0133 | 150 | 200 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | text_watermark | 0.9577 | 0.8582 | 0.2000 | 150 | 43 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | rectangle_annotation | 0.9922 | 0.9718 | 0.0133 | 150 | 52 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | arrow_annotation | 0.9984 | 0.9953 | 0.0133 | 150 | 51 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | composite_layout | 0.9928 | 0.9686 | 0.0133 | 150 | 46 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | blur_artifact | 0.5827 | 0.3001 | 0.8200 | 150 | 49 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | border_crop | 0.8882 | 0.7477 | 0.4467 | 150 | 58 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | gaussian_noise | 0.7902 | 0.5751 | 0.6000 | 150 | 51 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | jpeg_compression | 0.6437 | 0.3398 | 0.7133 | 150 | 50 |
| balanced_by_type | patchcore_l3 | PatchCore L3 | patchcore_layer3 | semantic_outlier | cifar10_natural | 0.9919 | 0.9963 | 0.0267 | 150 | 400 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | colour_fundus | 0.9839 | 0.9788 | 0.0600 | 150 | 200 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | text_watermark | 0.9519 | 0.8433 | 0.1600 | 150 | 43 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | rectangle_annotation | 0.9997 | 0.9993 | 0.0000 | 150 | 52 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | arrow_annotation | 0.9993 | 0.9981 | 0.0067 | 150 | 51 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 46 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | blur_artifact | 0.5966 | 0.2971 | 0.8400 | 150 | 49 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | border_crop | 0.8897 | 0.8008 | 0.5600 | 150 | 58 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | gaussian_noise | 0.6952 | 0.4409 | 0.8000 | 150 | 51 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | jpeg_compression | 0.6123 | 0.3209 | 0.8133 | 150 | 50 |
| balanced_by_type | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | semantic_outlier | cifar10_natural | 0.9906 | 0.9958 | 0.0400 | 150 | 400 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | modality_shift | colour_fundus | 0.8406 | 0.8198 | 0.4333 | 150 | 200 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | modality_shift | oct_screenshot | 0.9693 | 0.9350 | 0.0733 | 150 | 200 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | text_watermark | 0.5129 | 0.5271 | 0.9533 | 150 | 150 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | rectangle_annotation | 0.7349 | 0.6786 | 0.6133 | 150 | 150 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | arrow_annotation | 0.7120 | 0.6707 | 0.6600 | 150 | 150 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | composite_layout | 0.9538 | 0.9119 | 0.0733 | 150 | 150 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | blur_artifact | 0.6533 | 0.6385 | 0.7800 | 150 | 150 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | border_crop | 0.6991 | 0.6349 | 0.6867 | 150 | 150 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | gaussian_noise | 0.8683 | 0.8211 | 0.5133 | 150 | 150 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | sensory_artifact | jpeg_compression | 0.5104 | 0.5218 | 0.9467 | 150 | 150 |
| full_ood_stress | image_statistics | Image statistics | image_statistics | semantic_outlier | cifar10_natural | 0.9971 | 0.9991 | 0.0133 | 150 | 500 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | modality_shift | oct_screenshot | 0.9855 | 0.9833 | 0.0267 | 150 | 200 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | text_watermark | 0.5295 | 0.5292 | 0.9067 | 150 | 150 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | arrow_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | composite_layout | 0.9932 | 0.9827 | 0.0133 | 150 | 150 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | blur_artifact | 0.1116 | 0.3253 | 1.0000 | 150 | 150 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | border_crop | 0.4336 | 0.4550 | 0.9600 | 150 | 150 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | gaussian_noise | 0.8740 | 0.8622 | 0.5067 | 150 | 150 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | sensory_artifact | jpeg_compression | 0.5057 | 0.5137 | 0.9600 | 150 | 150 |
| full_ood_stress | autoencoder | Autoencoder | autoencoder_baseline | semantic_outlier | cifar10_natural | 0.9734 | 0.9928 | 0.1400 | 150 | 500 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | text_watermark | 0.5700 | 0.5509 | 0.8200 | 150 | 150 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | rectangle_annotation | 0.9988 | 0.9988 | 0.0133 | 150 | 150 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | arrow_annotation | 0.9584 | 0.9517 | 0.1800 | 150 | 150 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | blur_artifact | 0.9854 | 0.9847 | 0.0733 | 150 | 150 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | border_crop | 0.9619 | 0.9589 | 0.1800 | 150 | 150 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | gaussian_noise | 0.9549 | 0.9493 | 0.1800 | 150 | 150 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | sensory_artifact | jpeg_compression | 0.9745 | 0.9761 | 0.1133 | 150 | 150 |
| full_ood_stress | global_feature_knn | Global feature kNN | global_feature_knn | semantic_outlier | cifar10_natural | 1.0000 | 1.0000 | 0.0000 | 150 | 500 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | text_watermark | 0.7361 | 0.6876 | 0.6467 | 150 | 150 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | arrow_annotation | 0.9968 | 0.9965 | 0.0133 | 150 | 150 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | blur_artifact | 0.9964 | 0.9962 | 0.0133 | 150 | 150 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | border_crop | 0.9946 | 0.9935 | 0.0133 | 150 | 150 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | gaussian_noise | 0.9818 | 0.9685 | 0.0667 | 150 | 150 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | sensory_artifact | jpeg_compression | 0.9910 | 0.9912 | 0.0467 | 150 | 150 |
| full_ood_stress | mahalanobis_feature | Mahalanobis feature | mahalanobis_feature | semantic_outlier | cifar10_natural | 1.0000 | 1.0000 | 0.0000 | 150 | 500 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | colour_fundus | 0.9695 | 0.9473 | 0.0667 | 150 | 200 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | modality_shift | oct_screenshot | 0.9949 | 0.9953 | 0.0133 | 150 | 200 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | text_watermark | 0.9572 | 0.9467 | 0.2000 | 150 | 150 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | rectangle_annotation | 0.9929 | 0.9910 | 0.0200 | 150 | 150 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | arrow_annotation | 0.9984 | 0.9983 | 0.0133 | 150 | 150 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | composite_layout | 0.9931 | 0.9904 | 0.0133 | 150 | 150 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | blur_artifact | 0.5776 | 0.5519 | 0.9000 | 150 | 150 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | border_crop | 0.8779 | 0.8682 | 0.4467 | 150 | 150 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | gaussian_noise | 0.7501 | 0.7490 | 0.7400 | 150 | 150 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | sensory_artifact | jpeg_compression | 0.5979 | 0.5539 | 0.8200 | 150 | 150 |
| full_ood_stress | patchcore_l3 | PatchCore L3 | patchcore_layer3 | semantic_outlier | cifar10_natural | 0.9924 | 0.9972 | 0.0267 | 150 | 500 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | colour_fundus | 0.9839 | 0.9788 | 0.0600 | 150 | 200 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 200 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | text_watermark | 0.9594 | 0.9517 | 0.1400 | 150 | 150 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | rectangle_annotation | 0.9999 | 0.9999 | 0.0000 | 150 | 150 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | arrow_annotation | 0.9994 | 0.9994 | 0.0067 | 150 | 150 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | blur_artifact | 0.5531 | 0.5294 | 0.8400 | 150 | 150 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | border_crop | 0.8884 | 0.8946 | 0.5267 | 150 | 150 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | gaussian_noise | 0.6432 | 0.6496 | 0.8000 | 150 | 150 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | sensory_artifact | jpeg_compression | 0.5756 | 0.5411 | 0.8667 | 150 | 150 |
| full_ood_stress | patchcore_l2_l3 | PatchCore L2+L3 | patchcore_layer2_layer3 | semantic_outlier | cifar10_natural | 0.9907 | 0.9967 | 0.0400 | 150 | 500 |
