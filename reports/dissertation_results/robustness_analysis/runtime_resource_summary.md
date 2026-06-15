# Runtime and Resource Summary

| scheme | scheme_label | artifact_size_mb | scoring_time_seconds | scoring_images | scoring_ms_per_image | fit_time_seconds | fit_time_note | localization_support | implementation_complexity | primary_auroc | primary_fpr_at_95_tpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics | Image statistics | 0.0013 | 0.2545 | 22 | 11.5703 |  | not remeasured; existing PR #21 training artifacts reused | none | very low | 0.7681 | 0.7467 |
| autoencoder | Autoencoder | 0.2553 | 0.2419 | 22 | 10.9943 |  | not remeasured; existing PR #21 training artifacts reused | weak reconstruction | low | 0.7649 | 0.9467 |
| global_feature_knn | Global feature kNN | 2.5021 | 0.8289 | 22 | 37.6755 |  | not remeasured; existing PR #21 training artifacts reused | none | medium | 0.9458 | 0.3800 |
| mahalanobis_feature | Mahalanobis feature | 6.2061 | 0.8660 | 22 | 39.3631 |  | not remeasured; existing PR #21 training artifacts reused | none | medium | 0.9724 | 0.2000 |
| patchcore_l3 | PatchCore L3 | 16.0489 | 1.2290 | 22 | 55.8631 |  | not remeasured; existing PR #21 training artifacts reused | patch heatmap | high | 0.8819 | 0.6333 |
| patchcore_l2_l3 | PatchCore L2+L3 | 31.3294 | 1.2642 | 22 | 57.4653 |  | not remeasured; existing PR #21 training artifacts reused | patch heatmap | very high | 0.8722 | 0.6867 |
