# Train-Size Sensitivity

| scheme | scheme_label | train_size | threshold | auroc | auprc | fpr_at_95_tpr | id_false_rejection_rate | ood_recall_at_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics | Image statistics | 50 | 1.8397 | 0.8013 | 0.9762 | 0.8067 | 0.0733 | 0.4697 |
| global_feature_knn | Global feature kNN | 50 | 5.5997 | 0.8948 | 0.9894 | 0.5333 | 0.0267 | 0.5861 |
| mahalanobis_feature | Mahalanobis feature | 50 | 112.6255 | 0.9389 | 0.9941 | 0.4067 | 0.0200 | 0.6830 |
| image_statistics | Image statistics | 100 | 1.7877 | 0.8386 | 0.9820 | 0.7667 | 0.0400 | 0.5952 |
| global_feature_knn | Global feature kNN | 100 | 5.1085 | 0.9123 | 0.9912 | 0.5200 | 0.0267 | 0.6273 |
| mahalanobis_feature | Mahalanobis feature | 100 | 86.5314 | 0.9500 | 0.9952 | 0.3000 | 0.0267 | 0.7685 |
| image_statistics | Image statistics | 250 | 1.6023 | 0.7955 | 0.9745 | 0.7733 | 0.0600 | 0.3285 |
| global_feature_knn | Global feature kNN | 250 | 4.3720 | 0.9234 | 0.9924 | 0.4867 | 0.0533 | 0.7564 |
| mahalanobis_feature | Mahalanobis feature | 250 | 58.9802 | 0.9571 | 0.9960 | 0.2867 | 0.0600 | 0.8539 |
| image_statistics | Image statistics | 500 | 1.7571 | 0.7664 | 0.9713 | 0.7267 | 0.0267 | 0.2848 |
| global_feature_knn | Global feature kNN | 500 | 3.9309 | 0.9389 | 0.9941 | 0.3800 | 0.0533 | 0.8133 |
| mahalanobis_feature | Mahalanobis feature | 500 | 43.2519 | 0.9694 | 0.9972 | 0.1933 | 0.0467 | 0.8982 |
| image_statistics | Image statistics | 700 | 1.7220 | 0.7681 | 0.9717 | 0.7467 | 0.0267 | 0.2994 |
| global_feature_knn | Global feature kNN | 700 | 3.9069 | 0.9458 | 0.9949 | 0.3800 | 0.0333 | 0.8024 |
| mahalanobis_feature | Mahalanobis feature | 700 | 38.5824 | 0.9724 | 0.9975 | 0.2000 | 0.0467 | 0.9079 |
