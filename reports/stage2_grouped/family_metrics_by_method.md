# Family Metrics by Method

| method | split | status | feature_set | estimator | complexity | family_accuracy | family_argmax_accuracy | family_macro_f1 | family_balanced_accuracy | unknown_threshold | known_coverage_at_gamma | unknown_rate_at_gamma | accuracy_excluding_unknown | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics_logreg | val | ok | statistics | logistic_regression | 1 | 0.9167 | 0.9167 | 0.9121 | 0.9411 | 0.5000 | 1.0000 | 0.0000 | 0.9167 | 0.1111 |
| global_feature_knn | val | ok | global | knn | 3 | 0.9452 | 0.9452 | 0.9391 | 0.9486 | 0.5000 | 1.0000 | 0.0000 | 0.9452 | 0.0165 |
| nearest_centroid | val | ok | global | nearest_centroid | 2 | 0.8286 | 0.8310 | 0.8110 | 0.8422 | 0.5000 | 0.9952 | 0.0048 | 0.8325 | 0.0162 |
| logistic_regression | val | ok | global | logistic_regression | 3 | 0.9643 | 0.9643 | 0.9601 | 0.9714 | 0.5000 | 1.0000 | 0.0000 | 0.9643 | 6.1128 |
| linear_svm | val | ok | global | linear_svm | 4 | 0.9833 | 0.9857 | 0.9846 | 0.9806 | 0.5000 | 0.9976 | 0.0024 | 0.9857 | 9.3523 |
| random_forest_or_gradient_boosting | val | ok | global | random_forest | 5 | 0.9810 | 0.9857 | 0.9924 | 0.9850 | 0.5000 | 0.9810 | 0.0190 | 1.0000 | 4.5095 |
| feature_statistics_fusion | val | ok | fusion | logistic_regression | 4 | 0.9929 | 0.9929 | 0.9925 | 0.9919 | 0.5000 | 1.0000 | 0.0000 | 0.9929 | 4.9431 |
| hierarchical_classifier | val | ok | fusion | hierarchical_logistic | 6 | 0.9929 | 0.9929 | 0.9925 | 0.9919 | 0.5000 | 1.0000 | 0.0000 | 0.9929 | 7.3968 |
| image_statistics_logreg | test | ok | statistics | logistic_regression | 1 | 0.9452 | 0.9452 | 0.9405 | 0.9625 | 0.5000 | 1.0000 | 0.0000 | 0.9452 | 0.1111 |
| global_feature_knn | test | ok | global | knn | 3 | 0.9714 | 0.9714 | 0.9676 | 0.9592 | 0.5000 | 1.0000 | 0.0000 | 0.9714 | 0.0165 |
| nearest_centroid | test | ok | global | nearest_centroid | 2 | 0.8024 | 0.8048 | 0.7735 | 0.7875 | 0.5000 | 0.9976 | 0.0024 | 0.8043 | 0.0162 |
| logistic_regression | test | ok | global | logistic_regression | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 6.1128 |
| linear_svm | test | ok | global | linear_svm | 4 | 0.9976 | 0.9976 | 0.9963 | 0.9967 | 0.5000 | 1.0000 | 0.0000 | 0.9976 | 9.3523 |
| random_forest_or_gradient_boosting | test | ok | global | random_forest | 5 | 0.9976 | 1.0000 | 0.9979 | 0.9958 | 0.5000 | 0.9976 | 0.0024 | 1.0000 | 4.5095 |
| feature_statistics_fusion | test | ok | fusion | logistic_regression | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 4.9431 |
| hierarchical_classifier | test | ok | fusion | hierarchical_logistic | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 7.3968 |
