# Subtype Metrics by Method

| method | split | status | feature_set | estimator | complexity | subtype_accuracy | subtype_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics_logreg | val | ok | statistics | logistic_regression | 1 | 0.8905 | 0.8603 |
| image_statistics_logreg | test | ok | statistics | logistic_regression | 1 | 0.9048 | 0.8802 |
| global_feature_knn | val | ok | global | knn | 3 | 0.5810 | 0.4797 |
| global_feature_knn | test | ok | global | knn | 3 | 0.5690 | 0.4683 |
| nearest_centroid | val | ok | global | nearest_centroid | 2 | 0.5476 | 0.4253 |
| nearest_centroid | test | ok | global | nearest_centroid | 2 | 0.5405 | 0.4137 |
| logistic_regression | val | ok | global | logistic_regression | 3 | 0.6833 | 0.6043 |
| logistic_regression | test | ok | global | logistic_regression | 3 | 0.6952 | 0.6303 |
| linear_svm | val | ok | global | linear_svm | 4 | 0.6405 | 0.5499 |
| linear_svm | test | ok | global | linear_svm | 4 | 0.6524 | 0.5894 |
| random_forest_or_gradient_boosting | val | ok | global | random_forest | 5 | 0.7238 | 0.6524 |
| random_forest_or_gradient_boosting | test | ok | global | random_forest | 5 | 0.7405 | 0.6786 |
| feature_statistics_fusion | val | ok | fusion | logistic_regression | 4 | 0.8214 | 0.7748 |
| feature_statistics_fusion | test | ok | fusion | logistic_regression | 4 | 0.8476 | 0.8134 |
| hierarchical_classifier | val | ok | fusion | hierarchical_logistic | 6 | 0.8929 | 0.8671 |
| hierarchical_classifier | test | ok | fusion | hierarchical_logistic | 6 | 0.9238 | 0.9059 |
