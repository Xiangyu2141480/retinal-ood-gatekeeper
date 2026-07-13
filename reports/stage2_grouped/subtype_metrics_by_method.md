# Subtype Metrics by Method

| method | split | status | feature_set | estimator | complexity | subtype_accuracy | subtype_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics_logreg | val | ok | statistics | logistic_regression | 1 | 0.8476 | 0.8049 |
| global_feature_knn | val | ok | global | knn | 3 | 0.6452 | 0.5225 |
| nearest_centroid | val | ok | global | nearest_centroid | 2 | 0.5762 | 0.4679 |
| logistic_regression | val | ok | global | logistic_regression | 3 | 0.7667 | 0.7028 |
| linear_svm | val | ok | global | linear_svm | 4 | 0.6810 | 0.6114 |
| random_forest_or_gradient_boosting | val | ok | global | random_forest | 5 | 0.8238 | 0.7671 |
| feature_statistics_fusion | val | ok | fusion | logistic_regression | 4 | 0.8667 | 0.8343 |
| hierarchical_classifier | val | ok | fusion | hierarchical_logistic | 6 | 0.9238 | 0.9059 |
| image_statistics_logreg | test | ok | statistics | logistic_regression | 1 | 0.9095 | 0.8861 |
| global_feature_knn | test | ok | global | knn | 3 | 0.6548 | 0.5430 |
| nearest_centroid | test | ok | global | nearest_centroid | 2 | 0.5643 | 0.4499 |
| logistic_regression | test | ok | global | logistic_regression | 3 | 0.7619 | 0.6937 |
| linear_svm | test | ok | global | linear_svm | 4 | 0.6929 | 0.6133 |
| random_forest_or_gradient_boosting | test | ok | global | random_forest | 5 | 0.8310 | 0.7696 |
| feature_statistics_fusion | test | ok | fusion | logistic_regression | 4 | 0.9048 | 0.8777 |
| hierarchical_classifier | test | ok | fusion | hierarchical_logistic | 6 | 0.9357 | 0.9176 |
