# Stage2 Method Summary

| method | feature_set | estimator | family_accuracy | family_macro_f1 | subtype_accuracy | subtype_macro_f1 | selected_role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feature_statistics_fusion | fusion | logistic_regression | 0.9952 | 0.9939 | 0.8476 | 0.8134 |  |
| hierarchical_classifier | fusion | hierarchical_logistic | 0.9952 | 0.9939 | 0.9238 | 0.9059 | Selected subtype method |
| random_forest_or_gradient_boosting | global | random_forest | 0.9905 | 0.9911 | 0.7405 | 0.6786 |  |
| logistic_regression | global | logistic_regression | 0.9929 | 0.9901 | 0.6952 | 0.6303 |  |
| linear_svm | global | linear_svm | 0.9881 | 0.9901 | 0.6524 | 0.5894 | Selected family method |
| global_feature_knn | global | knn | 0.9762 | 0.9740 | 0.5690 | 0.4683 |  |
| image_statistics_logreg | statistics | logistic_regression | 0.9476 | 0.9433 | 0.9048 | 0.8802 |  |
| nearest_centroid | global | nearest_centroid | 0.7810 | 0.7665 | 0.5405 | 0.4137 |  |
