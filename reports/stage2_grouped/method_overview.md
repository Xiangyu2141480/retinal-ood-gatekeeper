# Reason Attribution Method Overview

| method | status | feature_set | estimator | complexity | description | skip_reason | validation_family_macro_f1 | test_family_macro_f1 | validation_subtype_macro_f1 | test_subtype_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics_logreg | ok | statistics | logistic_regression | 1 | Balanced logistic regression on low-level image statistics h(x). |  | 0.9121 | 0.9405 | 0.8049 | 0.8861 |
| global_feature_knn | ok | global | knn | 3 | K-nearest neighbours on deterministic pooled image features z(x). |  | 0.9391 | 0.9676 | 0.5225 | 0.5430 |
| nearest_centroid | ok | global | nearest_centroid | 2 | Class prototype baseline in pooled image-feature space. |  | 0.8110 | 0.7735 | 0.4679 | 0.4499 |
| logistic_regression | ok | global | logistic_regression | 3 | Balanced multinomial logistic regression on pooled image features z(x). |  | 0.9601 | 1.0000 | 0.7028 | 0.6937 |
| linear_svm | ok | global | linear_svm | 4 | Balanced linear SVM margin baseline on pooled image features z(x). |  | 0.9846 | 0.9963 | 0.6114 | 0.6133 |
| random_forest_or_gradient_boosting | ok | global | random_forest | 5 | Balanced random forest non-linear classical ML baseline. |  | 0.9924 | 0.9979 | 0.7671 | 0.7696 |
| feature_statistics_fusion | ok | fusion | logistic_regression | 4 | Balanced logistic regression on concatenated z(x) and h(x). |  | 0.9925 | 1.0000 | 0.8343 | 0.8777 |
| hierarchical_classifier | ok | fusion | hierarchical_logistic | 6 | Family-first classifier; subtype prediction is routed by predicted family. |  | 0.9925 | 1.0000 | 0.9059 | 0.9176 |
