# Reason Attribution Method Overview

| method | status | feature_set | estimator | complexity | description | skip_reason | validation_family_macro_f1 | test_family_macro_f1 | validation_subtype_macro_f1 | test_subtype_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics_logreg | ok | statistics | logistic_regression | 1 | Balanced logistic regression on low-level image statistics h(x). |  | 0.9619 | 0.9433 | 0.8603 | 0.8802 |
| global_feature_knn | ok | global | knn | 3 | K-nearest neighbours on deterministic pooled image features z(x). |  | 0.9678 | 0.9740 | 0.4797 | 0.4683 |
| nearest_centroid | ok | global | nearest_centroid | 2 | Class prototype baseline in pooled image-feature space. |  | 0.7948 | 0.7665 | 0.4253 | 0.4137 |
| logistic_regression | ok | global | logistic_regression | 3 | Balanced multinomial logistic regression on pooled image features z(x). |  | 0.9877 | 0.9901 | 0.6043 | 0.6303 |
| linear_svm | ok | global | linear_svm | 4 | Balanced linear SVM margin baseline on pooled image features z(x). |  | 0.9908 | 0.9901 | 0.5499 | 0.5894 |
| random_forest_or_gradient_boosting | ok | global | random_forest | 5 | Balanced random forest non-linear classical ML baseline. |  | 0.9897 | 0.9911 | 0.6524 | 0.6786 |
| feature_statistics_fusion | ok | fusion | logistic_regression | 4 | Balanced logistic regression on concatenated z(x) and h(x). |  | 0.9877 | 0.9939 | 0.7748 | 0.8134 |
| hierarchical_classifier | ok | fusion | hierarchical_logistic | 6 | Family-first classifier; subtype prediction is routed by predicted family. |  | 0.9877 | 0.9939 | 0.8671 | 0.9059 |
