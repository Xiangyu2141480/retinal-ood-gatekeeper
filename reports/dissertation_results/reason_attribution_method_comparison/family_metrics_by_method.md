# Family Metrics by Method

| method | split | status | feature_set | estimator | complexity | family_accuracy | family_argmax_accuracy | family_macro_f1 | family_balanced_accuracy | unknown_threshold | known_coverage_at_gamma | unknown_rate_at_gamma | accuracy_excluding_unknown | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics_logreg | val | ok | statistics | logistic_regression | 1 | 0.9643 | 0.9643 | 0.9619 | 0.9736 | 0.5000 | 0.9976 | 0.0024 | 0.9666 | 0.1319 |
| image_statistics_logreg | test | ok | statistics | logistic_regression | 1 | 0.9476 | 0.9476 | 0.9433 | 0.9647 | 0.5000 | 1.0000 | 0.0000 | 0.9476 | 0.1319 |
| global_feature_knn | val | ok | global | knn | 3 | 0.9714 | 0.9714 | 0.9678 | 0.9600 | 0.5000 | 1.0000 | 0.0000 | 0.9714 | 0.0227 |
| global_feature_knn | test | ok | global | knn | 3 | 0.9762 | 0.9762 | 0.9740 | 0.9686 | 0.5000 | 1.0000 | 0.0000 | 0.9762 | 0.0227 |
| nearest_centroid | val | ok | global | nearest_centroid | 2 | 0.8190 | 0.8190 | 0.7948 | 0.8186 | 0.5000 | 0.9976 | 0.0024 | 0.8210 | 0.0286 |
| nearest_centroid | test | ok | global | nearest_centroid | 2 | 0.7810 | 0.7810 | 0.7665 | 0.8033 | 0.5000 | 0.9976 | 0.0024 | 0.7828 | 0.0286 |
| logistic_regression | val | ok | global | logistic_regression | 3 | 0.9905 | 0.9905 | 0.9877 | 0.9867 | 0.5000 | 1.0000 | 0.0000 | 0.9905 | 5.2737 |
| logistic_regression | test | ok | global | logistic_regression | 3 | 0.9929 | 0.9929 | 0.9901 | 0.9900 | 0.5000 | 1.0000 | 0.0000 | 0.9929 | 5.2737 |
| linear_svm | val | ok | global | linear_svm | 4 | 0.9905 | 0.9929 | 0.9908 | 0.9886 | 0.5000 | 0.9976 | 0.0024 | 0.9928 | 12.6635 |
| linear_svm | test | ok | global | linear_svm | 4 | 0.9881 | 0.9929 | 0.9901 | 0.9872 | 0.5000 | 0.9952 | 0.0048 | 0.9928 | 12.6635 |
| random_forest_or_gradient_boosting | val | ok | global | random_forest | 5 | 0.9905 | 0.9905 | 0.9897 | 0.9850 | 0.5000 | 1.0000 | 0.0000 | 0.9905 | 7.4653 |
| random_forest_or_gradient_boosting | test | ok | global | random_forest | 5 | 0.9905 | 0.9929 | 0.9911 | 0.9867 | 0.5000 | 0.9976 | 0.0024 | 0.9928 | 7.4653 |
| feature_statistics_fusion | val | ok | fusion | logistic_regression | 4 | 0.9905 | 0.9905 | 0.9877 | 0.9867 | 0.5000 | 1.0000 | 0.0000 | 0.9905 | 4.1454 |
| feature_statistics_fusion | test | ok | fusion | logistic_regression | 4 | 0.9952 | 0.9952 | 0.9939 | 0.9933 | 0.5000 | 1.0000 | 0.0000 | 0.9952 | 4.1454 |
| hierarchical_classifier | val | ok | fusion | hierarchical_logistic | 6 | 0.9905 | 0.9905 | 0.9877 | 0.9867 | 0.5000 | 1.0000 | 0.0000 | 0.9905 | 6.4598 |
| hierarchical_classifier | test | ok | fusion | hierarchical_logistic | 6 | 0.9952 | 0.9952 | 0.9939 | 0.9933 | 0.5000 | 1.0000 | 0.0000 | 0.9952 | 6.4598 |
