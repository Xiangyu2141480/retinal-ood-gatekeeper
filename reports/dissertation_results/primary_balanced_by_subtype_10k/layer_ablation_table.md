# PatchCore Layer Ablation

PatchCore-only comparison of feature layers.

| experiment | model | backbone | layers | auroc | auprc | fpr_at_95_tpr | threshold_at_95_tpr | threshold | threshold_source | id_false_rejection_rate | id_count | ood_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patchcore_layer2 | patchcore | resnet50 | layer2 | 0.8280 | 0.9828 | 0.7867 | 24.6483 | 32.9785 | validation_id_quantile | 0.0600 | 150 | 1650 |
| patchcore_layer3 | patchcore | resnet50 | layer3 | 0.8819 | 0.9880 | 0.6333 | 26.8046 | 34.7633 | validation_id_quantile | 0.0667 | 150 | 1650 |
| patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | 0.8722 | 0.9874 | 0.6867 | 35.7651 | 44.2789 | validation_id_quantile | 0.0733 | 150 | 1650 |
