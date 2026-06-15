# Autoencoder vs PatchCore

Baseline comparison table. PatchCore L2+L3 is preferred when available.

| method | experiment | model | backbone | layers | auroc | auprc | fpr_at_95_tpr | threshold_at_95_tpr | threshold | threshold_source | id_false_rejection_rate | id_count | ood_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoencoder | autoencoder_baseline | conv_autoencoder |  |  | 0.7649 | 0.9763 | 0.9467 | 0.0002 | 0.0017 | validation_id_quantile | 0.0267 | 150 | 1650 |
| PatchCore | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | 0.8722 | 0.9874 | 0.6867 | 35.7651 | 44.2789 | validation_id_quantile | 0.0733 | 150 | 1650 |
