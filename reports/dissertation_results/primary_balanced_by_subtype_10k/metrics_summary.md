# Experiment Grid Summary

Generated from metrics.json and scores.csv only. Raw images, checkpoints, heatmaps, and patient identifiers are not included.

| experiment | model | backbone | layers | auroc | auprc | fpr_at_95_tpr | threshold_at_95_tpr | threshold | threshold_source | id_false_rejection_rate | total_count | id_count | ood_count | warnings | metrics_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patchcore_layer2 | patchcore | resnet50 | layer2 | 0.8280 | 0.9828 | 0.7867 | 24.6483 | 32.9785 | validation_id_quantile | 0.0600 | 1800 | 150 | 1650 |  | runs/patchcore_layer2/evaluation/metrics.json |
| patchcore_layer3 | patchcore | resnet50 | layer3 | 0.8819 | 0.9880 | 0.6333 | 26.8046 | 34.7633 | validation_id_quantile | 0.0667 | 1800 | 150 | 1650 |  | runs/patchcore_layer3/evaluation/metrics.json |
| patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | 0.8722 | 0.9874 | 0.6867 | 35.7651 | 44.2789 | validation_id_quantile | 0.0733 | 1800 | 150 | 1650 |  | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| autoencoder_baseline | conv_autoencoder |  |  | 0.7649 | 0.9763 | 0.9467 | 0.0002 | 0.0017 | validation_id_quantile | 0.0267 | 1800 | 150 | 1650 |  | runs/autoencoder_baseline/evaluation/metrics.json |
