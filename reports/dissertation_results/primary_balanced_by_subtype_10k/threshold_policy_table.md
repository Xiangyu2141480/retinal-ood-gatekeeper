# Threshold Policy

research_threshold: computed using test labels for paper evaluation only. deployment_threshold: computed from val ID score quantile only, for UI/demo decision.

| experiment | model | layers | id_count | ood_count | threshold_policy | threshold | source | intended_use | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patchcore_layer2 | patchcore | layer2 | 150 | 1650 | research_threshold | 24.6483 | test labels | paper evaluation | Research only: computed using test labels for paper evaluation only; do not use for UI/demo deployment. |
| patchcore_layer2 | patchcore | layer2 | 150 | 1650 | deployment_threshold | 32.9785 | validation_id_quantile | UI/demo decision |  |
| patchcore_layer3 | patchcore | layer3 | 150 | 1650 | research_threshold | 26.8046 | test labels | paper evaluation | Research only: computed using test labels for paper evaluation only; do not use for UI/demo deployment. |
| patchcore_layer3 | patchcore | layer3 | 150 | 1650 | deployment_threshold | 34.7633 | validation_id_quantile | UI/demo decision |  |
| patchcore_layer2_layer3 | patchcore | layer2+layer3 | 150 | 1650 | research_threshold | 35.7651 | test labels | paper evaluation | Research only: computed using test labels for paper evaluation only; do not use for UI/demo deployment. |
| patchcore_layer2_layer3 | patchcore | layer2+layer3 | 150 | 1650 | deployment_threshold | 44.2789 | validation_id_quantile | UI/demo decision |  |
| autoencoder_baseline | conv_autoencoder |  | 150 | 1650 | research_threshold | 0.0002 | test labels | paper evaluation | Research only: computed using test labels for paper evaluation only; do not use for UI/demo deployment. |
| autoencoder_baseline | conv_autoencoder |  | 150 | 1650 | deployment_threshold | 0.0017 | validation_id_quantile | UI/demo decision |  |
