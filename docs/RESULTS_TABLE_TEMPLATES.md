# Results Table Templates

## Main model comparison

| Model | Backbone | Layers | AUROC ↑ | AUPRC ↑ | FPR@95%TPR ↓ | Inference time | Comment |
|---|---|---|---:|---:|---:|---:|---|
| Autoencoder | Custom CNN | pixel | | | | | baseline |
| PatchCore | ResNet50 | L1 | | | | | shallow feature |
| PatchCore | ResNet50 | L2 | | | | | mid-level |
| PatchCore | ResNet50 | L3 | | | | | mid-level |
| PatchCore | ResNet50 | L2+L3 | | | | | primary |
| PatchCore | ResNet50 | L4 | | | | | deep feature |

## Per-OOD-category performance

| Model | OOD category | AUROC ↑ | AUPRC ↑ | FPR@95%TPR ↓ | Failure pattern |
|---|---|---:|---:|---:|---|
| PatchCore L2+L3 | Modality shift | | | | |
| PatchCore L2+L3 | Sensory artifact | | | | |
| PatchCore L2+L3 | Semantic outlier | | | | |

## Threshold operating points

| Threshold policy | Threshold value | ID false reject rate ↓ | OOD detection rate ↑ | Notes |
|---|---:|---:|---:|---|
| Validation ID 95% quantile | | | | deployment-like |
| Validation ID 99% quantile | | | | conservative |
| FPR@95%TPR threshold | | | 95% | evaluation-only |
