# Per-Category Metrics

Per-OOD category/subtype metrics for dissertation comparison.

| category_level | category | experiment | model | backbone | layers | ood_type | ood_subtype | auroc | auprc | fpr_at_95_tpr | id_count | ood_count | metrics_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ood_subtype | arrow_annotation | patchcore_layer2 | patchcore | resnet50 | layer2 | sensory_artifact | arrow_annotation | 0.9999 | 0.9999 | 0.0000 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | blur_artifact | patchcore_layer2 | patchcore | resnet50 | layer2 | sensory_artifact | blur_artifact | 0.4927 | 0.4813 | 0.9067 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | border_crop | patchcore_layer2 | patchcore | resnet50 | layer2 | sensory_artifact | border_crop | 0.7492 | 0.7766 | 0.7800 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | cifar10_natural | patchcore_layer2 | patchcore | resnet50 | layer2 | semantic_outlier | cifar10_natural | 0.9392 | 0.9201 | 0.2000 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | colour_fundus | patchcore_layer2 | patchcore | resnet50 | layer2 | modality_shift | colour_fundus | 0.9936 | 0.9933 | 0.0400 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | composite_layout | patchcore_layer2 | patchcore | resnet50 | layer2 | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | gaussian_noise | patchcore_layer2 | patchcore | resnet50 | layer2 | sensory_artifact | gaussian_noise | 0.4256 | 0.4481 | 0.9467 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | jpeg_compression | patchcore_layer2 | patchcore | resnet50 | layer2 | sensory_artifact | jpeg_compression | 0.5585 | 0.5207 | 0.8533 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | oct_screenshot | patchcore_layer2 | patchcore | resnet50 | layer2 | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | rectangle_annotation | patchcore_layer2 | patchcore | resnet50 | layer2 | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | text_watermark | patchcore_layer2 | patchcore | resnet50 | layer2 | sensory_artifact | text_watermark | 0.9492 | 0.9485 | 0.2267 | 150 | 150 | runs/patchcore_layer2/evaluation/metrics.json |
| ood_subtype | arrow_annotation | patchcore_layer3 | patchcore | resnet50 | layer3 | sensory_artifact | arrow_annotation | 0.9984 | 0.9983 | 0.0133 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | blur_artifact | patchcore_layer3 | patchcore | resnet50 | layer3 | sensory_artifact | blur_artifact | 0.5776 | 0.5519 | 0.9000 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | border_crop | patchcore_layer3 | patchcore | resnet50 | layer3 | sensory_artifact | border_crop | 0.8779 | 0.8682 | 0.4467 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | cifar10_natural | patchcore_layer3 | patchcore | resnet50 | layer3 | semantic_outlier | cifar10_natural | 0.9918 | 0.9903 | 0.0400 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | colour_fundus | patchcore_layer3 | patchcore | resnet50 | layer3 | modality_shift | colour_fundus | 0.9696 | 0.9336 | 0.0667 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | composite_layout | patchcore_layer3 | patchcore | resnet50 | layer3 | sensory_artifact | composite_layout | 0.9931 | 0.9904 | 0.0133 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | gaussian_noise | patchcore_layer3 | patchcore | resnet50 | layer3 | sensory_artifact | gaussian_noise | 0.7501 | 0.7490 | 0.7400 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | jpeg_compression | patchcore_layer3 | patchcore | resnet50 | layer3 | sensory_artifact | jpeg_compression | 0.5979 | 0.5539 | 0.8200 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | oct_screenshot | patchcore_layer3 | patchcore | resnet50 | layer3 | modality_shift | oct_screenshot | 0.9948 | 0.9935 | 0.0133 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | rectangle_annotation | patchcore_layer3 | patchcore | resnet50 | layer3 | sensory_artifact | rectangle_annotation | 0.9929 | 0.9910 | 0.0200 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | text_watermark | patchcore_layer3 | patchcore | resnet50 | layer3 | sensory_artifact | text_watermark | 0.9572 | 0.9467 | 0.2000 | 150 | 150 | runs/patchcore_layer3/evaluation/metrics.json |
| ood_subtype | arrow_annotation | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | sensory_artifact | arrow_annotation | 0.9994 | 0.9994 | 0.0067 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | blur_artifact | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | sensory_artifact | blur_artifact | 0.5531 | 0.5294 | 0.8400 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | border_crop | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | sensory_artifact | border_crop | 0.8884 | 0.8946 | 0.5267 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | cifar10_natural | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | semantic_outlier | cifar10_natural | 0.9912 | 0.9899 | 0.0267 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | colour_fundus | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | modality_shift | colour_fundus | 0.9837 | 0.9725 | 0.0733 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | composite_layout | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | sensory_artifact | composite_layout | 1.0000 | 1.0000 | 0.0000 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | gaussian_noise | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | sensory_artifact | gaussian_noise | 0.6432 | 0.6496 | 0.8000 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | jpeg_compression | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | sensory_artifact | jpeg_compression | 0.5756 | 0.5411 | 0.8667 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | oct_screenshot | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | modality_shift | oct_screenshot | 1.0000 | 1.0000 | 0.0000 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | rectangle_annotation | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | sensory_artifact | rectangle_annotation | 0.9999 | 0.9999 | 0.0000 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | text_watermark | patchcore_layer2_layer3 | patchcore | resnet50 | layer2+layer3 | sensory_artifact | text_watermark | 0.9594 | 0.9517 | 0.1400 | 150 | 150 | runs/patchcore_layer2_layer3/evaluation/metrics.json |
| ood_subtype | arrow_annotation | autoencoder_baseline | conv_autoencoder |  |  | sensory_artifact | arrow_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | blur_artifact | autoencoder_baseline | conv_autoencoder |  |  | sensory_artifact | blur_artifact | 0.1116 | 0.3253 | 1.0000 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | border_crop | autoencoder_baseline | conv_autoencoder |  |  | sensory_artifact | border_crop | 0.4336 | 0.4550 | 0.9600 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | cifar10_natural | autoencoder_baseline | conv_autoencoder |  |  | semantic_outlier | cifar10_natural | 0.9809 | 0.9854 | 0.0867 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | colour_fundus | autoencoder_baseline | conv_autoencoder |  |  | modality_shift | colour_fundus | 1.0000 | 1.0000 | 0.0000 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | composite_layout | autoencoder_baseline | conv_autoencoder |  |  | sensory_artifact | composite_layout | 0.9932 | 0.9827 | 0.0133 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | gaussian_noise | autoencoder_baseline | conv_autoencoder |  |  | sensory_artifact | gaussian_noise | 0.8740 | 0.8622 | 0.5067 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | jpeg_compression | autoencoder_baseline | conv_autoencoder |  |  | sensory_artifact | jpeg_compression | 0.5057 | 0.5137 | 0.9600 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | oct_screenshot | autoencoder_baseline | conv_autoencoder |  |  | modality_shift | oct_screenshot | 0.9854 | 0.9778 | 0.0267 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | rectangle_annotation | autoencoder_baseline | conv_autoencoder |  |  | sensory_artifact | rectangle_annotation | 1.0000 | 1.0000 | 0.0000 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
| ood_subtype | text_watermark | autoencoder_baseline | conv_autoencoder |  |  | sensory_artifact | text_watermark | 0.5295 | 0.5292 | 0.9067 | 150 | 150 | runs/autoencoder_baseline/evaluation/metrics.json |
