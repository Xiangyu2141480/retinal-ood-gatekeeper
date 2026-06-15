# Subtype Influence Diagnostics

| scheme | scheme_label | ood_subtype | analysis | auroc | auprc | fpr_at_95_tpr | id_count | ood_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mahalanobis_feature | Mahalanobis feature | arrow_annotation | only_subtype | 0.9968 | 0.9965 | 0.0133 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | arrow_annotation | removed_subtype | 0.9700 | 0.9970 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | blur_artifact | only_subtype | 0.9964 | 0.9962 | 0.0133 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | blur_artifact | removed_subtype | 0.9700 | 0.9970 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | border_crop | only_subtype | 0.9946 | 0.9935 | 0.0133 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | border_crop | removed_subtype | 0.9702 | 0.9970 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | cifar10_natural | only_subtype | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | cifar10_natural | removed_subtype | 0.9697 | 0.9969 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | colour_fundus | only_subtype | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | colour_fundus | removed_subtype | 0.9697 | 0.9969 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | composite_layout | only_subtype | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | composite_layout | removed_subtype | 0.9697 | 0.9969 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | gaussian_noise | only_subtype | 0.9818 | 0.9685 | 0.0667 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | gaussian_noise | removed_subtype | 0.9715 | 0.9971 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | jpeg_compression | only_subtype | 0.9910 | 0.9912 | 0.0467 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | jpeg_compression | removed_subtype | 0.9706 | 0.9970 | 0.2200 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | oct_screenshot | only_subtype | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | oct_screenshot | removed_subtype | 0.9697 | 0.9969 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | rectangle_annotation | only_subtype | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | rectangle_annotation | removed_subtype | 0.9697 | 0.9969 | 0.2267 | 150 | 1500 |
| mahalanobis_feature | Mahalanobis feature | text_watermark | only_subtype | 0.7361 | 0.6876 | 0.6467 | 150 | 150 |
| mahalanobis_feature | Mahalanobis feature | text_watermark | removed_subtype | 0.9961 | 0.9996 | 0.0133 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | arrow_annotation | only_subtype | 0.9984 | 0.9983 | 0.0133 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | arrow_annotation | removed_subtype | 0.8703 | 0.9853 | 0.6333 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | blur_artifact | only_subtype | 0.5776 | 0.5519 | 0.9000 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | blur_artifact | removed_subtype | 0.9124 | 0.9903 | 0.5133 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | border_crop | only_subtype | 0.8779 | 0.8682 | 0.4467 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | border_crop | removed_subtype | 0.8823 | 0.9871 | 0.6333 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | cifar10_natural | only_subtype | 0.9918 | 0.9903 | 0.0400 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | cifar10_natural | removed_subtype | 0.8709 | 0.9855 | 0.6333 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | colour_fundus | only_subtype | 0.9696 | 0.9336 | 0.0667 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | colour_fundus | removed_subtype | 0.8732 | 0.9859 | 0.6333 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | composite_layout | only_subtype | 0.9931 | 0.9904 | 0.0133 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | composite_layout | removed_subtype | 0.8708 | 0.9855 | 0.6333 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | gaussian_noise | only_subtype | 0.7501 | 0.7490 | 0.7400 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | gaussian_noise | removed_subtype | 0.8951 | 0.9885 | 0.6200 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | jpeg_compression | only_subtype | 0.5979 | 0.5539 | 0.8200 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | jpeg_compression | removed_subtype | 0.9103 | 0.9902 | 0.5333 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | oct_screenshot | only_subtype | 0.9948 | 0.9935 | 0.0133 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | oct_screenshot | removed_subtype | 0.8706 | 0.9854 | 0.6333 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | rectangle_annotation | only_subtype | 0.9929 | 0.9910 | 0.0200 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | rectangle_annotation | removed_subtype | 0.8708 | 0.9855 | 0.6333 | 150 | 1500 |
| patchcore_l3 | PatchCore L3 | text_watermark | only_subtype | 0.9572 | 0.9467 | 0.2000 | 150 | 150 |
| patchcore_l3 | PatchCore L3 | text_watermark | removed_subtype | 0.8744 | 0.9861 | 0.6333 | 150 | 1500 |
| autoencoder | Autoencoder | arrow_annotation | only_subtype | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| autoencoder | Autoencoder | arrow_annotation | removed_subtype | 0.7414 | 0.9709 | 0.9533 | 150 | 1500 |
| autoencoder | Autoencoder | blur_artifact | only_subtype | 0.1116 | 0.3253 | 1.0000 | 150 | 150 |
| autoencoder | Autoencoder | blur_artifact | removed_subtype | 0.8302 | 0.9817 | 0.8333 | 150 | 1500 |
| autoencoder | Autoencoder | border_crop | only_subtype | 0.4336 | 0.4550 | 0.9600 | 150 | 150 |
| autoencoder | Autoencoder | border_crop | removed_subtype | 0.7980 | 0.9784 | 0.9400 | 150 | 1500 |
| autoencoder | Autoencoder | cifar10_natural | only_subtype | 0.9809 | 0.9854 | 0.0867 | 150 | 150 |
| autoencoder | Autoencoder | cifar10_natural | removed_subtype | 0.7433 | 0.9712 | 0.9533 | 150 | 1500 |
| autoencoder | Autoencoder | colour_fundus | only_subtype | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| autoencoder | Autoencoder | colour_fundus | removed_subtype | 0.7414 | 0.9709 | 0.9533 | 150 | 1500 |
| autoencoder | Autoencoder | composite_layout | only_subtype | 0.9932 | 0.9827 | 0.0133 | 150 | 150 |
| autoencoder | Autoencoder | composite_layout | removed_subtype | 0.7421 | 0.9711 | 0.9533 | 150 | 1500 |
| autoencoder | Autoencoder | gaussian_noise | only_subtype | 0.8740 | 0.8622 | 0.5067 | 150 | 150 |
| autoencoder | Autoencoder | gaussian_noise | removed_subtype | 0.7540 | 0.9729 | 0.9533 | 150 | 1500 |
| autoencoder | Autoencoder | jpeg_compression | only_subtype | 0.5057 | 0.5137 | 0.9600 | 150 | 150 |
| autoencoder | Autoencoder | jpeg_compression | removed_subtype | 0.7908 | 0.9775 | 0.9400 | 150 | 1500 |
| autoencoder | Autoencoder | oct_screenshot | only_subtype | 0.9854 | 0.9778 | 0.0267 | 150 | 150 |
| autoencoder | Autoencoder | oct_screenshot | removed_subtype | 0.7428 | 0.9712 | 0.9533 | 150 | 1500 |
| autoencoder | Autoencoder | rectangle_annotation | only_subtype | 1.0000 | 1.0000 | 0.0000 | 150 | 150 |
| autoencoder | Autoencoder | rectangle_annotation | removed_subtype | 0.7414 | 0.9709 | 0.9533 | 150 | 1500 |
| autoencoder | Autoencoder | text_watermark | only_subtype | 0.5295 | 0.5292 | 0.9067 | 150 | 150 |
| autoencoder | Autoencoder | text_watermark | removed_subtype | 0.7884 | 0.9773 | 0.9533 | 150 | 1500 |
| image_statistics | Image statistics | arrow_annotation | only_subtype | 0.7120 | 0.6707 | 0.6600 | 150 | 150 |
| image_statistics | Image statistics | arrow_annotation | removed_subtype | 0.7737 | 0.9702 | 0.7600 | 150 | 1500 |
| image_statistics | Image statistics | blur_artifact | only_subtype | 0.6533 | 0.6385 | 0.7800 | 150 | 150 |
| image_statistics | Image statistics | blur_artifact | removed_subtype | 0.7796 | 0.9709 | 0.7467 | 150 | 1500 |
| image_statistics | Image statistics | border_crop | only_subtype | 0.6991 | 0.6349 | 0.6867 | 150 | 150 |
| image_statistics | Image statistics | border_crop | removed_subtype | 0.7750 | 0.9705 | 0.7600 | 150 | 1500 |
| image_statistics | Image statistics | cifar10_natural | only_subtype | 0.9972 | 0.9971 | 0.0133 | 150 | 150 |
| image_statistics | Image statistics | cifar10_natural | removed_subtype | 0.7452 | 0.9615 | 0.7600 | 150 | 1500 |
| image_statistics | Image statistics | colour_fundus | only_subtype | 0.8364 | 0.7733 | 0.4467 | 150 | 150 |
| image_statistics | Image statistics | colour_fundus | removed_subtype | 0.7613 | 0.9685 | 0.7600 | 150 | 1500 |
| image_statistics | Image statistics | composite_layout | only_subtype | 0.9538 | 0.9119 | 0.0733 | 150 | 150 |
| image_statistics | Image statistics | composite_layout | removed_subtype | 0.7496 | 0.9662 | 0.7600 | 150 | 1500 |
| image_statistics | Image statistics | gaussian_noise | only_subtype | 0.8683 | 0.8211 | 0.5133 | 150 | 150 |
| image_statistics | Image statistics | gaussian_noise | removed_subtype | 0.7581 | 0.9679 | 0.7600 | 150 | 1500 |
| image_statistics | Image statistics | jpeg_compression | only_subtype | 0.5104 | 0.5218 | 0.9467 | 150 | 150 |
| image_statistics | Image statistics | jpeg_compression | removed_subtype | 0.7939 | 0.9726 | 0.6867 | 150 | 1500 |
| image_statistics | Image statistics | oct_screenshot | only_subtype | 0.9710 | 0.9201 | 0.0733 | 150 | 150 |
| image_statistics | Image statistics | oct_screenshot | removed_subtype | 0.7478 | 0.9656 | 0.7600 | 150 | 1500 |
| image_statistics | Image statistics | rectangle_annotation | only_subtype | 0.7349 | 0.6786 | 0.6133 | 150 | 150 |
| image_statistics | Image statistics | rectangle_annotation | removed_subtype | 0.7714 | 0.9700 | 0.7600 | 150 | 1500 |
| image_statistics | Image statistics | text_watermark | only_subtype | 0.5129 | 0.5271 | 0.9533 | 150 | 150 |
| image_statistics | Image statistics | text_watermark | removed_subtype | 0.7936 | 0.9725 | 0.6867 | 150 | 1500 |
