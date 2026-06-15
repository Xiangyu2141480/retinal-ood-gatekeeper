# Artifact Severity Stress Test

| scheme | artifact_type | severity_value | mean_score | median_score | deployment_reject_rate | research_reject_rate | n_images | severity_score_spearman | scheme_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autoencoder | blur_artifact | 3.0000 | 0.0005 | 0.0004 | 0.0000 | 0.8333 | 12 | -0.7506 | Autoencoder |
| autoencoder | blur_artifact | 5.0000 | 0.0003 | 0.0003 | 0.0000 | 0.7500 | 12 | -0.7506 | Autoencoder |
| autoencoder | blur_artifact | 9.0000 | 0.0001 | 0.0001 | 0.0000 | 0.2500 | 12 | -0.7506 | Autoencoder |
| autoencoder | blur_artifact | 15.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 12 | -0.7506 | Autoencoder |
| autoencoder | border_crop | 0.0200 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | -0.1722 | Autoencoder |
| autoencoder | border_crop | 0.0500 | 0.0008 | 0.0008 | 0.0000 | 0.9167 | 12 | -0.1722 | Autoencoder |
| autoencoder | border_crop | 0.1000 | 0.0008 | 0.0007 | 0.0000 | 0.9167 | 12 | -0.1722 | Autoencoder |
| autoencoder | border_crop | 0.2000 | 0.0005 | 0.0005 | 0.0000 | 0.8333 | 12 | -0.1722 | Autoencoder |
| autoencoder | gaussian_noise | 5.0000 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.7129 | Autoencoder |
| autoencoder | gaussian_noise | 10.0000 | 0.0008 | 0.0008 | 0.0000 | 1.0000 | 12 | 0.7129 | Autoencoder |
| autoencoder | gaussian_noise | 20.0000 | 0.0012 | 0.0012 | 0.1667 | 1.0000 | 12 | 0.7129 | Autoencoder |
| autoencoder | gaussian_noise | 40.0000 | 0.0023 | 0.0022 | 0.7500 | 1.0000 | 12 | 0.7129 | Autoencoder |
| autoencoder | jpeg_compression | 10.0000 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.0163 | Autoencoder |
| autoencoder | jpeg_compression | 30.0000 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.0163 | Autoencoder |
| autoencoder | jpeg_compression | 50.0000 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.0163 | Autoencoder |
| autoencoder | jpeg_compression | 70.0000 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.0163 | Autoencoder |
| autoencoder | jpeg_compression | 90.0000 | 0.0007 | 0.0007 | 0.0000 | 0.8333 | 12 | 0.0163 | Autoencoder |
| autoencoder | rectangle_annotation | 2.0000 | 0.0020 | 0.0019 | 0.8333 | 1.0000 | 12 | 0.9685 | Autoencoder |
| autoencoder | rectangle_annotation | 5.0000 | 0.0056 | 0.0055 | 1.0000 | 1.0000 | 12 | 0.9685 | Autoencoder |
| autoencoder | rectangle_annotation | 9.0000 | 0.0101 | 0.0100 | 1.0000 | 1.0000 | 12 | 0.9685 | Autoencoder |
| autoencoder | rectangle_annotation | 15.0000 | 0.0166 | 0.0166 | 1.0000 | 1.0000 | 12 | 0.9685 | Autoencoder |
| autoencoder | text_watermark | 0.1000 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.0807 | Autoencoder |
| autoencoder | text_watermark | 0.2500 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.0807 | Autoencoder |
| autoencoder | text_watermark | 0.5000 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.0807 | Autoencoder |
| autoencoder | text_watermark | 0.7500 | 0.0007 | 0.0007 | 0.0000 | 0.9167 | 12 | 0.0807 | Autoencoder |
| image_statistics | blur_artifact | 3.0000 | 0.9620 | 0.8587 | 0.0833 | 1.0000 | 12 | 0.3699 | Image statistics |
| image_statistics | blur_artifact | 5.0000 | 1.0573 | 0.9594 | 0.0833 | 1.0000 | 12 | 0.3699 | Image statistics |
| image_statistics | blur_artifact | 9.0000 | 1.2041 | 1.1005 | 0.0833 | 1.0000 | 12 | 0.3699 | Image statistics |
| image_statistics | blur_artifact | 15.0000 | 1.3071 | 1.2177 | 0.0833 | 1.0000 | 12 | 0.3699 | Image statistics |
| image_statistics | border_crop | 0.0200 | 1.2762 | 1.2240 | 0.0833 | 1.0000 | 12 | 0.8205 | Image statistics |
| image_statistics | border_crop | 0.0500 | 1.2998 | 1.2496 | 0.0833 | 1.0000 | 12 | 0.8205 | Image statistics |
| image_statistics | border_crop | 0.1000 | 1.4906 | 1.4636 | 0.0833 | 1.0000 | 12 | 0.8205 | Image statistics |
| image_statistics | border_crop | 0.2000 | 2.5053 | 2.4863 | 1.0000 | 1.0000 | 12 | 0.8205 | Image statistics |
| image_statistics | gaussian_noise | 5.0000 | 0.8795 | 0.8189 | 0.0000 | 0.8333 | 12 | 0.7250 | Image statistics |
| image_statistics | gaussian_noise | 10.0000 | 0.8730 | 0.8306 | 0.0000 | 0.8333 | 12 | 0.7250 | Image statistics |
| image_statistics | gaussian_noise | 20.0000 | 1.0468 | 1.0336 | 0.0000 | 1.0000 | 12 | 0.7250 | Image statistics |
| image_statistics | gaussian_noise | 40.0000 | 2.2567 | 2.2379 | 1.0000 | 1.0000 | 12 | 0.7250 | Image statistics |
| image_statistics | jpeg_compression | 10.0000 | 0.8937 | 0.8316 | 0.0000 | 0.8333 | 12 | -0.0027 | Image statistics |
| image_statistics | jpeg_compression | 30.0000 | 0.8910 | 0.8224 | 0.0000 | 0.8333 | 12 | -0.0027 | Image statistics |
| image_statistics | jpeg_compression | 50.0000 | 0.8937 | 0.8371 | 0.0000 | 0.8333 | 12 | -0.0027 | Image statistics |
| image_statistics | jpeg_compression | 70.0000 | 0.8882 | 0.8293 | 0.0000 | 0.8333 | 12 | -0.0027 | Image statistics |
| image_statistics | jpeg_compression | 90.0000 | 1.0293 | 0.8268 | 0.0833 | 0.9167 | 12 | -0.0027 | Image statistics |
| image_statistics | rectangle_annotation | 2.0000 | 1.0215 | 0.8444 | 0.0833 | 0.8333 | 12 | 0.1856 | Image statistics |
| image_statistics | rectangle_annotation | 5.0000 | 1.1550 | 0.9942 | 0.1667 | 1.0000 | 12 | 0.1856 | Image statistics |
| image_statistics | rectangle_annotation | 9.0000 | 1.1605 | 0.9784 | 0.1667 | 1.0000 | 12 | 0.1856 | Image statistics |
| image_statistics | rectangle_annotation | 15.0000 | 1.1927 | 0.9907 | 0.1667 | 1.0000 | 12 | 0.1856 | Image statistics |
| image_statistics | text_watermark | 0.1000 | 0.8945 | 0.8313 | 0.0000 | 0.8333 | 12 | 0.0309 | Image statistics |
| image_statistics | text_watermark | 0.2500 | 0.8934 | 0.8313 | 0.0000 | 0.8333 | 12 | 0.0309 | Image statistics |
| image_statistics | text_watermark | 0.5000 | 0.8988 | 0.8317 | 0.0833 | 0.8333 | 12 | 0.0309 | Image statistics |
| image_statistics | text_watermark | 0.7500 | 0.9280 | 0.8318 | 0.0833 | 0.8333 | 12 | 0.0309 | Image statistics |
| mahalanobis_feature | blur_artifact | 3.0000 | 34.4966 | 34.9346 | 0.0833 | 0.7500 | 12 | 0.9644 | Mahalanobis feature |
| mahalanobis_feature | blur_artifact | 5.0000 | 53.3489 | 54.4013 | 1.0000 | 1.0000 | 12 | 0.9644 | Mahalanobis feature |
| mahalanobis_feature | blur_artifact | 9.0000 | 95.3981 | 97.9370 | 1.0000 | 1.0000 | 12 | 0.9644 | Mahalanobis feature |
| mahalanobis_feature | blur_artifact | 15.0000 | 128.5198 | 132.0983 | 1.0000 | 1.0000 | 12 | 0.9644 | Mahalanobis feature |
| mahalanobis_feature | border_crop | 0.0200 | 42.5613 | 44.1271 | 0.7500 | 1.0000 | 12 | 0.9685 | Mahalanobis feature |
| mahalanobis_feature | border_crop | 0.0500 | 69.3958 | 70.4740 | 1.0000 | 1.0000 | 12 | 0.9685 | Mahalanobis feature |
| mahalanobis_feature | border_crop | 0.1000 | 107.3052 | 105.9069 | 1.0000 | 1.0000 | 12 | 0.9685 | Mahalanobis feature |
| mahalanobis_feature | border_crop | 0.2000 | 129.7519 | 128.1739 | 1.0000 | 1.0000 | 12 | 0.9685 | Mahalanobis feature |
| mahalanobis_feature | gaussian_noise | 5.0000 | 28.4000 | 27.7196 | 0.0000 | 0.1667 | 12 | 0.9268 | Mahalanobis feature |
| mahalanobis_feature | gaussian_noise | 10.0000 | 33.4181 | 32.8908 | 0.0833 | 0.6667 | 12 | 0.9268 | Mahalanobis feature |
| mahalanobis_feature | gaussian_noise | 20.0000 | 42.0022 | 42.2626 | 0.8333 | 1.0000 | 12 | 0.9268 | Mahalanobis feature |
| mahalanobis_feature | gaussian_noise | 40.0000 | 50.3876 | 49.7434 | 1.0000 | 1.0000 | 12 | 0.9268 | Mahalanobis feature |
| mahalanobis_feature | jpeg_compression | 10.0000 | 25.0593 | 24.7515 | 0.0000 | 0.0833 | 12 | 0.7234 | Mahalanobis feature |
| mahalanobis_feature | jpeg_compression | 30.0000 | 25.4244 | 25.3789 | 0.0000 | 0.0833 | 12 | 0.7234 | Mahalanobis feature |
| mahalanobis_feature | jpeg_compression | 50.0000 | 28.7579 | 28.7103 | 0.0000 | 0.1667 | 12 | 0.7234 | Mahalanobis feature |
| mahalanobis_feature | jpeg_compression | 70.0000 | 31.1509 | 29.3650 | 0.0833 | 0.3333 | 12 | 0.7234 | Mahalanobis feature |
| mahalanobis_feature | jpeg_compression | 90.0000 | 113.4644 | 79.1916 | 1.0000 | 1.0000 | 12 | 0.7234 | Mahalanobis feature |
| mahalanobis_feature | rectangle_annotation | 2.0000 | 75.4514 | 67.5344 | 1.0000 | 1.0000 | 12 | 0.5205 | Mahalanobis feature |
| mahalanobis_feature | rectangle_annotation | 5.0000 | 90.8643 | 86.6157 | 1.0000 | 1.0000 | 12 | 0.5205 | Mahalanobis feature |
| mahalanobis_feature | rectangle_annotation | 9.0000 | 98.4417 | 96.8067 | 1.0000 | 1.0000 | 12 | 0.5205 | Mahalanobis feature |
| mahalanobis_feature | rectangle_annotation | 15.0000 | 108.0134 | 107.5844 | 1.0000 | 1.0000 | 12 | 0.5205 | Mahalanobis feature |
| mahalanobis_feature | text_watermark | 0.1000 | 24.7870 | 24.7199 | 0.0000 | 0.0833 | 12 | 0.1776 | Mahalanobis feature |
| mahalanobis_feature | text_watermark | 0.2500 | 25.1426 | 24.6562 | 0.0000 | 0.0833 | 12 | 0.1776 | Mahalanobis feature |
| mahalanobis_feature | text_watermark | 0.5000 | 26.0038 | 24.9467 | 0.0000 | 0.1667 | 12 | 0.1776 | Mahalanobis feature |
| mahalanobis_feature | text_watermark | 0.7500 | 26.7865 | 25.5873 | 0.0000 | 0.1667 | 12 | 0.1776 | Mahalanobis feature |
| patchcore_l3 | blur_artifact | 3.0000 | 28.1177 | 27.3606 | 0.0000 | 0.5833 | 12 | 0.4143 | PatchCore L3 |
| patchcore_l3 | blur_artifact | 5.0000 | 28.8821 | 28.0533 | 0.1667 | 0.6667 | 12 | 0.4143 | PatchCore L3 |
| patchcore_l3 | blur_artifact | 9.0000 | 30.9456 | 29.4350 | 0.1667 | 1.0000 | 12 | 0.4143 | PatchCore L3 |
| patchcore_l3 | blur_artifact | 15.0000 | 32.8542 | 31.9863 | 0.1667 | 1.0000 | 12 | 0.4143 | PatchCore L3 |
| patchcore_l3 | border_crop | 0.0200 | 30.5970 | 30.1815 | 0.0000 | 1.0000 | 12 | 0.7775 | PatchCore L3 |
| patchcore_l3 | border_crop | 0.0500 | 32.3968 | 32.3976 | 0.0000 | 1.0000 | 12 | 0.7775 | PatchCore L3 |
| patchcore_l3 | border_crop | 0.1000 | 34.6041 | 34.8576 | 0.5000 | 1.0000 | 12 | 0.7775 | PatchCore L3 |
| patchcore_l3 | border_crop | 0.2000 | 35.6614 | 35.9581 | 0.7500 | 1.0000 | 12 | 0.7775 | PatchCore L3 |
| patchcore_l3 | gaussian_noise | 5.0000 | 27.8596 | 27.5224 | 0.0000 | 0.5833 | 12 | 0.6537 | PatchCore L3 |
| patchcore_l3 | gaussian_noise | 10.0000 | 27.8461 | 27.7101 | 0.0000 | 0.6667 | 12 | 0.6537 | PatchCore L3 |
| patchcore_l3 | gaussian_noise | 20.0000 | 39.1737 | 38.3020 | 0.5833 | 1.0000 | 12 | 0.6537 | PatchCore L3 |
| patchcore_l3 | gaussian_noise | 40.0000 | 35.4088 | 34.6860 | 0.5000 | 1.0000 | 12 | 0.6537 | PatchCore L3 |
| patchcore_l3 | jpeg_compression | 10.0000 | 27.8297 | 27.9356 | 0.0000 | 0.5833 | 12 | 0.1674 | PatchCore L3 |
| patchcore_l3 | jpeg_compression | 30.0000 | 27.9132 | 27.8900 | 0.0000 | 0.5833 | 12 | 0.1674 | PatchCore L3 |
| patchcore_l3 | jpeg_compression | 50.0000 | 28.7468 | 29.0519 | 0.0000 | 0.6667 | 12 | 0.1674 | PatchCore L3 |
| patchcore_l3 | jpeg_compression | 70.0000 | 28.5444 | 28.2714 | 0.0833 | 0.5000 | 12 | 0.1674 | PatchCore L3 |
| patchcore_l3 | jpeg_compression | 90.0000 | 30.1454 | 29.0677 | 0.0833 | 0.9167 | 12 | 0.1674 | PatchCore L3 |
| patchcore_l3 | rectangle_annotation | 2.0000 | 39.3365 | 38.8035 | 0.9167 | 1.0000 | 12 | 0.3995 | PatchCore L3 |
| patchcore_l3 | rectangle_annotation | 5.0000 | 41.5961 | 40.9357 | 1.0000 | 1.0000 | 12 | 0.3995 | PatchCore L3 |
| patchcore_l3 | rectangle_annotation | 9.0000 | 42.6413 | 42.2144 | 1.0000 | 1.0000 | 12 | 0.3995 | PatchCore L3 |
| patchcore_l3 | rectangle_annotation | 15.0000 | 44.8117 | 44.5364 | 1.0000 | 1.0000 | 12 | 0.3995 | PatchCore L3 |
| patchcore_l3 | text_watermark | 0.1000 | 27.9416 | 27.8826 | 0.0000 | 0.5833 | 12 | 0.5259 | PatchCore L3 |
| patchcore_l3 | text_watermark | 0.2500 | 29.3218 | 29.9511 | 0.0000 | 0.7500 | 12 | 0.5259 | PatchCore L3 |
| patchcore_l3 | text_watermark | 0.5000 | 31.8688 | 32.7990 | 0.1667 | 0.8333 | 12 | 0.5259 | PatchCore L3 |
| patchcore_l3 | text_watermark | 0.7500 | 34.5888 | 33.2687 | 0.4167 | 1.0000 | 12 | 0.5259 | PatchCore L3 |
