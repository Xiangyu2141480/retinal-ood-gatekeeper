# Stage 2 Grouped Split Audit

- seed: 42
- algorithm: parent_grouped_stratified_split
- ratios: train=0.6, validation=0.2, test=0.2
- input manifest sha256: 5e2f94755ea052bb46f025384ab81d51d64af4d089708d4212ce6fbeb05a8a63
- train manifest sha256: 7c4ccfce72d1e169351633c341af68b69e1e65a6a9c28b9cb2f45dfe243a4b55
- val manifest sha256: 2250c2694b6f8479523b7fce62cf63476cb0881abb20878946a1b76194644f61
- test manifest sha256: 0f53228034975967e1ed15dba214e71bc89e2d770ea0e8f087f10aaaef059367

## Split Summary

| split | total_rows | unique_image_paths | unique_groups | modality_shift | sensory_artifact | semantic_outlier |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1260 | 1260 | 630 | 240 | 720 | 300 |
| validation | 420 | 420 | 210 | 80 | 240 | 100 |
| test | 420 | 420 | 210 | 80 | 240 | 100 |

## Group Overlap Summary

| overlap_scope | image_path_overlap_count | group_id_overlap_count |
| --- | --- | --- |
| train-val | 0 | 0 |
| train-test | 0 | 0 |
| val-test | 0 | 0 |
| all-three | 0 | 0 |

## Subtype Distribution

| family | subtype | train | validation | test | total |
| --- | --- | --- | --- | --- | --- |
| modality_shift | colour_fundus | 120 | 40 | 40 | 200 |
| modality_shift | oct_screenshot | 120 | 40 | 40 | 200 |
| sensory_artifact | text_watermark | 90 | 30 | 30 | 150 |
| sensory_artifact | rectangle_annotation | 90 | 30 | 30 | 150 |
| sensory_artifact | arrow_annotation | 90 | 30 | 30 | 150 |
| sensory_artifact | composite_layout | 90 | 30 | 30 | 150 |
| sensory_artifact | blur_artifact | 90 | 30 | 30 | 150 |
| sensory_artifact | border_crop | 90 | 30 | 30 | 150 |
| sensory_artifact | gaussian_noise | 90 | 30 | 30 | 150 |
| sensory_artifact | jpeg_compression | 90 | 30 | 30 | 150 |
| semantic_outlier | cifar10_natural | 300 | 100 | 100 | 500 |

shared image paths across splits: 0
shared group IDs across splits: 0
