# Vincent Feedback Code, Split, and Dissertation Consistency Audit

Audit target: the dissertation manuscript repository associated with this evidence package.
Evidence source: local `Xiangyu2141480/retinal-ood-gatekeeper` checkout, `main` at commit `cb4b975`.

> Scope: this audit only adds this Markdown report and two audit CSV files. It does not modify formal manifests, model code, committed experiment outputs, or `main.tex`.

## 1. Executive summary

- Stage 1 remains an ID-only unsupervised OOD gatekeeper. Evidence: `configs/datasets/dissertation_dataset_v1.yaml:8-16`, `src/retinal_ood/data/dissertation_dataset.py:109-117`, `src/retinal_ood/models/patchcore.py:36-41,62-92`, `src/retinal_ood/baselines/feature_distance.py:62-82`, and `src/retinal_ood/models/autoencoder.py:85-127`.
- OOD labels are evaluation-only for Stage 1 and supervised targets only for optional Stage 2 explanation. Evidence: `main.tex:1473-1477`, `main.tex:3012-3018`, and `src/retinal_ood/reason_attribution/comparison.py:131-145,151-190`.
- Principal package counts match the dissertation: 1,000 synthetic FAF-like ID images plus 2,100 OOD images, with Stage 2 manifests reusing OOD rows rather than adding new unique images.
- The OOD taxonomy is exactly 3 families and 11 subtypes. Sensory artifacts are derived from parent synthetic FAF images; modality and semantic rows do not carry parent hashes in the final manifests.
- Stage 2 reason splits are disjoint by `image_path` but not by `parent_image_hash`: train-val 127 shared parent hashes, train-test 128, val-test 108, and 108 shared by all three splits. This can make Stage 2 artifact-family and artifact-subtype metrics optimistic.
- `k=1` is used by both global feature kNN and all final PatchCore configs. No committed k-ablation was found, so k=1 should be described as a fixed setting, not an empirically optimal value.
- Stage 2 family selection is validation-driven: `linear_svm` is selected for family attribution; `hierarchical_classifier` is selected for subtype attribution and is non-oracle because it routes by predicted family.

## 2. Dataset lineage and revised Table 3

Primary image split means a manifest row whose image is copied/imported as a standalone packaged image and has no non-empty `parent_image_hash`. Derived image means a generated row with a non-empty `parent_image_hash`, currently the generated sensory-artifact rows. Derived manifest view means a CSV that samples or partitions existing images without creating new image files, such as balanced OOD views and Stage 2 reason splits. Evidence: `src/retinal_ood/data/dissertation_dataset.py:620-634`, `src/retinal_ood/data/dissertation_dataset.py:690-704`, `src/retinal_ood/data/dissertation_dataset.py:194-213`, and `scripts/build_reason_attribution_manifests.py:53-90`.

### Revised Table 3-ready table
| Dataset / split | Stage | Primary or derived | Parent/source | Role | Labels used | Number of images |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 1 nominal training set | Stage 1 | Primary image split | UCL SynthEye synthetic FAF=700 | Fits Stage 1 nominal references only | label=0 and ood_type=id enforce ID-only fitting | 700 |
| Stage 1 ID validation/calibration set | Stage 1 | Primary image split | UCL SynthEye synthetic FAF=150 | Calibrates ID-only threshold policies | label=0 for ID-threshold calibration | 150 |
| Stage 1 synthetic ID fallback test set | Stage 1 | Primary image split | UCL SynthEye synthetic FAF=150 | Held-out synthetic ID false-rejection estimate | label=0 for final synthetic ID false-rejection reporting | 150 |
| Full OOD evaluation set | Stage 1 eval / Stage 2 source | Manifest view / alias | 1200 rows with parent_image_hash; source_dataset=derived_from_synthetic_faf=1200; subtypes=arrow_annotation=150; blur_artifact=150; border_crop=150; composite_layout=150; gaussian_noise=150; jpeg_compression=150; rectangle_annotation=150; text_watermark=150 | Complete OOD evaluation and Stage 2 source | label=1, ood_type, ood_subtype for evaluation grouping and Stage 2 split construction | 2100 |
| Family-balanced OOD evaluation view | Stage 1 eval | Derived evaluation view | 400 rows with parent_image_hash; source_dataset=derived_from_synthetic_faf=400; subtypes=arrow_annotation=51; blur_artifact=49; border_crop=58; composite_layout=46; gaussian_noise=51; jpeg_compression=50; rectangle_annotation=52; text_watermark=43 | Supporting Stage 1 family-balanced evaluation | label=1, ood_type, ood_subtype for evaluation grouping | 1200 |
| Subtype-balanced OOD evaluation view | Stage 1 eval | Derived evaluation view | 1200 rows with parent_image_hash; source_dataset=derived_from_synthetic_faf=1200; subtypes=arrow_annotation=150; blur_artifact=150; border_crop=150; composite_layout=150; gaussian_noise=150; jpeg_compression=150; rectangle_annotation=150; text_watermark=150 | Primary Stage 1 benchmark | label=1, ood_type, ood_subtype for evaluation grouping | 1650 |
| Stage 2 reason-attribution training set | Stage 2 | Derived manifest view of OOD rows | 720 rows with parent_image_hash; source_dataset=derived_from_synthetic_faf=720; subtypes=arrow_annotation=90; blur_artifact=90; border_crop=90; composite_layout=90; gaussian_noise=90; jpeg_compression=90; rectangle_annotation=90; text_watermark=90 | Fits optional Stage 2 reason-attribution models | ood_type and ood_subtype are supervised Stage 2 targets | 1260 |
| Stage 2 reason-attribution validation set | Stage 2 | Derived manifest view of OOD rows | 240 rows with parent_image_hash; source_dataset=derived_from_synthetic_faf=240; subtypes=arrow_annotation=30; blur_artifact=30; border_crop=30; composite_layout=30; gaussian_noise=30; jpeg_compression=30; rectangle_annotation=30; text_watermark=30 | Validation model selection for Stage 2 | ood_type and ood_subtype are validation targets | 420 |
| Stage 2 reason-attribution test set | Stage 2 | Derived manifest view of OOD rows | 240 rows with parent_image_hash; source_dataset=derived_from_synthetic_faf=240; subtypes=arrow_annotation=30; blur_artifact=30; border_crop=30; composite_layout=30; gaussian_noise=30; jpeg_compression=30; rectangle_annotation=30; text_watermark=30 | Held-out standalone Stage 2 evaluation | ood_type and ood_subtype are final test targets | 420 |

### Full manifest audit
| dataset_split | manifest | rows | label | ood_family | ood_subtype | source_dataset | primary_or_derived | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage 1 nominal training set | datasets/dissertation_v1/manifests/train_id.csv | 700 | 0=700 | id=700 | id=700 | UCL SynthEye synthetic FAF=700 | Primary image split | Fits Stage 1 nominal references only |
| Stage 1 ID validation/calibration set | datasets/dissertation_v1/manifests/val_id.csv | 150 | 0=150 | id=150 | id=150 | UCL SynthEye synthetic FAF=150 | Primary image split | Calibrates ID-only threshold policies |
| Stage 1 synthetic ID fallback test set | datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv | 150 | 0=150 | id=150 | id=150 | UCL SynthEye synthetic FAF=150 | Primary image split | Held-out synthetic ID false-rejection estimate |
| test_artifact | datasets/dissertation_v1/manifests/test_artifact.csv | 1200 | 1=1200 | sensory_artifact=1200 | arrow_annotation=150; blur_artifact=150; border_crop=150; composite_layout=150; gaussian_noise=150; jpeg_compression=150; rectangle_annotation=150; text_watermark=150 | derived_from_synthetic_faf=1200 | Derived images | OOD evaluation grouping |
| test_ood_artifact | datasets/dissertation_v1/manifests/test_ood_artifact.csv | 1200 | 1=1200 | sensory_artifact=1200 | arrow_annotation=150; blur_artifact=150; border_crop=150; composite_layout=150; gaussian_noise=150; jpeg_compression=150; rectangle_annotation=150; text_watermark=150 | derived_from_synthetic_faf=1200 | Manifest view / alias | OOD evaluation grouping |
| test_modality | datasets/dissertation_v1/manifests/test_modality.csv | 400 | 1=400 | modality_shift=400 | colour_fundus=200; oct_screenshot=200 | prepared_colour_fundus=200; prepared_oct_screenshot=200 | Primary image split | OOD evaluation grouping |
| test_ood_modality | datasets/dissertation_v1/manifests/test_ood_modality.csv | 400 | 1=400 | modality_shift=400 | colour_fundus=200; oct_screenshot=200 | prepared_colour_fundus=200; prepared_oct_screenshot=200 | Manifest view / alias | OOD evaluation grouping |
| test_semantic | datasets/dissertation_v1/manifests/test_semantic.csv | 500 | 1=500 | semantic_outlier=500 | cifar10_natural=500 | prepared_cifar10_or_natural=500 | Primary image split | OOD evaluation grouping |
| test_ood_semantic | datasets/dissertation_v1/manifests/test_ood_semantic.csv | 500 | 1=500 | semantic_outlier=500 | cifar10_natural=500 | prepared_cifar10_or_natural=500 | Manifest view / alias | OOD evaluation grouping |
| test_ood | datasets/dissertation_v1/manifests/test_ood.csv | 2100 | 1=2100 | modality_shift=400; semantic_outlier=500; sensory_artifact=1200 | arrow_annotation=150; blur_artifact=150; border_crop=150; cifar10_natural=500; colour_fundus=200; composite_layout=150; gaussian_noise=150; jpeg_compression=150; oct_screenshot=200; rectangle_annotation=150; text_watermark=150 | derived_from_synthetic_faf=1200; prepared_cifar10_or_natural=500; prepared_colour_fundus=200; prepared_oct_screenshot=200 | Manifest view / alias | OOD evaluation grouping |
| Full OOD evaluation set | datasets/dissertation_v1/manifests/test_ood_full.csv | 2100 | 1=2100 | modality_shift=400; semantic_outlier=500; sensory_artifact=1200 | arrow_annotation=150; blur_artifact=150; border_crop=150; cifar10_natural=500; colour_fundus=200; composite_layout=150; gaussian_noise=150; jpeg_compression=150; oct_screenshot=200; rectangle_annotation=150; text_watermark=150 | derived_from_synthetic_faf=1200; prepared_cifar10_or_natural=500; prepared_colour_fundus=200; prepared_oct_screenshot=200 | Manifest view / alias | Complete OOD evaluation and Stage 2 source |
| Family-balanced OOD evaluation view | datasets/dissertation_v1/manifests/test_ood_balanced_by_type.csv | 1200 | 1=1200 | modality_shift=400; semantic_outlier=400; sensory_artifact=400 | arrow_annotation=51; blur_artifact=49; border_crop=58; cifar10_natural=400; colour_fundus=200; composite_layout=46; gaussian_noise=51; jpeg_compression=50; oct_screenshot=200; rectangle_annotation=52; text_watermark=43 | derived_from_synthetic_faf=400; prepared_cifar10_or_natural=400; prepared_colour_fundus=200; prepared_oct_screenshot=200 | Derived evaluation view | Supporting Stage 1 family-balanced evaluation |
| Subtype-balanced OOD evaluation view | datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv | 1650 | 1=1650 | modality_shift=300; semantic_outlier=150; sensory_artifact=1200 | arrow_annotation=150; blur_artifact=150; border_crop=150; cifar10_natural=150; colour_fundus=150; composite_layout=150; gaussian_noise=150; jpeg_compression=150; oct_screenshot=150; rectangle_annotation=150; text_watermark=150 | derived_from_synthetic_faf=1200; prepared_cifar10_or_natural=150; prepared_colour_fundus=150; prepared_oct_screenshot=150 | Derived evaluation view | Primary Stage 1 benchmark |
| test_ood_smoke | datasets/dissertation_v1/manifests/test_ood_smoke.csv | 22 | 1=22 | modality_shift=4; semantic_outlier=2; sensory_artifact=16 | arrow_annotation=2; blur_artifact=2; border_crop=2; cifar10_natural=2; colour_fundus=2; composite_layout=2; gaussian_noise=2; jpeg_compression=2; oct_screenshot=2; rectangle_annotation=2; text_watermark=2 | derived_from_synthetic_faf=16; prepared_cifar10_or_natural=2; prepared_colour_fundus=2; prepared_oct_screenshot=2 | Derived evaluation view | OOD evaluation grouping |
| Stage 2 reason-attribution training set | datasets/dissertation_v1/manifests/reason_train.csv | 1260 | 1=1260 | modality_shift=240; semantic_outlier=300; sensory_artifact=720 | arrow_annotation=90; blur_artifact=90; border_crop=90; cifar10_natural=300; colour_fundus=120; composite_layout=90; gaussian_noise=90; jpeg_compression=90; oct_screenshot=120; rectangle_annotation=90; text_watermark=90 | derived_from_synthetic_faf=720; prepared_cifar10_or_natural=300; prepared_colour_fundus=120; prepared_oct_screenshot=120 | Derived manifest view of OOD rows | Fits optional Stage 2 reason-attribution models |
| Stage 2 reason-attribution validation set | datasets/dissertation_v1/manifests/reason_val.csv | 420 | 1=420 | modality_shift=80; semantic_outlier=100; sensory_artifact=240 | arrow_annotation=30; blur_artifact=30; border_crop=30; cifar10_natural=100; colour_fundus=40; composite_layout=30; gaussian_noise=30; jpeg_compression=30; oct_screenshot=40; rectangle_annotation=30; text_watermark=30 | derived_from_synthetic_faf=240; prepared_cifar10_or_natural=100; prepared_colour_fundus=40; prepared_oct_screenshot=40 | Derived manifest view of OOD rows | Validation model selection for Stage 2 |
| Stage 2 reason-attribution test set | datasets/dissertation_v1/manifests/reason_test.csv | 420 | 1=420 | modality_shift=80; semantic_outlier=100; sensory_artifact=240 | arrow_annotation=30; blur_artifact=30; border_crop=30; cifar10_natural=100; colour_fundus=40; composite_layout=30; gaussian_noise=30; jpeg_compression=30; oct_screenshot=40; rectangle_annotation=30; text_watermark=30 | derived_from_synthetic_faf=240; prepared_cifar10_or_natural=100; prepared_colour_fundus=40; prepared_oct_screenshot=40 | Derived manifest view of OOD rows | Held-out standalone Stage 2 evaluation |


## 3. Complete family/subtype taxonomy

The exact family names are `modality_shift`, `sensory_artifact`, and `semantic_outlier`. The exact subtype names are confirmed by `configs/datasets/dissertation_dataset_v1.yaml:30-44` and the actual `test_ood_full.csv` manifest.
| OOD family | OOD subtype | Full OOD count | Source / generation | Derived from parent image | Representative image | Open OK | Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| modality_shift | colour_fundus | 200 | Prepared public wrong-modality source (prepared_colour_fundus=200) | no | images/dissertation_v1/ood/modality_shift/colour_fundus/colour_fundus_000000.png | True | 2004x1690 |
| modality_shift | oct_screenshot | 200 | Prepared public wrong-modality source (prepared_oct_screenshot=200) | no | images/dissertation_v1/ood/modality_shift/oct_screenshot/oct_screenshot_000000.png | True | 504x496 |
| semantic_outlier | cifar10_natural | 500 | Prepared natural-image semantic source (prepared_cifar10_or_natural=500) | no | images/dissertation_v1/ood/semantic_outlier/cifar10_natural/cifar10_natural_000000.png | True | 224x224 |
| sensory_artifact | arrow_annotation | 150 | Generated by synthetic_transform=arrow_annotation from held-out synthetic FAF parent images | yes | images/dissertation_v1/ood/sensory_artifact/arrow_annotation/artifact_000000_02_arrow_annotation.png | True | 512x512 |
| sensory_artifact | blur_artifact | 150 | Generated by synthetic_transform=blur_artifact from held-out synthetic FAF parent images | yes | images/dissertation_v1/ood/sensory_artifact/blur_artifact/artifact_000000_04_blur_artifact.png | True | 512x512 |
| sensory_artifact | border_crop | 150 | Generated by synthetic_transform=border_crop from held-out synthetic FAF parent images | yes | images/dissertation_v1/ood/sensory_artifact/border_crop/artifact_000000_05_border_crop.png | True | 512x512 |
| sensory_artifact | composite_layout | 150 | Generated by synthetic_transform=composite_layout from held-out synthetic FAF parent images | yes | images/dissertation_v1/ood/sensory_artifact/composite_layout/artifact_000000_03_composite_layout.png | True | 512x512 |
| sensory_artifact | gaussian_noise | 150 | Generated by synthetic_transform=gaussian_noise from held-out synthetic FAF parent images | yes | images/dissertation_v1/ood/sensory_artifact/gaussian_noise/artifact_000000_06_gaussian_noise.png | True | 512x512 |
| sensory_artifact | jpeg_compression | 150 | Generated by synthetic_transform=jpeg_compression from held-out synthetic FAF parent images | yes | images/dissertation_v1/ood/sensory_artifact/jpeg_compression/artifact_000000_07_jpeg_compression.png | True | 512x512 |
| sensory_artifact | rectangle_annotation | 150 | Generated by synthetic_transform=rectangle_annotation from held-out synthetic FAF parent images | yes | images/dissertation_v1/ood/sensory_artifact/rectangle_annotation/artifact_000000_01_rectangle_annotation.png | True | 512x512 |
| sensory_artifact | text_watermark | 150 | Generated by synthetic_transform=text_watermark from held-out synthetic FAF parent images | yes | images/dissertation_v1/ood/sensory_artifact/text_watermark/artifact_000000_00_text_watermark.png | True | 512x512 |


## 4. Recommended Figure 8 examples

The current dissertation example figure has four broad panels (`main.tex:1650-1710`). For a taxonomy figure, use one nominal image plus one image per OOD subtype. All images below opened successfully and had non-zero grayscale standard deviation.
| panel | family | subtype | manifest | image_path | open_ok | size | std_intensity | suitability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_faf_nominal | id | id | datasets/dissertation_v1/manifests/train_id.csv | images/dissertation_v1/id/train/id_train_000033.png | True | 512x512 | 47.7560 | Suitable nominal example |
| colour_fundus | modality_shift | colour_fundus | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/modality_shift/colour_fundus/colour_fundus_000000.png | True | 2004x1690 | 56.4010 | Suitable; opens and is non-blank |
| oct_screenshot | modality_shift | oct_screenshot | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/modality_shift/oct_screenshot/oct_screenshot_000000.png | True | 504x496 | 53.0870 | Suitable; opens and is non-blank |
| cifar10_natural | semantic_outlier | cifar10_natural | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/semantic_outlier/cifar10_natural/cifar10_natural_000000.png | True | 224x224 | 35.5090 | Suitable; opens and is non-blank |
| arrow_annotation | sensory_artifact | arrow_annotation | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/sensory_artifact/arrow_annotation/artifact_000000_02_arrow_annotation.png | True | 512x512 | 34.7220 | Suitable; opens and is non-blank |
| blur_artifact | sensory_artifact | blur_artifact | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/sensory_artifact/blur_artifact/artifact_000000_04_blur_artifact.png | True | 512x512 | 14.0060 | Suitable; opens and is non-blank |
| border_crop | sensory_artifact | border_crop | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/sensory_artifact/border_crop/artifact_000000_05_border_crop.png | True | 512x512 | 18.1240 | Suitable; opens and is non-blank |
| composite_layout | sensory_artifact | composite_layout | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/sensory_artifact/composite_layout/artifact_000000_03_composite_layout.png | True | 512x512 | 24.6700 | Suitable; opens and is non-blank |
| gaussian_noise | sensory_artifact | gaussian_noise | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/sensory_artifact/gaussian_noise/artifact_000000_06_gaussian_noise.png | True | 512x512 | 21.1230 | Suitable; opens and is non-blank |
| jpeg_compression | sensory_artifact | jpeg_compression | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/sensory_artifact/jpeg_compression/artifact_000000_07_jpeg_compression.png | True | 512x512 | 14.0950 | Suitable; opens and is non-blank |
| rectangle_annotation | sensory_artifact | rectangle_annotation | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/sensory_artifact/rectangle_annotation/artifact_000000_01_rectangle_annotation.png | True | 512x512 | 30.9700 | Suitable; opens and is non-blank |
| text_watermark | sensory_artifact | text_watermark | datasets/dissertation_v1/manifests/test_ood_full.csv | images/dissertation_v1/ood/sensory_artifact/text_watermark/artifact_000000_00_text_watermark.png | True | 512x512 | 16.0210 | Suitable; opens and is non-blank |


## 5. Parent-overlap audit

The parent identifier is `parent_image_hash`. It is populated for generated sensory-artifact variants from the source ID row's `source_image_hash` (`src/retinal_ood/data/dissertation_dataset.py:700-702`). Stage 2 split generation stratifies by `ood_subtype` and shuffles rows with `random_state=seed + stable_subtype_offset`, but does not group by parent (`scripts/build_reason_attribution_manifests.py:157-188`).

### Measured overlap
| section | comparison | split | metric | value | notes |
| --- | --- | --- | --- | --- | --- |
| pair_parent_overlap | train-val |  | shared_parent_hashes | 127 | non-empty parent_image_hash only |
| pair_image_path_overlap | train-val |  | shared_image_paths | 0 | image_path disjointness check |
| pair_parent_overlap | train-test |  | shared_parent_hashes | 128 | non-empty parent_image_hash only |
| pair_image_path_overlap | train-test |  | shared_image_paths | 0 | image_path disjointness check |
| pair_parent_overlap | val-test |  | shared_parent_hashes | 108 | non-empty parent_image_hash only |
| pair_image_path_overlap | val-test |  | shared_image_paths | 0 | image_path disjointness check |
| all_three_parent_overlap | train-val-test |  | shared_parent_hashes | 108 | non-empty parent_image_hash shared by all three reason splits |
| rows_with_parent_in_any_other_split | against_other_splits | train | rows | 696 | rows whose parent appears in at least one other reason split |
| rows_with_parent_in_any_other_split | against_other_splits | val | rows | 240 | rows whose parent appears in at least one other reason split |
| rows_with_parent_in_any_other_split | against_other_splits | test | rows | 240 | rows whose parent appears in at least one other reason split |

### By-family and by-subtype overlap rows
| split | family | subtype | metric | value | notes |
| --- | --- | --- | --- | --- | --- |
| train | sensory_artifact | arrow_annotation | rows | 87 | unique_overlapping_parents=87 |
| train | sensory_artifact | blur_artifact | rows | 87 | unique_overlapping_parents=87 |
| train | sensory_artifact | border_crop | rows | 87 | unique_overlapping_parents=87 |
| train | sensory_artifact | composite_layout | rows | 87 | unique_overlapping_parents=87 |
| train | sensory_artifact | gaussian_noise | rows | 87 | unique_overlapping_parents=87 |
| train | sensory_artifact | jpeg_compression | rows | 87 | unique_overlapping_parents=87 |
| train | sensory_artifact | rectangle_annotation | rows | 87 | unique_overlapping_parents=87 |
| train | sensory_artifact | text_watermark | rows | 87 | unique_overlapping_parents=87 |
| val | sensory_artifact | arrow_annotation | rows | 30 | unique_overlapping_parents=30 |
| val | sensory_artifact | blur_artifact | rows | 30 | unique_overlapping_parents=30 |
| val | sensory_artifact | border_crop | rows | 30 | unique_overlapping_parents=30 |
| val | sensory_artifact | composite_layout | rows | 30 | unique_overlapping_parents=30 |
| val | sensory_artifact | gaussian_noise | rows | 30 | unique_overlapping_parents=30 |
| val | sensory_artifact | jpeg_compression | rows | 30 | unique_overlapping_parents=30 |
| val | sensory_artifact | rectangle_annotation | rows | 30 | unique_overlapping_parents=30 |
| val | sensory_artifact | text_watermark | rows | 30 | unique_overlapping_parents=30 |
| test | sensory_artifact | arrow_annotation | rows | 30 | unique_overlapping_parents=30 |
| test | sensory_artifact | blur_artifact | rows | 30 | unique_overlapping_parents=30 |
| test | sensory_artifact | border_crop | rows | 30 | unique_overlapping_parents=30 |
| test | sensory_artifact | composite_layout | rows | 30 | unique_overlapping_parents=30 |
| test | sensory_artifact | gaussian_noise | rows | 30 | unique_overlapping_parents=30 |
| test | sensory_artifact | jpeg_compression | rows | 30 | unique_overlapping_parents=30 |
| test | sensory_artifact | rectangle_annotation | rows | 30 | unique_overlapping_parents=30 |
| test | sensory_artifact | text_watermark | rows | 30 | unique_overlapping_parents=30 |

Generated outputs: `reports/audit/parent_overlap_summary.csv` and `reports/audit/parent_overlap_examples.csv`. The examples CSV contains 1176 rows whose parent appears in another Stage 2 split.

Judgement: Stage 2 attribution metrics may be optimistic for generated sensory artifacts. This is not exact file duplication, because image-path overlap is zero; it is parent-level dependence among generated variants. Stage 1 fitting is not affected because Stage 1 uses ID train rows only.

## 6. Parent-grouped split recommendation

Do not overwrite current results during this audit. Add a grouped Stage 2 split option in a future change, using `parent_image_hash` as the group id when present and `image_path` as the group id for rows without parents. Keep seed 42 and keep test isolated from model selection.

Recommended approach: custom grouped stratification. `GroupShuffleSplit` enforces grouping but not subtype balance; `StratifiedGroupKFold` is less suitable because sensory parent groups are multi-label across eight artifact subtypes and the desired protocol is 60/20/20 rather than k-fold CV.

Expected current-data grouped split counts:
| family | subtype | train | val | test |
| --- | --- | --- | --- | --- |
| modality_shift | colour_fundus | 120 | 40 | 40 |
| modality_shift | oct_screenshot | 120 | 40 | 40 |
| semantic_outlier | cifar10_natural | 300 | 100 | 100 |
| sensory_artifact | arrow_annotation | 90 | 30 | 30 |
| sensory_artifact | blur_artifact | 90 | 30 | 30 |
| sensory_artifact | border_crop | 90 | 30 | 30 |
| sensory_artifact | composite_layout | 90 | 30 | 30 |
| sensory_artifact | gaussian_noise | 90 | 30 | 30 |
| sensory_artifact | jpeg_compression | 90 | 30 | 30 |
| sensory_artifact | rectangle_annotation | 90 | 30 | 30 |
| sensory_artifact | text_watermark | 90 | 30 | 30 |

Rows without parent identifiers should be treated as single-image groups:
| family | subtype | has_parent_image_hash | rows |
| --- | --- | --- | --- |
| modality_shift | colour_fundus | False | 200 |
| modality_shift | oct_screenshot | False | 200 |
| semantic_outlier | cifar10_natural | False | 500 |
| sensory_artifact | arrow_annotation | True | 150 |
| sensory_artifact | blur_artifact | True | 150 |
| sensory_artifact | border_crop | True | 150 |
| sensory_artifact | composite_layout | True | 150 |
| sensory_artifact | gaussian_noise | True | 150 |
| sensory_artifact | jpeg_compression | True | 150 |
| sensory_artifact | rectangle_annotation | True | 150 |
| sensory_artifact | text_watermark | True | 150 |


## 7. k=1 evidence

Global feature kNN uses `model.nearest_neighbors: 1` in `configs/global_feature_knn.yaml:17-23`. PatchCore L2/L3/L4/L2+L3 use `nearest_neighbors: 1` in `configs/patchcore_l2.yaml:16-24`, `configs/patchcore_l3.yaml:16-24`, `configs/patchcore_l4.yaml:16-24`, and `configs/patchcore_l23.yaml:16-24`. Code defaults also set 1 in `src/retinal_ood/baselines/feature_distance.py:21-28` and `src/retinal_ood/models/patchcore.py:24-30`. The scoring functions support other k values, but no committed experiment uses another k; tests only exercise k=2 at function level.

> Dissertation-ready wording: The implemented global kNN and PatchCore detectors used one nearest neighbour (`k=1`) as a fixed configuration. This study did not perform a k-ablation, so k=1 should be interpreted as a pre-specified implementation setting rather than a validated optimum.

## 8. Exact scoring functions
| Method | Nominal representation | Raw scoring function | Image-level aggregation | Higher means |
| --- | --- | --- | --- | --- |
| Image statistics | Train-ID mean/std of 24 low-level descriptors | sqrt(mean(z^2)) after train-ID standardisation | Already image-level descriptor | more anomalous |
| Autoencoder | ID-trained convolutional autoencoder | mean((x - AE(x))^2) over pixels | Mean over flattened pixels | more anomalous |
| Global feature kNN | All pooled ResNet-50 layer-3 ID embeddings | min_i \|\|z(x)-z_i\|\|_2 with k=1 | Global average-pooled embedding | more anomalous |
| Mahalanobis feature distance | Train-ID mean and regularised covariance of pooled layer-3 embeddings | sqrt((z-mu)^T(Sigma+lambda I)^+(z-mu)) | Global average-pooled embedding | more anomalous |
| PatchCore L2 | Seeded memory of ID layer-2 patch embeddings | patch min distance; image max patch distance | Maximum over patch scores | more anomalous |
| PatchCore L3 | Seeded memory of ID layer-3 patch embeddings | patch min distance; image max patch distance | Maximum over patch scores | more anomalous |
| PatchCore L4 | Seeded memory of ID layer-4 patch embeddings | patch min distance; image max patch distance | Maximum over patch scores | more anomalous |
| PatchCore L2+L3 | Seeded memory of aligned layer-2/layer-3 ID patch embeddings | patch min distance; image max patch distance | Maximum over patch scores | more anomalous |

Code evidence: image statistics `src/retinal_ood/baselines/image_statistics.py:34-65,94-146`; autoencoder `src/retinal_ood/models/autoencoder.py:48-52,130-154`; global feature pooling and distances `src/retinal_ood/baselines/feature_distance.py:168-220`; PatchCore patch extraction and max aggregation `src/retinal_ood/models/patchcore.py:96-132,201-240,243-272`; higher-score convention `src/retinal_ood/evaluation/metrics.py:3` and `main.tex:3072-3074`.

Replacement English for Section 5.8:

> All Stage 1 methods output scalar anomaly scores for which larger values indicate stronger evidence of OOD. The image-statistics baseline computes a 24-dimensional descriptor and scores the root-mean-square z-normalised deviation from the ID-training descriptor mean. The autoencoder score is the mean squared pixel reconstruction error. The global feature kNN detector extracts frozen ResNet-50 layer-3 feature maps, average-pools them to an image embedding, and scores the Euclidean distance to the nearest ID-training embedding. The Mahalanobis detector uses the same pooled embedding but scores the regularised covariance-normalised distance from the ID mean. PatchCore stores ID patch embeddings from the selected ResNet layer(s), scores each query patch by nearest-memory Euclidean distance, and aggregates to image level by the maximum patch score; the patch-distance map is used only for localisation-oriented visualisation. Validation data are used only for threshold calibration, and test/OOD labels are not used inside any Stage 1 scoring function.

## 9. Stage 2 SVM selection evidence

The selected family `linear_svm` uses deterministic pooled-pixel global features, not metadata and not Stage 1 anomaly scores (`src/retinal_ood/reason_attribution/methods.py:321-351`; `main.tex:2691-2738`). Its sklearn pipeline fits `StandardScaler` and `LinearSVC` on the training matrix only; `LinearSVC` uses `class_weight="balanced"`, `dual="auto"`, `max_iter=5000`, `random_state=seed`, and default `C=1.0` (`src/retinal_ood/reason_attribution/methods.py:476-489,518-519`).

Selection uses validation macro-F1, then lower complexity and method name for ties; test performance is retrospective (`src/retinal_ood/reason_attribution/comparison.py:205-206,279-302`; `main.tex:3209-3217`). The selected family method is `linear_svm`; the selected subtype method is `hierarchical_classifier`, which routes by predicted family and is non-oracle (`src/retinal_ood/reason_attribution/methods.py:102-123`; `main.tex:2835-2837,3248-3251`).
| item | value |
| --- | --- |
| best_reason_family_method | linear_svm |
| best_reason_family_validation_macro_f1 | 0.9908 |
| best_reason_family_test_accuracy | 0.9881 |
| best_reason_family_test_macro_f1 | 0.9901 |
| best_reason_family_test_balanced_accuracy | 0.9872 |
| family_macro_f1_delta_vs_pr24_baseline | +0.0954 |
| family_improves_over_pr24_baseline | True |
| unknown_threshold_gamma | 0.50 |
| known_coverage_at_gamma | 0.9952 |
| unknown_rate_at_gamma | 0.0048 |
| accuracy_excluding_unknown | 0.9928 |
| best_subtype_method | hierarchical_classifier |
| best_subtype_validation_macro_f1 | 0.8671 |
| best_subtype_test_macro_f1 | 0.9059 |
| subtype_macro_f1_delta_vs_pr24_baseline | 0.2335 |
| hardest_family | semantic_outlier (F1=0.9848) |
| hardest_subtype | rectangle_annotation (F1=0.7241) |
| selected_final_method | linear_svm |
| selection_rationale | Selected by validation reason-family macro-F1, with simpler methods preferred on ties. |
| conceptual_boundary | Stage 1 remains ID-only unsupervised OOD detection; Stage 2 is post-hoc explanation. |
| clinical_scope | Reason labels are likely rejection explanations, not disease predictions. |

Family metrics by method:
| method | split | feature_set | estimator | family_accuracy | family_macro_f1 | family_balanced_accuracy | known_coverage_at_gamma | unknown_rate_at_gamma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_statistics_logreg | val | statistics | logistic_regression | 0.9643 | 0.9619 | 0.9736 | 0.9976 | 0.0024 |
| image_statistics_logreg | test | statistics | logistic_regression | 0.9476 | 0.9433 | 0.9647 | 1.0000 | 0.0000 |
| global_feature_knn | val | global | knn | 0.9714 | 0.9678 | 0.9600 | 1.0000 | 0.0000 |
| global_feature_knn | test | global | knn | 0.9762 | 0.9740 | 0.9686 | 1.0000 | 0.0000 |
| nearest_centroid | val | global | nearest_centroid | 0.8190 | 0.7948 | 0.8186 | 0.9976 | 0.0024 |
| nearest_centroid | test | global | nearest_centroid | 0.7810 | 0.7665 | 0.8033 | 0.9976 | 0.0024 |
| logistic_regression | val | global | logistic_regression | 0.9905 | 0.9877 | 0.9867 | 1.0000 | 0.0000 |
| logistic_regression | test | global | logistic_regression | 0.9929 | 0.9901 | 0.9900 | 1.0000 | 0.0000 |
| linear_svm | val | global | linear_svm | 0.9905 | 0.9908 | 0.9886 | 0.9976 | 0.0024 |
| linear_svm | test | global | linear_svm | 0.9881 | 0.9901 | 0.9872 | 0.9952 | 0.0048 |
| random_forest_or_gradient_boosting | val | global | random_forest | 0.9905 | 0.9897 | 0.9850 | 1.0000 | 0.0000 |
| random_forest_or_gradient_boosting | test | global | random_forest | 0.9905 | 0.9911 | 0.9867 | 0.9976 | 0.0024 |
| feature_statistics_fusion | val | fusion | logistic_regression | 0.9905 | 0.9877 | 0.9867 | 1.0000 | 0.0000 |
| feature_statistics_fusion | test | fusion | logistic_regression | 0.9952 | 0.9939 | 0.9933 | 1.0000 | 0.0000 |
| hierarchical_classifier | val | fusion | hierarchical_logistic | 0.9905 | 0.9877 | 0.9867 | 1.0000 | 0.0000 |
| hierarchical_classifier | test | fusion | hierarchical_logistic | 0.9952 | 0.9939 | 0.9933 | 1.0000 | 0.0000 |

Subtype metrics by method:
| method | split | feature_set | estimator | subtype_accuracy | subtype_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| image_statistics_logreg | val | statistics | logistic_regression | 0.8905 | 0.8603 |
| image_statistics_logreg | test | statistics | logistic_regression | 0.9048 | 0.8802 |
| global_feature_knn | val | global | knn | 0.5810 | 0.4797 |
| global_feature_knn | test | global | knn | 0.5690 | 0.4683 |
| nearest_centroid | val | global | nearest_centroid | 0.5476 | 0.4253 |
| nearest_centroid | test | global | nearest_centroid | 0.5405 | 0.4137 |
| logistic_regression | val | global | logistic_regression | 0.6833 | 0.6043 |
| logistic_regression | test | global | logistic_regression | 0.6952 | 0.6303 |
| linear_svm | val | global | linear_svm | 0.6405 | 0.5499 |
| linear_svm | test | global | linear_svm | 0.6524 | 0.5894 |
| random_forest_or_gradient_boosting | val | global | random_forest | 0.7238 | 0.6524 |
| random_forest_or_gradient_boosting | test | global | random_forest | 0.7405 | 0.6786 |
| feature_statistics_fusion | val | fusion | logistic_regression | 0.8214 | 0.7748 |
| feature_statistics_fusion | test | fusion | logistic_regression | 0.8476 | 0.8134 |
| hierarchical_classifier | val | fusion | hierarchical_logistic | 0.8929 | 0.8671 |
| hierarchical_classifier | test | fusion | hierarchical_logistic | 0.9238 | 0.9059 |


The high family scores are plausible because the three reason families are visually separable in this controlled taxonomy; the selected family confusion matrix has only three class-to-class errors, all from semantic outliers. The hierarchy reaches stronger subtype performance because broad family routing decomposes the 11-way problem into within-family decisions. The limitation is that generated artifact parent overlap can inflate artifact-subtype performance.

## 10. Inconsistencies between code, manifests and dissertation

No direct numeric contradiction was found for the headline counts and metrics checked. Precision issues to tighten:

1. `main.tex:1482-1574` Table 3 is accurate but too compressed; replace or supplement it with the revised Table 3 above.
2. `main.tex:1650-1710` Figure 8 currently has four broad examples; it does not display all 11 subtypes.
3. `main.tex:2384-2401` and `main.tex:2571-2573` should explicitly say no k-ablation was performed.
4. `main.tex:2736-2738` should enumerate excluded Stage 2 metadata fields from `docs/experiments/reason_attribution_method_comparison.md:44-50`.
5. `main.tex:1722-1733`, `main.tex:1758-1762`, and `main.tex:4167-4172` accurately state parent overlap; add that Stage 1 is unaffected while Stage 2 artifact metrics may be optimistic.
6. Keep the nuance in `main.tex:2848-2853` and `main.tex:3915-3918`: Stage 2 standalone evaluation is on all labelled OOD test rows, not only the subset rejected by a specific Stage 1 threshold.

## 11. Exact dissertation sections and sentences that should be revised

- Section 4.2 / Table 3 (`main.tex:1482-1574`): expand columns to include primary/derived status, parent/source, label usage, and role.
- Section 4.5 / Figure 8 (`main.tex:1588-1711`): replace the four-panel example with a 12-panel taxonomy grid or move the full grid to the appendix.
- Section 4.6/4.7 (`main.tex:1722-1733`, `main.tex:1758-1762`): add root cause: subtype-stratified row splitting without parent grouping in `scripts/build_reason_attribution_manifests.py:157-188`.
- Section 5.6.1 (`main.tex:2384-2401`, `main.tex:2571-2573`): add the k=1 wording above.
- Section 5.8 (`main.tex:1905-1971`, `main.tex:2681-2685`): replace vague scoring language with the formal paragraph above.
- Section 5 Stage 2 features (`main.tex:2691-2738`): add that Stage 1 anomaly scores are not used as Stage 2 features and enumerate excluded metadata fields.
- Section 6 Stage 2 results (`main.tex:3780-3918`): keep `linear_svm` as validation-selected family method and `hierarchical_classifier` as validation-selected subtype method; do not imply the family method is the highest retrospective test row.
- Discussion (`main.tex:4093-4123`, `main.tex:4154-4210`): retain proof-of-concept, parent-overlap, non-clinical, and non-disease-classifier limitations.

## 12. List of outputs generated

- `docs/dissertation/vincent_feedback_code_audit.md`
- `reports/audit/parent_overlap_summary.csv`
- `reports/audit/parent_overlap_examples.csv`

## 13. Validation commands

Commands were run after generating the audit files. No formal manifests, model outputs, code files, or `main.tex` were modified.

| Command | Working directory | Result |
| --- | --- | --- |
| `python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/val_id.csv datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv datasets/dissertation_v1/manifests/test_ood_full.csv` | code repository | Passed. Totals: 3,100 rows, 0 missing files, 0 duplicate image paths, 0 non-empty patient_id. Counts: train ID 700, validation ID 150, synthetic fallback ID 150, full OOD 2,100. |
| `python scripts/audit_dataset_images.py --root-dir data --manifest datasets/dissertation_v1/manifests/train_id.csv --manifest datasets/dissertation_v1/manifests/val_id.csv --manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv --manifest datasets/dissertation_v1/manifests/test_ood_full.csv --fail-on-corrupt --fail-on-duplicate-content-across-splits` | code repository | Passed. Images 3,100; opened OK 3,100; missing 0; corrupt 0; private paths 0; non-empty patient_id 0; blank/near-blank 0; all-black 0; all-white 0; tiny 0; duplicate SHA256 groups 0; duplicate content across splits 0. |
| `pytest -q tests/test_dataset_schema.py tests/test_dissertation_dataset_builder.py tests/test_reason_attribution_manifests.py tests/test_reason_attribution_method_comparison.py tests/test_lightweight_baselines.py tests/test_patchcore.py tests/test_autoencoder.py` | code repository | Passed: 53 passed, 11 warnings in 32.63 s. Warnings were third-party PyParsing deprecation warnings from Matplotlib. |
| `git diff --check` | dissertation project | Passed after final report update; no whitespace errors. |
| `git status --short` | dissertation project | Shows only the allowed new audit paths: `docs/` and `reports/audit/`. |

Could not verify: external clinical FAF validity, license clearance of prepared public OOD sources, patient/source-grouped generalisation, and parent-independent Stage 2 attribution performance. These require new data or regenerated grouped splits and are intentionally outside this audit.

Repository files modified by this audit: only `docs/dissertation/vincent_feedback_code_audit.md`, `reports/audit/parent_overlap_summary.csv`, and `reports/audit/parent_overlap_examples.csv`.

## Appendix A. Stage 1 headline metrics from committed reports
| scheme | scheme_label | auroc | auprc | fpr_at_95_tpr | threshold | threshold_source | id_false_rejection_rate | ood_recall_at_threshold | id_count | ood_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mahalanobis_feature | Mahalanobis feature | 0.9724 | 0.9975 | 0.2000 | 38.5824 | validation_id_quantile | 0.0467 | 0.9079 | 150 | 1650 |
| global_feature_knn | Global feature kNN | 0.9458 | 0.9949 | 0.3800 | 3.9069 | validation_id_quantile | 0.0333 | 0.8024 | 150 | 1650 |
| patchcore_l3 | PatchCore L3 | 0.8819 | 0.9880 | 0.6333 | 34.7633 | validation_id_quantile | 0.0667 | 0.7097 | 150 | 1650 |
| patchcore_l2_l3 | PatchCore L2+L3 | 0.8722 | 0.9874 | 0.6867 | 44.2789 | validation_id_quantile | 0.0733 | 0.6933 | 150 | 1650 |
| patchcore_l2 | PatchCore L2 | 0.8280 | 0.9828 | 0.7867 | 32.9785 | validation_id_quantile | 0.0600 | 0.6200 | 150 | 1650 |
| image_statistics | Image statistics | 0.7681 | 0.9717 | 0.7467 | 1.7220 | validation_id_quantile | 0.0267 | 0.2994 | 150 | 1650 |
| autoencoder | Autoencoder | 0.7649 | 0.9763 | 0.9467 | 0.0017 | validation_id_quantile | 0.0267 | 0.5588 | 150 | 1650 |
| patchcore_l4 | PatchCore L4 | 0.7502 | 0.9692 | 0.8467 | 60.8534 | validation_id_quantile | 0.0333 | 0.3527 | 150 | 1650 |
