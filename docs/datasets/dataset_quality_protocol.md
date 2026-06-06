# Dataset Quality Protocol

This project is an unsupervised retinal FAF OOD gatekeeper, not a disease classifier. Dataset QA is
therefore focused on input validity, modality/category separation, privacy, and reproducible
evaluation subsets. Images, local manifests, contact sheets, JSON audits, model weights, runs, and
heatmaps must remain ignored and must not be committed.

## Local Acceptance Gates

Before running dissertation experiments, run manifest validation and image-level audit locally. The
local audit should satisfy these gates:

```text
opened_ok: 3100
corrupt: 0
missing: 0
blank_or_near_blank: ideally 0, or manually justified
duplicate_content_across_splits: 0
private_paths: 0
non_empty_patient_id: 0
```

Hard stop conditions:

- Any image path is absolute, private, or resolves outside `--root-dir`.
- Any image is missing or corrupt.
- Any `patient_id` value is non-empty.
- Any train row has `label=1` or `ood_type != id`.
- Any sensory-artifact row was generated from `source_split=train`.
- Duplicate SHA256 image content appears across train/val/test splits.
- `data/images/`, contact sheets, local JSON audits, generated reports, or model weights are staged.

Visual QA is manual and local only. Contact sheets are generated under
`reports/local_audits/contact_sheets/`, which is ignored by Git. Use them to quickly inspect whether
groups look visually coherent, but do not treat contact sheets as quantitative evidence.

## Current Local Manifest Counts

The current local proof-of-concept dataset should have these manifest-level counts:

| Manifest | Expected rows | Label | OOD type | Notes |
|---|---:|---:|---|---|
| `train_synthetic_faf.csv` | 700 | 0 | `id` | ID-only training |
| `val_synthetic_faf.csv` | 150 | 0 | `id` | validation ID threshold calibration |
| `test_real_id.csv` | 150 | 0 | `id` | synthetic fallback, not real clinical FAF |
| `test_id_synthetic_fallback.csv` | 150 | 0 | `id` | generated local alias with explicit warning |
| `test_artifact.csv` | 1200 | 1 | `sensory_artifact` | synthetic artifact stress test |
| `test_semantic.csv` | 500 | 1 | `semantic_outlier` | semantic stress test |
| `test_modality.csv` | 400 | 1 | `modality_shift` | wrong-modality retinal inputs |
| `test_ood.csv` | 2100 | 1 | mixed OOD | merged OOD evaluation manifest |

`test_real_id.csv` is currently a synthetic ID fallback. Reports and PR summaries must preserve the
warning that real clinical FAF validation has not been completed unless an explicitly approved,
anonymized real FAF manifest is supplied.

## OOD Subtype Counts

OOD rows must use the literature-review taxonomy:

```text
id
modality_shift
sensory_artifact
semantic_outlier
```

Subtype counts should remain visible because the dissertation needs per-category analysis:

| OOD type | OOD subtype | Expected count |
|---|---|---:|
| `modality_shift` | `colour_fundus` | 200 |
| `modality_shift` | `oct_screenshot` | 200 |
| `semantic_outlier` | `cifar10_natural` | 500 |
| `sensory_artifact` | `text_watermark` | shown separately |
| `sensory_artifact` | `rectangle_annotation` | shown separately |
| `sensory_artifact` | `arrow_annotation` | shown separately |
| `sensory_artifact` | `composite_layout` | shown separately |
| `sensory_artifact` | `blur_artifact` | shown separately |
| `sensory_artifact` | `border_crop` | shown separately |
| `sensory_artifact` | `gaussian_noise` | shown separately |
| `sensory_artifact` | `jpeg_compression` | shown separately |

Colour fundus and OCT screenshots must be separated by `ood_subtype` because they are visually and
technically different modality-shift cases. CIFAR-10/natural images are only semantic stress tests;
they do not represent clinical retinal imaging failures. Artifact OOD rows are synthetic
sensory-artifact stress tests and should be generated only from held-out validation/test ID images.

## Evaluation Manifest Subsets

Use `scripts/build_evaluation_manifests.py` to generate local-only evaluation subsets under
`data/manifests/generated/`:

- `test_ood_full.csv`: all OOD rows.
- `test_ood_balanced_by_type.csv`: equal count per `ood_type` where possible.
- `test_ood_balanced_by_subtype.csv`: equal count per `ood_subtype` where possible.
- `test_ood_smoke.csv`: tiny stratified OOD sample for CPU smoke tests.
- `test_id_synthetic_fallback.csv`: optional copy of `test_real_id.csv` with an explicit synthetic
  fallback warning.

These generated manifests are ignored by Git. They preserve source metadata such as `source`,
`ood_type`, `ood_subtype`, `source_split`, and `source_image_hash` when present. OOD rows must never
be sampled into `split=train`.

## Provenance Notes

OLIVES data and other public medical datasets may contain patient identifiers, acquisition labels,
or disease annotations. Those fields must not enter this project’s manifests. This project is not a
disease classifier.

Peacein/color-fundus-eye and any similar public colour-fundus source require manual license and
provenance verification before final dissertation claims. Record the dataset URL, access date,
license, and whether redistribution is allowed in a private research log or thesis appendix, not in
Git if it includes private access details.
