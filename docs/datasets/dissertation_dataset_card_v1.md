# Dataset Card: Dissertation FAF OOD v1

## Purpose

This dataset card describes a metadata-only package for dissertation experiments
on an unsupervised OOD gatekeeper for retinal FAF quality control. The package
is designed to evaluate whether an upstream model should accept a likely-valid
FAF input or reject an invalid/OOD input.

This is not a disease-classification dataset.

## Intended Composition

ID data:

- valid synthetic FAF images from SynthEye
- train/validation/test split produced deterministically
- `test_id_synthetic_fallback.csv` used only as synthetic fallback validation

OOD data:

- wrong modality: colour fundus, OCT screenshot/B-scan style images
- sensory artifact: generated text, annotation, composite, blur, crop, noise,
  and compression artifacts from held-out ID images only
- semantic outlier: natural/non-retinal images such as CIFAR/Open Images subsets

## Manifest Fields

Core fields:

- `image_path`
- `label`
- `split`
- `source`
- `ood_type`
- `ood_subtype`

Provenance and safety fields:

- `source_dataset`
- `source_split`
- `source_image_hash`
- `parent_image_hash`
- `synthetic_transform`
- `semantic_class`
- `is_synthetic_surrogate`
- `patient_id`
- `scanner`
- `license_status`
- `notes`

`patient_id` must be empty. Clinical disease labels, eye IDs, biomarker labels,
private paths, and raw source filenames must not enter the committed package.

## Known Limitations

- The ID test set is synthetic fallback unless real clinical FAF is added later.
- Public OOD sources must be license-reviewed before final dissertation release.
- Synthetic surrogate rows are for pipeline testing only and should not be used
  to claim real-world OOD performance.

## Safe Storage

Images remain under ignored local data directories. The repository stores only
metadata, safe manifests, and documentation.
