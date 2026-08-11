# Dataset Source Provenance Audit

This document records the public-source provenance audit for the dissertation
benchmark. It updates the manuscript evidence without changing benchmark
manifests, model code, Stage 1 results, Stage 2 results, or reported metrics.

## Benchmark Composition

| Component | Manifest source dataset | Family | Subtype | Images | Provenance status |
|---|---|---|---|---:|---|
| Synthetic FAF-like ID | `UCL SynthEye synthetic FAF` | `id` | `id` | 1,000 | Public dataset confirmed at dataset level |
| Sensory artefact OOD | `derived_from_synthetic_faf` | `sensory_artifact` | 8 artefact subtypes | 1,200 | Generated from held-out synthetic FAF-like parents |
| Colour fundus OOD | `prepared_colour_fundus` | `modality_shift` | `colour_fundus` | 200 | Public prepared source bucket confirmed; exact upstream dataset and per-image mapping unresolved |
| OCT screenshot OOD | `prepared_oct_screenshot` | `modality_shift` | `oct_screenshot` | 200 | Public prepared source bucket confirmed; exact upstream dataset and per-image mapping unresolved |
| Natural-image OOD | `prepared_cifar10_or_natural` | `semantic_outlier` | `cifar10_natural` | 500 | Public prepared source bucket confirmed; exact upstream dataset and per-image mapping unresolved |

The 900 imported OOD rows are distinct from the 1,200 sensory-artefact rows.
The sensory rows are transformations generated from held-out synthetic
FAF-like benchmark parents and do not introduce additional external image
sources.

## Confirmed Public Sources

The nominal ID source is the UCL SynthEye synthetic FAF dataset:

- UCL RDR record: <https://rdr.ucl.ac.uk/articles/dataset/Synthetic_dataset_of_100_fundus_auto_fluorescence_of_inherited_retinal_disease/28604234>
- DOI: <https://doi.org/10.5522/04/28604234>

The public OOD imports are confirmed in the committed benchmark as prepared
public source buckets:

- `prepared_colour_fundus`
- `prepared_oct_screenshot`
- `prepared_cifar10_or_natural`

The committed benchmark does not preserve enough information to assign each
imported OOD image to a specific upstream public dataset such as APTOS, RFMiD,
IRFundusSet, OLIVES, CIFAR-10, or Open Images. These datasets remain documented
as preparation candidates in the project runbook, not as per-image confirmed
sources in the current committed package.

Retinograd-AI is a related-work and FAF-gradability reference only. It was not
used as benchmark training, validation, calibration, or evaluation data.

## Evidence Used

The audit used the following repository evidence:

- `datasets/dissertation_v1/manifests/train_id.csv`
- `datasets/dissertation_v1/manifests/val_id.csv`
- `datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv`
- `datasets/dissertation_v1/manifests/test_ood_full.csv`
- `datasets/dissertation_v1/manifests/test_modality.csv`
- `datasets/dissertation_v1/manifests/test_semantic.csv`
- `configs/datasets/dissertation_dataset_v1.yaml`
- `src/retinal_ood/data/dissertation_dataset.py`
- `docs/datasets/dissertation_dataset_v1.md`
- `docs/datasets/dissertation_dataset_card_v1.md`
- `docs/EXPERIMENT_RUNBOOK_CN.md`
- `docs/LOCAL_RUN_AND_DATASET_GUIDE_CN.md`
- Git history for dataset preparation commits `1d1bf8b` and `0ee222b`

The dataset builder copies prepared OOD images from local directories into the
packaged benchmark and writes canonical names such as
`colour_fundus_000000.png`, `oct_screenshot_000000.png`, and
`cifar10_natural_000000.png`. It records the content hash of the source image
but does not write `original_filename`, `original_relative_path`, or
`source_url` fields to the manifest. Consequently, the current committed
package supports content-hash traceability within the benchmark, but not
per-image upstream-dataset traceability for the imported OOD rows.

## Per-Image Traceability

The generated per-image audit file is:

- `reports/audit/imported_ood_image_provenance.csv`

It covers the 900 imported OOD rows:

| Family | Subtype | Images | Traceability status |
|---|---|---:|---|
| `modality_shift` | `colour_fundus` | 200 | `public_source_confirmed_but_per_image_mapping_unavailable` |
| `modality_shift` | `oct_screenshot` | 200 | `public_source_confirmed_but_per_image_mapping_unavailable` |
| `semantic_outlier` | `cifar10_natural` | 500 | `public_source_confirmed_but_per_image_mapping_unavailable` |

No row is marked `confirmed_per_image`, because no retained file, manifest
field, or source directory in the current committed package provides the
original upstream filename or URL.

## Unresolved Provenance Limitations

1. The exact upstream public dataset for each imported OOD image cannot be
   reconstructed from the current committed package.
2. Licence review remains source-bucket level for the imported OOD images.
3. The subtype name `cifar10_natural` is not treated as sufficient evidence that
   every natural image came from CIFAR-10.
4. The benchmark confirms public/prepared OOD source buckets, but a future
   release should preserve immutable per-image source, filename, licence, and
   download-record metadata.

These limitations affect source documentation and redistribution evidence. They
do not change the manifest rows, model fitting protocol, Stage 1 ID-only
boundary, Stage 2 grouped evaluation, or reported metrics.
