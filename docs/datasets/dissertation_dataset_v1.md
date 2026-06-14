# Dissertation Dataset v1

This page defines the dataset package used for dissertation experiments in this
repository. The task is an unsupervised retinal FAF OOD gatekeeper, not disease
classification.

## Contract

- Contract YAML: `configs/datasets/dissertation_dataset_v1.yaml`
- Package directory: `datasets/dissertation_v1/`
- Curated images: `data/images/dissertation_v1/` (tracked through Git LFS)
- Local generated manifests: `data/manifests/generated/dissertation_v1/` (ignored)
- Local audit output: `reports/local_audits/dissertation_v1/` (ignored)
- Checksums: `datasets/dissertation_v1/checksums.sha256`

The repository includes the final curated dissertation images through Git LFS.
It must not contain raw download/cache folders, model weights, private paths,
patient identifiers, Eye_ID values, clinical labels, biomarker labels, API keys,
or disease labels as prediction targets.

## Taxonomy

Training is ID-only:

- `label=0`
- `ood_type=id`
- `ood_subtype=id`
- `source=synthetic_faf`

OOD rows are evaluation-only:

- `label=1`
- `ood_type=modality_shift`, `sensory_artifact`, or `semantic_outlier`
- every official OOD row must include non-empty `ood_subtype`

The expected v1 OOD subtypes are:

- `modality_shift`: `colour_fundus`, `oct_screenshot`
- `semantic_outlier`: `cifar10_natural`
- `sensory_artifact`: `text_watermark`, `rectangle_annotation`,
  `arrow_annotation`, `composite_layout`, `blur_artifact`, `border_crop`,
  `gaussian_noise`, `jpeg_compression`

## Synthetic Fallback Warning

`test_id_synthetic_fallback.csv` is synthetic fallback ID data only. It is not
real clinical FAF validation. Dissertation claims should describe this as a
proof-of-concept limitation unless a license-approved real clinical FAF test
set is added later.

## Build Command

```bash
python scripts/prepare_dissertation_dataset.py \
  --config configs/datasets/dissertation_dataset_v1.yaml \
  --root-dir data \
  --prepared-syntheye-dir data/images/synthetic_faf \
  --prepared-colour-fundus-dir data/images/ood_modality/colour_fundus \
  --prepared-oct-dir data/images/ood_modality/oct_screenshot \
  --prepared-cifar-dir data/images/ood_semantic/natural \
  --out-image-dir data/images/dissertation_v1 \
  --repo-dataset-dir datasets/dissertation_v1 \
  --seed 42 \
  --commit-individual-lfs-images
```

If public OOD folders are unavailable and you need to test the pipeline shape,
add `--allow-synthetic-surrogates`. Those rows are clearly marked and must not
be reported as real public OOD evidence.

## School Server Workflow

After cloning the private repository on the school server:

```bash
git lfs install
git lfs pull
pip install -e ".[dev]"
python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/test_ood_full.csv
python scripts/audit_dataset_images.py --root-dir data --manifest datasets/dissertation_v1/manifests/train_id.csv --manifest datasets/dissertation_v1/manifests/val_id.csv --manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv --manifest datasets/dissertation_v1/manifests/test_ood_full.csv --fail-on-corrupt --fail-on-duplicate-content-across-splits
```

No manual copy, external download, or unpack step is required for the primary
workflow. `scripts/unpack_dissertation_dataset.py --verify-checksums` remains as
a compatibility verifier and checks the direct Git LFS image files when no
archive is present.

## Validation

Run these after rebuilding the package:

```bash
python scripts/validate_manifests.py \
  --root-dir data \
  --fail-on-duplicates \
  datasets/dissertation_v1/manifests/train_id.csv \
  datasets/dissertation_v1/manifests/val_id.csv \
  datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  datasets/dissertation_v1/manifests/test_ood_artifact.csv \
  datasets/dissertation_v1/manifests/test_ood_modality.csv \
  datasets/dissertation_v1/manifests/test_ood_semantic.csv \
  datasets/dissertation_v1/manifests/test_ood_full.csv \
  datasets/dissertation_v1/manifests/test_ood_balanced_by_type.csv \
  datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv \
  datasets/dissertation_v1/manifests/test_ood_smoke.csv
```

```bash
python scripts/audit_dataset_images.py \
  --root-dir data \
  --manifest datasets/dissertation_v1/manifests/train_id.csv \
  --manifest datasets/dissertation_v1/manifests/val_id.csv \
  --manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --manifest datasets/dissertation_v1/manifests/test_ood_full.csv \
  --fail-on-corrupt \
  --fail-on-duplicate-content-across-splits
```

Then run:

```bash
pytest
ruff check .
git diff --check
```

Before opening a PR, also verify:

```bash
git lfs ls-files
git lfs ls-files | findstr data/images/dissertation_v1
git diff --cached --check
python scripts/unpack_dissertation_dataset.py --dataset-dir datasets/dissertation_v1 --root-dir data --verify-checksums
```

## Dissertation Use

Recommended table naming:

- ID train/val/test synthetic fallback counts
- full OOD counts by `ood_type`
- OOD counts by `ood_subtype`
- balanced-by-type evaluation subset
- balanced-by-subtype evaluation subset
- smoke subset for quick CLI/UI verification

Do not use SynthEye source class folders as disease labels. They are retained
only as split/audit notes.
