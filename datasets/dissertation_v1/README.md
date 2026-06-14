# Dissertation Dataset v1

This package defines the dissertation dataset for the retinal FAF OOD gatekeeper.
It is not a disease-classification dataset.

This dataset package contains manifests and metadata only; image files are not committed.
Individual PNG/JPG/TIFF files are not committed as normal Git blobs; the curated image package is stored as a Git LFS archive.

Use `scripts/prepare_dissertation_dataset.py` to rebuild local private images and
manifests from your prepared data folders. The committed CSV files contain only
relative paths, labels, taxonomy fields, hashes, and safe provenance notes.

The curated image archive is stored through Git LFS at
`lfs/dissertation_v1_images.zip`.

## Manifests

- `manifests/train_id.csv`: ID-only synthetic FAF training rows.
- `manifests/val_id.csv`: ID-only synthetic FAF validation rows.
- `manifests/test_id_synthetic_fallback.csv`: synthetic fallback ID test rows,
  not real clinical FAF validation.
- `manifests/test_artifact.csv`: generated sensory artifact OOD rows.
- `manifests/test_modality.csv`: wrong-modality OOD rows.
- `manifests/test_semantic.csv`: semantic outlier OOD rows.
- `manifests/test_ood*.csv`: dissertation-named artifact/modality/semantic,
  full, balanced, and smoke OOD evaluation subsets.

## School Server

```bash
git lfs pull
python scripts/unpack_dissertation_dataset.py --dataset-dir datasets/dissertation_v1 --root-dir data --verify-checksums
```
