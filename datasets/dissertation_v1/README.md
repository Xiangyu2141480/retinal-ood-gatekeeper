# Dissertation Dataset v1

This package defines the dissertation dataset for the retinal FAF OOD gatekeeper.
It is not a disease-classification dataset.

This repository includes the actual final dissertation dataset images under `data/images/dissertation_v1/` through Git LFS.
Individual PNG/JPG/TIFF files are tracked by Git LFS and must not be stored as normal Git blobs.

Use `scripts/prepare_dissertation_dataset.py` to rebuild local private images and
manifests from your prepared data folders. The committed CSV files contain only
relative paths, labels, taxonomy fields, hashes, and safe provenance notes.

The canonical image files are under `data/images/dissertation_v1/`.

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
python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/test_ood_full.csv
python scripts/audit_dataset_images.py --root-dir data --manifest datasets/dissertation_v1/manifests/train_id.csv --manifest datasets/dissertation_v1/manifests/val_id.csv --manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv --manifest datasets/dissertation_v1/manifests/test_ood_full.csv --fail-on-corrupt --fail-on-duplicate-content-across-splits
python scripts/unpack_dissertation_dataset.py --dataset-dir datasets/dissertation_v1 --root-dir data --verify-checksums
```
