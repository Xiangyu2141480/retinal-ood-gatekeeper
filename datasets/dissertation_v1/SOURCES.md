# Sources

## ID FAF

- UCL SynthEye synthetic FAF dataset, used as proof-of-concept ID FAF.
- SynthEye class folder names are used only for deterministic split/audit notes,
  not as disease labels.

## OOD

- Sensory artifacts are generated from held-out ID rows only.
- Colour fundus, OCT screenshot, and natural-image sources must be prepared
  locally and license-reviewed before dissertation publication.
- Rows marked `is_synthetic_surrogate=true` are pipeline placeholders only and
  must not be reported as real public OOD evidence.

## School Server Reproducibility

```bash
git lfs pull
python scripts/unpack_dissertation_dataset.py --dataset-dir datasets/dissertation_v1 --root-dir data --verify-checksums
python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/val_id.csv datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv datasets/dissertation_v1/manifests/test_ood.csv
```
