# Final Reproducibility Checklist

One-command final verification after environment setup and `git lfs pull`:

```bash
python scripts/build_dissertation_delivery_bundle.py --strict --out-dir reports/dissertation_final && python scripts/final_repository_audit.py && python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/test_ood_full.csv
```

- [ ] `git lfs install && git lfs pull`
- [ ] `pip install -e ".[dev]"`
- [ ] `python scripts/validate_manifests.py --root-dir data datasets/dissertation_v1/manifests/train_id.csv datasets/dissertation_v1/manifests/test_ood_full.csv`
- [ ] `python scripts/audit_dataset_images.py --root-dir data --manifest datasets/dissertation_v1/manifests/train_id.csv --manifest datasets/dissertation_v1/manifests/test_ood_full.csv --fail-on-corrupt`
- [ ] `python scripts/run_experiment_grid.py --grid-config configs/multi_scheme_grid.yaml --root-dir data --out-dir reports/generated/dissertation_delivery_smoke --only mahalanobis_feature --train-manifest datasets/dissertation_v1/manifests/train_id.csv --val-manifest datasets/dissertation_v1/manifests/val_id.csv --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_smoke.csv --max-train-images 8 --max-test-images 8 --device cpu --seed 42`
- [ ] `python scripts/build_dissertation_delivery_bundle.py --strict --out-dir reports/dissertation_final`
- [ ] `python scripts/final_repository_audit.py`
