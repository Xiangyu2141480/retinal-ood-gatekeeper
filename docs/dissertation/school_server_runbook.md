# School Server / Jackpot Runbook

This guide is for reproducing the final dissertation package on a school server or Jackpot-like Linux environment. Keep all generated outputs outside Git history.

## Clone and Pull LFS Data

```bash
git clone https://github.com/Xiangyu2141480/retinal-ood-gatekeeper.git
cd retinal-ood-gatekeeper
git lfs install
git lfs pull
```

Check that images are present:

```bash
find data/images/dissertation_v1 -type f | head
```

## Environment Setup

CPU-only environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

CUDA/GPU environment, if available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

## CPU-Only Smoke Test

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/server_smoke \
  --only mahalanobis_feature \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_smoke.csv \
  --max-train-images 8 \
  --max-test-images 8 \
  --device cpu \
  --seed 42
```

## CUDA/GPU Run If Available

Use the same command with `--device cuda` for a selected run:

```bash
python scripts/run_experiment_grid.py \
  --grid-config configs/multi_scheme_grid.yaml \
  --root-dir data \
  --out-dir reports/generated/server_selected \
  --only mahalanobis_feature \
  --train-manifest datasets/dissertation_v1/manifests/train_id.csv \
  --val-manifest datasets/dissertation_v1/manifests/val_id.csv \
  --test-id-manifest datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv \
  --test-ood-manifest datasets/dissertation_v1/manifests/test_ood_balanced_by_subtype.csv \
  --device cuda \
  --seed 42 \
  --skip-existing
```

Do not rerun the full grid unless required by a missing output or examiner request.

## Avoid Committing Runs and Checkpoints

Generated run outputs belong under ignored directories:

- `reports/generated/`
- `reports/local_audits/`
- `runs/`

Before committing:

```bash
git status --short
git ls-files | grep -E '\.(pt|pth|ckpt|npz)$' || true
git ls-files | grep -E '(^|/)runs/' || true
```

The final PR should commit only compact Markdown/CSV summaries, scripts, tests, and curated figures already selected for the dissertation package.

## Optional SLURM Templates

Templates are provided under:

- `scripts/slurm/run_smoke_experiment.slurm`
- `scripts/slurm/run_grid_selected.slurm`

Edit partition, account, time, and module lines for the local server policy before submitting.

## Storage Warning

Git LFS data and generated reports can consume significant storage. Keep a separate scratch area for `reports/generated/` if the school server home quota is small, and copy only compact final outputs back into the repository.

## Exact Verification Commands

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
git diff --check
python scripts/build_dissertation_delivery_bundle.py --strict --out-dir reports/dissertation_final
python scripts/final_repository_audit.py
python scripts/validate_manifests.py --root-dir data \
  datasets/dissertation_v1/manifests/train_id.csv \
  datasets/dissertation_v1/manifests/test_ood_full.csv
```

Final expected handoff files:

- `docs/dissertation/final_evidence_index.md`
- `docs/dissertation/final_figure_shortlist.md`
- `docs/dissertation/final_table_shortlist.md`
- `docs/dissertation/reproducibility_runbook.md`
- `docs/dissertation/claims_and_limitations_matrix.md`
- `reports/dissertation_final/final_result_summary.md`
