# AGENTS.md — Codex Instructions for This Repository

## Project context

This repository implements a final-year project on **Unsupervised Out-of-Distribution Detection for Quality Control in Retinal Fundus Autofluorescence (FAF) Imaging**.

The system is not a disease classifier. It is an upstream quality-control gatekeeper:

```text
Clinical image -> OOD detector -> accept valid FAF OR reject invalid/OOD input
```

The main detector should learn from valid/normal FAF images only. The primary implementation target is PatchCore-style feature-space anomaly detection with mid-level CNN features. A simple autoencoder baseline is also required for comparison.

## Non-negotiable constraints

- Do not commit or create real medical image data in the repository.
- Do not hard-code private local file paths.
- Do not log patient IDs or sensitive metadata.
- Keep all data paths configurable through YAML configs or command-line arguments.
- Prefer deterministic, reproducible code: set seeds, record config, write metrics JSON.
- Every new feature should include tests where feasible.
- Keep implementation modular: data, models, evaluation, visualization, scripts.

## Preferred stack

- Python 3.10+
- PyTorch
- torchvision
- timm for pretrained backbones
- scikit-learn for metrics
- numpy, pandas, Pillow, opencv-python, matplotlib
- pytest for tests
- ruff/black for formatting

## Coding style

- Use type hints for public functions.
- Keep functions small and testable.
- Use dataclasses or config dictionaries for model settings.
- Avoid unnecessary global state.
- Use clear error messages for missing files, malformed manifests, or empty datasets.
- Write docstrings explaining the research meaning of key functions.

## Expected repository structure

```text
configs/                  # YAML experiment configs
docs/                     # project planning and thesis docs
scripts/                  # CLI entry points
src/retinal_ood/          # package source code
tests/                    # unit tests
reports/                  # generated markdown summaries, not raw data
runs/                     # generated outputs, gitignored
```

## Validation before finishing a task

Run:

```bash
pytest
ruff check .
```

If a task modifies model code, also run the smallest relevant smoke test. Do not claim success unless commands pass or you explicitly state what failed and why.

## Review guidelines

When reviewing pull requests, focus on:

- data leakage between train/validation/test sets,
- misuse of OOD labels during unsupervised training,
- metrics implemented incorrectly, especially FPR@95%TPR and AUPRC,
- privacy leaks in logs, configs, or manifests,
- brittle code that depends on a single local machine,
- heatmaps that are visually misleading or not aligned with input images,
- experiments that cannot be reproduced from configs.
