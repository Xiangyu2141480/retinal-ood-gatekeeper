# Codex Task Breakdown with Ready-to-Paste Prompts

Use these tasks as GitHub issues or Codex tasks. Do them in order.

---

## Phase 0 — Repository foundation

### Task 0.1 — Verify scaffold and improve setup

```text
Use the Codex Master Prompt.

Goal:
Inspect the repository scaffold and make it installable as an editable Python package.

Requirements:
- Ensure pyproject.toml package discovery works.
- Ensure pytest can run.
- Ensure README quickstart is accurate.
- Do not add data or model weights.

Validation:
- pip install -e ".[dev]"
- pytest
- ruff check .
```

### Task 0.2 — Add CI workflow

```text
Use the Codex Master Prompt.

Goal:
Add a GitHub Actions CI workflow for Python tests and ruff linting.

Requirements:
- Use Python 3.10 or 3.11.
- Install package with dev dependencies.
- Run pytest and ruff.
- Do not assume GPU.

Validation:
- Check YAML syntax.
- pytest locally if possible.
```

---

## Phase 1 — Data pipeline

### Task 1.1 — Implement manifest dataset

```text
Use the Codex Master Prompt.

Goal:
Implement ManifestImageDataset in src/retinal_ood/data/dataset.py.

Requirements:
- Read CSV manifest with columns image_path,label,split,source,ood_type.
- Validate required columns.
- Support relative paths with an optional root_dir.
- Return image tensor, label, and metadata dictionary.
- Raise clear errors for missing files or malformed manifests.
- Add tests using temporary toy images generated inside tests.

Validation:
- pytest tests/test_dataset_schema.py
- ruff check .
```

### Task 1.2 — Implement image transforms

```text
Use the Codex Master Prompt.

Goal:
Implement build_transforms in src/retinal_ood/data/transforms.py.

Requirements:
- Support image_size.
- Support grayscale_to_rgb.
- Support normalize modes: imagenet, minmax, none.
- Avoid random augmentation in test/evaluation transforms.
- Add tests for output shape and dtype.

Validation:
- pytest tests/test_transforms.py
- ruff check .
```

### Task 1.3 — Implement synthetic artifact generator

```text
Use the Codex Master Prompt.

Goal:
Implement scripts/generate_artifacts.py and src/retinal_ood/data/artifacts.py.

Requirements:
- Given an input manifest of valid images, create corrupted copies for sensory_artifact testing.
- Support text watermark, rectangle annotation, border/crop, gaussian noise, jpeg compression.
- Write a new OOD manifest with label=1 and ood_type=sensory_artifact.
- Never modify original files.
- Add deterministic seed support.

Validation:
- pytest tests/test_artifacts.py
- ruff check .
```

---

## Phase 2 — Evaluation utilities

### Task 2.1 — Implement metrics correctly

```text
Use the Codex Master Prompt.

Goal:
Implement metric utilities in src/retinal_ood/evaluation/metrics.py and thresholds.py.

Requirements:
- compute AUROC, AUPRC, FPR@95%TPR.
- positive label is anomaly/OOD = 1.
- Higher score means more anomalous.
- Handle ties and edge cases with clear errors.
- Add unit tests with hand-calculated arrays.

Validation:
- pytest tests/test_metrics.py
- ruff check .
```

### Task 2.2 — Implement score CSV and report table writer

```text
Use the Codex Master Prompt.

Goal:
Implement utilities that save per-image scores and aggregate metrics.

Requirements:
- Save scores.csv with image_path,label,ood_type,score,prediction,threshold.
- Save metrics.json.
- Save a markdown table for reports.
- Keep outputs under runs/<run_name>/.

Validation:
- pytest tests/test_reporting.py
- ruff check .
```

---

## Phase 3 — Baseline model

### Task 3.1 — Implement convolutional autoencoder baseline

```text
Use the Codex Master Prompt.

Goal:
Implement a simple convolutional autoencoder baseline.

Requirements:
- Training uses only label=0 ID images.
- Anomaly score is reconstruction error.
- Save model checkpoint and config.
- Provide train script scripts/train_autoencoder.py.
- Provide inference function returning image-level scores.
- Keep architecture simple and stable.

Validation:
- Add a smoke test with tiny generated images.
- pytest
- ruff check .
```

---

## Phase 4 — Main PatchCore model

### Task 4.1 — Implement feature extractor

```text
Use the Codex Master Prompt.

Goal:
Implement a timm/torchvision feature extractor that returns intermediate CNN layers.

Requirements:
- Support resnet50 layers: layer1, layer2, layer3, layer4.
- Freeze backbone weights.
- Return feature maps in a dictionary.
- Work on CPU for tests.
- Add shape tests using random tensors.

Validation:
- pytest tests/test_feature_extractor.py
- ruff check .
```

### Task 4.2 — Implement PatchCore memory bank

```text
Use the Codex Master Prompt.

Goal:
Implement PatchCoreDetector in src/retinal_ood/models/patchcore.py.

Requirements:
- fit() extracts patch features from ID training images.
- coreset subsampling is configurable; implement a simple deterministic greedy or random baseline first.
- predict_scores() returns image-level anomaly scores and optional patch heatmaps.
- Higher scores mean more anomalous.
- Save/load model state without saving raw images.
- Add smoke tests with synthetic tensors.

Validation:
- pytest tests/test_patchcore.py
- ruff check .
```

### Task 4.3 — Implement PatchCore training CLI

```text
Use the Codex Master Prompt.

Goal:
Implement scripts/train_patchcore.py.

Requirements:
- Read YAML config.
- Load train manifest.
- Fit PatchCore on label=0 images only.
- Save memory bank/model state under runs/<run_name>/.
- Save resolved config.
- Log number of images and patches.
- Do not require GPU.

Validation:
- Run a toy manifest smoke test.
- pytest
- ruff check .
```

---

## Phase 5 — Evaluation and visualization

### Task 5.1 — Implement evaluation CLI

```text
Use the Codex Master Prompt.

Goal:
Implement scripts/evaluate.py for trained detectors.

Requirements:
- Load model checkpoint.
- Load ID and OOD test manifests.
- Compute scores for all images.
- Compute AUROC, AUPRC, FPR@95%TPR and per-OOD metrics.
- Save scores.csv and metrics.json.
- Save ROC/PR plots.

Validation:
- Add tests for metrics/reporting.
- ruff check .
```

### Task 5.2 — Implement heatmap visualization

```text
Use the Codex Master Prompt.

Goal:
Implement heatmap generation and overlay utilities.

Requirements:
- Convert patch-level anomaly map to input image size.
- Save original, heatmap, and overlay images.
- Keep colorbar/normalization consistent within a run.
- Add a script option to save top-K false positives, false negatives, true positives, true negatives.

Validation:
- Add a test that output image files are created for toy arrays.
- pytest
- ruff check .
```

---

## Phase 6 — Research experiments

### Task 6.1 — Add layer ablation configs

```text
Use the Codex Master Prompt.

Goal:
Create YAML configs for PatchCore layer ablations.

Requirements:
- configs/patchcore_l1.yaml
- configs/patchcore_l2.yaml
- configs/patchcore_l3.yaml
- configs/patchcore_l4.yaml
- configs/patchcore_l23.yaml
- Same dataset paths and seed.
- Only layers/run_name differ.

Validation:
- Check configs parse.
- ruff check .
```

### Task 6.2 — Add experiment runner

```text
Use the Codex Master Prompt.

Goal:
Implement scripts/run_experiment_grid.py.

Requirements:
- Accept a list of config files.
- Train and evaluate each config.
- Collect metrics into reports/generated/experiment_summary.csv and .md.
- Fail fast by default, with optional --continue-on-error.

Validation:
- Smoke test with dummy/tiny configs.
- pytest
- ruff check .
```

### Task 6.3 — Add final report artifacts generator

```text
Use the Codex Master Prompt.

Goal:
Implement scripts/generate_report_tables.py.

Requirements:
- Read all metrics.json files under runs/.
- Produce markdown tables for dissertation.
- Include model, backbone, layers, AUROC, AUPRC, FPR@95%TPR.
- Include per-OOD-category table if available.

Validation:
- pytest tests/test_report_tables.py
- ruff check .
```

---

## Phase 7 — Thesis polish

### Task 7.1 — Write results documentation template

```text
Use the Codex Master Prompt.

Goal:
Create docs/RESULTS_INTERPRETATION_TEMPLATE_CN.md.

Requirements:
- Explain how to interpret AUROC, AUPRC, FPR@95%TPR.
- Include text templates for when PatchCore wins, when AE fails, and when layer ablation shows sim-to-real gap.
- Include limitations and future work.

Validation:
- Documentation only; no tests required.
```

### Task 7.2 — Repository cleanup

```text
Use the Codex Master Prompt.

Goal:
Final cleanup before GitHub submission.

Requirements:
- Ensure README is accurate.
- Ensure AGENTS.md is accurate.
- Ensure no private paths or data are committed.
- Ensure tests pass.
- Ensure docs explain how to reproduce experiments.

Validation:
- pytest
- ruff check .
- git status
```
