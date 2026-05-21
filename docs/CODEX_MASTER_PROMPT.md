# Codex Master Prompt

Copy this into the beginning of Codex tasks when working on this repository.

```text
You are helping implement a final-year project repository called retinal-ood-gatekeeper.

Project context:
- This is an unsupervised OOD detection system for quality control in retinal Fundus Autofluorescence (FAF) imaging.
- It is not a disease classifier.
- The system trains only on valid/normal FAF images and rejects invalid inputs before a downstream diagnostic model.
- Main model target: PatchCore-style feature-space anomaly detection using pretrained CNN mid-level features, especially layer2/layer3.
- Baseline: convolutional autoencoder reconstruction error.
- Evaluation: AUROC, AUPRC, FPR@95%TPR, confusion matrix, per-OOD-type metrics, heatmap examples.
- OOD categories: modality shift, sensory artifacts, semantic outliers, scanner shift.

Repository constraints:
- Never add real clinical data, private paths, credentials, or patient identifiers.
- Keep all data paths configurable through YAML and CLI arguments.
- Use modular Python with type hints.
- Add tests for metrics, thresholding, manifests, and any deterministic utility.
- Run pytest and ruff before finishing.
- Preserve the repository structure in README and AGENTS.md.

When implementing a task:
1. Inspect existing files first.
2. Make the smallest coherent change.
3. Add or update tests.
4. Update docs if behavior changes.
5. Return a concise summary of changed files and validation commands.
```
```

## Prompt template for each new Codex task

```text
Task title: <short title>

Goal:
<what to implement>

Files likely involved:
- <file 1>
- <file 2>

Requirements:
- <functional requirement>
- <edge case>
- <privacy/reproducibility constraint>

Validation:
- pytest <specific tests>
- ruff check .

Do not:
- commit data or model weights
- hard-code local paths
- change unrelated files
```
