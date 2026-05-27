# Dissertation / Final Report Outline

## Title

Unsupervised Out-of-Distribution Detection for Quality Control in Retinal Fundus Autofluorescence Imaging

## Writing Timeline

Use `docs/TEN_WEEK_DISSERTATION_PLAN_CN.md` as the week-by-week plan for dataset preparation, coding, evaluation, dissertation drafting, and final polishing.

## 1. Introduction

- Deep learning in ophthalmology.
- FAF imaging and clinical relevance.
- Silent failure caused by invalid inputs.
- Need for upstream OOD gatekeeper.
- Contributions.

## 2. Background and Literature Review

- OOD detection and one-class classification.
- Reconstruction-based methods: AE/VAE, spectral bias, identity shortcut.
- Feature-space methods: PaDiM, PatchCore, Knowledge Distillation, Normalizing Flows.
- CNN vs ViT inductive biases.
- Self-supervised synthetic corruption methods.
- Sim-to-real domain gap.
- Clinical metrics: AUROC limitations, AUPRC, FPR@95%TPR.

## 3. Problem Formulation

- Define in-distribution FAF.
- Define anomaly categories.
- Define score function `S(x)` and threshold `tau`.
- Define training and test assumptions.

## 4. Data

- Synthetic Heidelberg FAF training set.
- Valid real FAF test set.
- OOD test subsets.
- Manifest and split strategy.
- Privacy and ethical considerations.

## 5. Methods

### 5.1 Autoencoder baseline

- Architecture.
- Reconstruction score.
- Expected limitation.

### 5.2 PatchCore main model

- Backbone and feature layers.
- Patch feature extraction.
- Memory bank.
- Coreset.
- Nearest-neighbor anomaly score.
- Heatmap construction.

### 5.3 Layer ablation

- Shallow vs mid-level vs deep features.
- Hypothesis for sim-to-real balance.

## 6. Evaluation Protocol

- Test subsets.
- Metrics.
- Threshold selection.
- Per-category analysis.
- Statistical/reproducibility details.

## 7. Results

- Main quantitative table.
- Per-OOD-category table.
- ROC/PR curves.
- Score distributions.
- Heatmap examples.
- False positives and false negatives.

## 8. Discussion

- Why feature-space works better than reconstruction.
- Sim-to-real gap evidence.
- Clinical threshold trade-offs.
- Interpretability and heatmap limitations.
- Failure modes.

## 9. Limitations

- Dataset size.
- Synthetic training distribution.
- Scanner/device availability.
- Lack of prospective clinical deployment.
- Heatmap resolution.

## 10. Conclusion

- Summarize gatekeeper role.
- Best model and key metrics.
- Clinical safety implications.
- Future work.
