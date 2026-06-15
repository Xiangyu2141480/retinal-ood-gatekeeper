# Dissertation Results Interpretation

## Project Framing

This project implements an unsupervised retinal FAF out-of-distribution gatekeeper. The system output is binary:

```text
input image -> OOD gatekeeper -> ACCEPT valid FAF / REJECT invalid or OOD
```

It is not a disease classifier. The `ood_type` and `ood_subtype` fields are used for evaluation grouping, stress testing, and figure stratification only. Training remains ID-only using `label=0` / `ood_type=id`; OOD images are evaluation-only.

The current ID test set is `test_id_synthetic_fallback.csv`. This is useful for reproducible method comparison, but it is synthetic fallback ID data and must not be presented as real clinical FAF validation.

## Why Multiple Schemes Were Compared

The dissertation comparison deliberately spans several unsupervised OOD paradigms:

- Image statistics: a weak sanity baseline using low-level intensity, histogram, entropy, and border/artifact features.
- Autoencoder: a pixel-space reconstruction baseline aligned with reconstruction-based OOD literature.
- Global feature kNN: a simple pretrained CNN feature-space nearest-neighbor baseline.
- Mahalanobis feature distance: a statistical feature-space baseline related to PaDiM-style modelling.
- PatchCore: the main patch-level feature-memory family, tested through layer ablation.

This shows that the work is not a single-model demonstration. It compares reconstruction, low-level statistics, global feature distances, statistical feature distances, and patch-level feature memory banks under the same ID-only training rule.

## Workload Demonstrated

The multi-scheme package reports 8 completed schemes, 4 completed PatchCore variants, 3 evaluation subsets, 3 OOD types, and 11 OOD subtypes. Across completed scheme/evaluation runs, 36,000 image evaluations were aggregated into compact result tables and dissertation figures.

PatchCore L1 is listed as runtime-limited rather than imputed. At 224 px, the shallow layer creates a dense patch grid and the current implementation gathers patch embeddings before coreset sampling, so a full L1 run would be memory-heavy on CPU. The dissertation should present this transparently as a practical limitation.

## Main Result

On the primary balanced-by-subtype evaluation set, the Mahalanobis feature baseline is the strongest completed scheme by all three headline metrics:

| Metric | Best scheme | Value |
| --- | --- | --- |
| AUROC | Mahalanobis feature | 0.9724 |
| AUPRC | Mahalanobis feature | 0.9975 |
| FPR@95%TPR | Mahalanobis feature | 0.2000 |

The global feature kNN baseline is second by AUROC and AUPRC on the primary split. This suggests that the synthetic/final FAF comparison strongly rewards pretrained global feature distances. That is a meaningful result, not a failure: it shows the importance of comparing several unsupervised gatekeeper families rather than assuming PatchCore will dominate every dataset.

Among completed PatchCore variants, PatchCore L3 remains the best PatchCore model on the primary split:

| PatchCore variant | AUROC | AUPRC | FPR@95%TPR |
| --- | ---: | ---: | ---: |
| PatchCore L2 | 0.8280 | 0.9828 | 0.7867 |
| PatchCore L3 | 0.8819 | 0.9880 | 0.6333 |
| PatchCore L4 | 0.7502 | 0.9692 | 0.8467 |
| PatchCore L2+L3 | 0.8722 | 0.9874 | 0.6867 |

## Safety Interpretation

The results are proof-of-concept, not deployment-ready. Even PatchCore L3, the best completed PatchCore variant, has FPR@95%TPR = 0.6333 on the primary split. That is high: achieving 95% OOD recall would falsely reject many ID fallback images.

The validation-ID threshold produces lower ID false rejection rates, but it also sacrifices OOD recall for several schemes. For PatchCore L3, the deployment-threshold OOD recall is 0.7097 with an ID false rejection rate of 0.0667. Mahalanobis improves this trade-off in the current synthetic-backed evaluation, but real clinical FAF validation and threshold calibration remain future work.

## Literature Alignment

The experiment matrix maps directly to the literature review:

- The autoencoder tests reconstruction-based limitations, including smoothing over local artifacts.
- Global feature kNN tests whether pretrained embedding distance is sufficient without patch memory.
- Mahalanobis tests a statistical feature-distance assumption related to PaDiM-style modelling.
- PatchCore layer ablation tests the mid-level feature hypothesis for local patch memory banks.
- FPR@95%TPR and AUPRC complement AUROC by exposing safety and class-imbalance behavior.
- Heatmap examples provide qualitative interpretability for the selected PatchCore model.

## Recommended Dissertation Framing

Present Mahalanobis feature distance as the best overall completed scheme on the current finalized synthetic-backed evaluation matrix. Present PatchCore L3 as the best completed PatchCore variant and the most interpretable patch-localization method. Emphasize that the project contribution is the ID-only FAF OOD gatekeeper evaluation framework, not a claim of clinical deployment readiness.
