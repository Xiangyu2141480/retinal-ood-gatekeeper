# Dissertation Robustness Key Findings

## Scope and Guardrails

This findings pass summarizes actual PR #22 outputs for the unsupervised FAF OOD gatekeeper. The task remains: input image -> OOD gatekeeper -> ACCEPT valid FAF or REJECT invalid/OOD. Training is ID-only. OOD images are used only for evaluation grouping, stress tests, and research-threshold analysis. No disease classification or supervised multi-class OOD training is introduced.

Important limitations apply throughout: `test_id_synthetic_fallback.csv` is synthetic FAF fallback, not real clinical FAF validation. These experiments are proof-of-concept stress tests, not clinical deployment validation. OOD stress sets are not clinical prevalence estimates. Mahalanobis performance may be strong because many current OOD subtypes create global feature shifts. Real clinical FAF validation remains future work.

## A. Bootstrap Confidence Intervals

Mahalanobis feature distance has the strongest AUROC CI: 0.9724 [0.9638, 0.9800]. The next closest method is Global feature kNN at 0.9458 [0.9328, 0.9575]. These AUROC intervals do not overlap in this bootstrap run, so the ranking is stable for the current synthetic-backed evaluation set.

Mahalanobis also has the strongest AUPRC CI: 0.9975 [0.9967, 0.9983]. Global feature kNN is close at 0.9949 [0.9935, 0.9964]. The intervals are extremely narrow and nearly adjacent, so this should be described as a stable but small AUPRC margin rather than broad statistical separation.

For FPR@95%TPR, Mahalanobis has the lowest estimate: 0.2000 [0.1061, 0.2901]. Global feature kNN is second at 0.3800 [0.2500, 0.5193]. The FPR intervals are wider and overlap, so this result should be treated as directional evidence that Mahalanobis is best on threshold-sensitive safety, not as a hard significance claim.

Overall, Mahalanobis remains the best quantitative model under uncertainty. Ranking metrics are stable; threshold-dependent safety metrics are less stable and need real clinical calibration.

## B. Train-Size Sensitivity

Mahalanobis improves steadily as the ID training set grows: 50: AUROC 0.9389, FPR95 0.4067, OOD recall 0.6830, 100: AUROC 0.9500, FPR95 0.3000, OOD recall 0.7685, 250: AUROC 0.9571, FPR95 0.2867, OOD recall 0.8539, 500: AUROC 0.9694, FPR95 0.1933, OOD recall 0.8982, 700: AUROC 0.9724, FPR95 0.2000, OOD recall 0.9079. It is already strong with 50 ID images and improves most clearly through 500 images. The 500 to 700 step is small for AUROC, suggesting saturation in ranking performance, while FPR@95%TPR and threshold recall still fluctuate slightly.

Global feature kNN also improves with more ID data: 50: AUROC 0.8948, FPR95 0.5333, OOD recall 0.5861, 100: AUROC 0.9123, FPR95 0.5200, OOD recall 0.6273, 250: AUROC 0.9234, FPR95 0.4867, OOD recall 0.7564, 500: AUROC 0.9389, FPR95 0.3800, OOD recall 0.8133, 700: AUROC 0.9458, FPR95 0.3800, OOD recall 0.8024. Image statistics are less stable: 50: AUROC 0.8013, FPR95 0.8067, OOD recall 0.4697, 100: AUROC 0.8386, FPR95 0.7667, OOD recall 0.5952, 250: AUROC 0.7955, FPR95 0.7733, OOD recall 0.3285, 500: AUROC 0.7664, FPR95 0.7267, OOD recall 0.2848, 700: AUROC 0.7681, FPR95 0.7467, OOD recall 0.2994. The most data-efficient method in this sweep is Mahalanobis because it starts highest at 50 images and remains highest at every train size.

PatchCore was not retrained in this train-size sweep, so PR #22 cannot prove whether PatchCore needs more ID samples or patches. The runtime/resource table does show that PatchCore L3 and L2+L3 are heavier memory-bank methods, and the main results show they trail Mahalanobis quantitatively. Future work should vary PatchCore memory-bank size, coreset ratio, and layer selection under ID-only training.

## C. Threshold Policy Analysis

The research threshold at 95% TPR uses OOD labels and is for evaluation only. For Mahalanobis, it gives OOD recall 0.9503 but rejects 0.2000 of synthetic ID fallback images. Other methods require even higher ID rejection to force the same research recall, including PatchCore L3 at 0.6333, image statistics at 0.7467, and autoencoder at 0.9467 ID false rejection.

Deployment thresholds must be calibrated from validation ID scores only. For Mahalanobis, `val_id_quantile_95` gives ID false rejection 0.0467 and OOD recall 0.9079. `val_id_quantile_97_5` gives ID false rejection 0.0200 and OOD recall 0.8836. `val_id_quantile_99` gives ID false rejection 0.0133 and OOD recall 0.8727.

Policy summary from `threshold_policy_sweep.csv`:

| scheme | policy | kind | ID false rejection | OOD recall |
| --- | --- | --- | ---: | ---: |
| Mahalanobis feature | `research_95_tpr` | research | 0.2000 | 0.9503 |
| Mahalanobis feature | `val_id_quantile_95` | deployment | 0.0467 | 0.9079 |
| Mahalanobis feature | `val_id_quantile_97_5` | deployment | 0.0200 | 0.8836 |
| Mahalanobis feature | `val_id_quantile_99` | deployment | 0.0133 | 0.8727 |
| Mahalanobis feature | `val_id_quantile_99_5` | deployment | 0.0133 | 0.8448 |
| Mahalanobis feature | `fixed_id_rejection_5` | deployment | 0.0467 | 0.9079 |
| Mahalanobis feature | `fixed_id_rejection_10` | deployment | 0.0600 | 0.9176 |
| Mahalanobis feature | `fixed_id_rejection_20` | deployment | 0.1200 | 0.9388 |
| PatchCore L3 | `research_95_tpr` | research | 0.6333 | 0.9503 |
| PatchCore L3 | `val_id_quantile_95` | deployment | 0.0667 | 0.7097 |
| PatchCore L3 | `val_id_quantile_97_5` | deployment | 0.0400 | 0.6339 |
| PatchCore L3 | `val_id_quantile_99` | deployment | 0.0133 | 0.3994 |
| PatchCore L3 | `val_id_quantile_99_5` | deployment | 0.0000 | 0.2855 |
| PatchCore L3 | `fixed_id_rejection_5` | deployment | 0.0667 | 0.7097 |
| PatchCore L3 | `fixed_id_rejection_10` | deployment | 0.1267 | 0.7533 |
| PatchCore L3 | `fixed_id_rejection_20` | deployment | 0.1867 | 0.7879 |
| Autoencoder | `research_95_tpr` | research | 0.9467 | 0.9503 |
| Autoencoder | `val_id_quantile_95` | deployment | 0.0267 | 0.5588 |
| Autoencoder | `val_id_quantile_97_5` | deployment | 0.0133 | 0.4982 |
| Autoencoder | `val_id_quantile_99` | deployment | 0.0067 | 0.4527 |
| Autoencoder | `val_id_quantile_99_5` | deployment | 0.0067 | 0.4115 |
| Autoencoder | `fixed_id_rejection_5` | deployment | 0.0267 | 0.5588 |
| Autoencoder | `fixed_id_rejection_10` | deployment | 0.0667 | 0.6067 |
| Autoencoder | `fixed_id_rejection_20` | deployment | 0.1400 | 0.6448 |
| Image statistics | `research_95_tpr` | research | 0.7467 | 0.9503 |
| Image statistics | `val_id_quantile_95` | deployment | 0.0267 | 0.2994 |
| Image statistics | `val_id_quantile_97_5` | deployment | 0.0267 | 0.2327 |
| Image statistics | `val_id_quantile_99` | deployment | 0.0133 | 0.1012 |
| Image statistics | `val_id_quantile_99_5` | deployment | 0.0067 | 0.0921 |
| Image statistics | `fixed_id_rejection_5` | deployment | 0.0267 | 0.2994 |
| Image statistics | `fixed_id_rejection_10` | deployment | 0.0867 | 0.4594 |
| Image statistics | `fixed_id_rejection_20` | deployment | 0.1733 | 0.5745 |

The strongest deployment-style sensitivity point is Mahalanobis `fixed_id_rejection_20`, with ID false rejection 0.1200 and OOD recall 0.9388. For a thesis prototype recommendation, Mahalanobis `val_id_quantile_95` is the best balanced default because it is ID-calibrated, maintains low synthetic-ID rejection, and still recalls over 90% of OOD examples. A clinical system would choose a stricter or looser ID quantile only after real validation.

## D. Artifact Severity Stress Test

Severity response is not universally monotonic, and the dissertation should say that plainly. The most monotonic score responses are Mahalanobis feature on border_crop and Autoencoder on rectangle_annotation, both with Spearman 0.9685. The weakest response is Autoencoder on blur_artifact with Spearman -0.7506, where the autoencoder score decreases as blur increases.

By artifact type, rectangle annotation is detected earliest and most consistently: the average deployment reject rate across methods/severities is 0.7708, and several methods reject at annotation width 2. Border crop is also easy for Mahalanobis, with 75% rejection at crop 0.02 and 100% by crop 0.05. Gaussian noise becomes detectable as severity increases, especially for Mahalanobis and the autoencoder at high sigma.

Text watermark is the hardest artifact family in the stress table: mean deployment reject rate 0.0469, maximum 0.4167. JPEG compression is also weak for most methods, with average deployment reject rate 0.0667. These non-monotonic or weak curves should not be hidden; they are useful evidence that small/local artifacts remain hard.

This partly supports the literature review claim that autoencoders can miss localized artifacts. The autoencoder misses or weakly tracks blur, watermark, border crop, and JPEG compression, but it is strong on rectangle annotation and high Gaussian noise. The correct dissertation wording is therefore nuanced: AE reconstruction can miss subtle localized artifacts, not all localized artifacts.

## E. Method Disagreement and Failure Analysis

Mahalanobis and PatchCore L3 agree with the full method set on 1195 cases. Mahalanobis is correct while PatchCore L3 is wrong on 446 cases, mostly sensory artifacts such as blur, JPEG compression, Gaussian noise, and border crop. This supports Mahalanobis as the stronger quantitative gatekeeper in the current dataset.

Mahalanobis has 152 OOD false negatives at the selected deployment threshold. These are dominated by text watermark (129), followed by Gaussian noise (13), JPEG compression (8), and blur artifact (2). PatchCore L3 catches 111 of the Mahalanobis false negatives, mostly text watermark cases. These are good dissertation discussion examples because they show the value of a localizable complementary method even when it is weaker overall.

There are 7 Mahalanobis false positives on synthetic ID fallback FAF images: `id_test_synthetic_fallback_000040.png`, `000103.png`, `000129.png`, `000140.png`, `000053.png`, `000072.png`, and `000117.png`. These should be described cautiously because the ID test set is synthetic fallback rather than clinical FAF. The best discussion figures are `figure_method_disagreement_examples.png`, `figure_false_negative_ood_examples.png`, and `figure_false_positive_id_examples.png`.

## F. Feature-Space Visualization

The PCA visualization explains why Mahalanobis performs well: semantic outliers and many modality shifts create large global feature-space shifts. CIFAR natural images are farthest from ID with centroid distance 12.6234 and mean Mahalanobis score 288.5050. OCT screenshots are also far from ID with distance 7.2531.

Colour fundus is closer to ID than OCT and CIFAR, with centroid distance 3.7114, but it is still clearly shifted relative to the ID centroid and Mahalanobis detects it well. Text watermark is closest to ID in PCA space, with centroid distance 0.0811 and mean score 32.3415. This matches the failure analysis: subtle local watermark artifacts look globally ID-like and are the hardest Mahalanobis subtype.

## G. Runtime and Resource Analysis

The runtime table is a local smoke scoring measurement, not a hardware-independent deployment benchmark. Fit time was not remeasured; existing PR #21 training artifacts were reused. The fastest measured scorer is Autoencoder at 10.9943 ms/image on 22 local images. The lightest artifact is Image statistics at 0.0013 MB.

The best performance/resource trade-off is Mahalanobis: AUROC 0.9724, FPR@95%TPR 0.2000, artifact size 6.2061 MB, and 39.3631 ms/image. PatchCore L3 is slower and heavier (16.0489 MB, 55.8631 ms/image) but provides patch heatmaps/localization. A software prototype should use Mahalanobis as the binary gatekeeper and PatchCore L3 as an optional explanation/localization companion.

## H. Overall Dissertation Takeaway

The strongest model is Mahalanobis feature distance. It wins the main comparison, bootstrap AUROC/AUPRC, FPR@95%TPR, train-size sensitivity, threshold trade-off, and practical performance/resource balance.

The most interpretable/localizable model is PatchCore L3 because it produces patch-level heatmaps and catches many Mahalanobis misses, especially text watermark cases. It is not the best quantitative gatekeeper in this dataset, but it is valuable for dissertation evidence and qualitative explanation.

The safest threshold recommendation is not the research 95% TPR threshold. The thesis should recommend validation-ID calibration, with Mahalanobis `val_id_quantile_95` as the balanced prototype threshold and higher ID quantiles for settings that prioritize fewer false ID rejections.

The biggest limitation is the lack of real clinical FAF validation. The current ID test set is synthetic fallback, OOD subtypes are curated stress/evaluation examples, and OOD stress prevalence is not clinical prevalence. Future work should collect real FAF validation data, evaluate prospective calibration, test additional real-world artifacts and acquisition devices, and explore Mahalanobis plus PatchCore hybrid decision/explanation workflows.
