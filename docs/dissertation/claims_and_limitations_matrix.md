# Claims and Limitations Matrix

Use this table to keep dissertation wording precise and defensible.

| claim | supported? | evidence | limitation | wording to use |
|---|---|---|---|---|
| The project implements an upstream FAF OOD gatekeeper. | Yes | `figure_system_pipeline_overview.png`; `README.md` | It does not diagnose disease or grade pathology. | "An upstream unsupervised binary FAF OOD gatekeeper." |
| Training is ID-only. | Yes | `datasets/dissertation_v1/manifests/train_id.csv`; `result_provenance.md` | Dataset v1 ID images are synthetic-backed. | "Training uses ID-only FAF rows (`label=0`, `ood_type=id`)." |
| OOD data is evaluation/stress-test only. | Yes | `result_provenance.md`; `threshold_policy_sweep.csv` | Research thresholds use OOD labels for evaluation analysis only. | "OOD labels are used only for held-out evaluation grouping and stress-test reporting." |
| Phase 2 reason attribution is optional post-rejection explanation. | Yes | `docs/experiments/reason_attribution_method_comparison.md`; `best_method_summary.md` | It is supervised explanation and must not be described as Stage 1 OOD training. | "Stage 2 is an optional post-rejection reason-attribution layer." |
| Stage 1 remains ID-only after adding Phase 2. | Yes | `train_id.csv`; `reason_attribution_method_comparison.md`; `leakage_sanity_check.md` | OOD labels appear only in Stage 2 reason manifests and evaluation outputs. | "The Stage 1 gatekeeper is still ID-only and unsupervised; OOD labels are Stage 2 explanation targets only." |
| `linear_svm` is the best Stage 2 family method. | Yes | `best_method_summary.md`; `family_metrics_by_method.md` | High scores may be optimistic for generated artifact variants because parent hashes overlap across reason splits. | "`linear_svm` achieved family test accuracy 0.9881 and macro-F1 0.9901." |
| The non-oracle `hierarchical_classifier` is the best Stage 2 subtype method. | Yes | `best_method_summary.md`; `subtype_metrics_by_method.md`; `leakage_sanity_check.md` | The hierarchy routes by predicted family at test time, but parent-hash overlap remains a limitation. | "The non-oracle hierarchical classifier achieved subtype accuracy 0.9238 and macro-F1 0.9059." |
| Mahalanobis is the best quantitative model on dataset v1. | Yes | `metrics_by_scheme.csv`; `bootstrap_ci.csv` | Strong performance may reflect global feature shifts in current OOD groups. | "Mahalanobis is strongest quantitatively on dataset v1." |
| PatchCore L3 is the best localization companion. | Yes | `metrics_summary.csv`; heatmap figures; method disagreement outputs | PatchCore L3 is weaker than Mahalanobis on main quantitative metrics. | "PatchCore L3 is useful for localizable heatmap evidence." |
| Autoencoder is a weaker reconstruction baseline. | Yes | `metrics_by_scheme.csv`; `model_selection_summary.md` | It can still detect some severe or obvious artifacts. | "The autoencoder is a weaker baseline overall, not a failed method." |
| `val_id_quantile_95` is the best prototype threshold policy. | Yes | `threshold_policy_sweep.csv` | Clinical threshold selection requires real clinical FAF validation and operating-cost targets. | "Mahalanobis `val_id_quantile_95` is the best balanced prototype threshold." |
| Text watermark is the hardest subtype. | Yes | `subtype_influence.csv`; `method_disagreement_cases.md` | It is a generated/curated stress subtype, not a clinical prevalence estimate. | "Text watermark is the hardest current OOD subtype." |
| Artifact severity increases anomaly score monotonically. | No | `artifact_severity_stress.csv`; robustness findings | Some responses are non-monotonic or weak. | "Artifact severity response is mixed and not universally monotonic." |
| The system is deployment-ready. | No | Limitations in robustness docs | No real clinical FAF validation or prospective calibration. | "Proof-of-concept, not deployment-ready." |
| Synthetic fallback ID validates clinical FAF performance. | No | Dataset card; robustness key findings | Synthetic fallback is not real clinical FAF validation. | "`test_id_synthetic_fallback` is synthetic ID fallback, not real clinical FAF validation." |
| OOD stress-test frequencies estimate clinical prevalence. | No | Dataset taxonomy and limitations | OOD sets are curated for evaluation/stress. | "OOD stress-test, not clinical prevalence." |
| Future clinical validation is required. | Yes | Limitation sections and final evidence index | Requires institutional data governance and prospective evaluation. | "Requires real clinical FAF validation before clinical use." |
| Reason labels are clinical diagnoses. | No | `docs/experiments/reason_attribution_method_comparison.md`; `leakage_sanity_check.md` | Reason labels describe likely rejection causes, not disease states. | "Reason labels are likely explanations for rejected inputs, not clinical diagnoses." |

Recommended limitation paragraph:

> These results are proof-of-concept stress-test evidence for an unsupervised binary FAF OOD gatekeeper. The ID test split uses synthetic FAF fallback rather than real clinical FAF validation, and the OOD groups are curated evaluation/stress examples rather than clinical prevalence estimates. The system is not a disease classifier and is not deployment-ready; real clinical FAF validation and prospective threshold calibration remain future work.

Recommended Phase 2 limitation sentence:

> The optional Stage 2 reason-attribution module uses OOD reason labels only after rejection to produce likely explanations, not clinical diagnoses; its reason splits are image-path disjoint but not parent-image-hash disjoint for generated sensory artifacts, so high attribution scores should be interpreted cautiously.
