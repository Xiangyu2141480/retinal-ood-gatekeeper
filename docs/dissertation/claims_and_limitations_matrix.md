# Claims and Limitations Matrix

Use this table to keep dissertation wording precise and defensible.

| claim | supported? | evidence | limitation | wording to use |
|---|---|---|---|---|
| The project implements an upstream FAF OOD gatekeeper. | Yes | `figure_system_pipeline_overview.png`; `README.md` | It does not diagnose disease or grade pathology. | "An upstream unsupervised binary FAF OOD gatekeeper." |
| Training is ID-only. | Yes | `datasets/dissertation_v1/manifests/train_id.csv`; `result_provenance.md` | Dataset v1 ID images are synthetic-backed. | "Training uses ID-only FAF rows (`label=0`, `ood_type=id`)." |
| OOD data is evaluation/stress-test only. | Yes | `result_provenance.md`; `threshold_policy_sweep.csv` | Research thresholds use OOD labels for evaluation analysis only. | "OOD labels are used only for held-out evaluation grouping and stress-test reporting." |
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

Recommended limitation paragraph:

> These results are proof-of-concept stress-test evidence for an unsupervised binary FAF OOD gatekeeper. The ID test split uses synthetic FAF fallback rather than real clinical FAF validation, and the OOD groups are curated evaluation/stress examples rather than clinical prevalence estimates. The system is not a disease classifier and is not deployment-ready; real clinical FAF validation and prospective threshold calibration remain future work.
